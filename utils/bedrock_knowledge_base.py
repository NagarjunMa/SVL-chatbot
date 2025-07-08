"""
Bedrock Knowledge Base Manager - Replaces FAISS-based system
Uses AWS Bedrock Knowledge Base with OpenSearch for vector storage
"""

import boto3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

from utils.logger import get_logger
from utils.observability import observability, trace_function

logger = get_logger("bedrock_knowledge_base")

class BedrockKnowledgeBaseManager:
    """
    AWS Bedrock Knowledge Base manager for SVL chatbot
    Replaces the custom FAISS-based knowledge base system
    """
    
    @trace_function("bedrock_kb_init")
    def __init__(self, 
                 knowledge_base_id: str = "7AKY8ACDXE",  # Updated with actual KB ID
                 guardrail_id: str = "7lf4bwfa7ncl",
                 aws_region: str = "us-east-1"):
        """
        Initialize Bedrock Knowledge Base Manager
        
        Args:
            knowledge_base_id: AWS Bedrock Knowledge Base ID
            guardrail_id: AWS Bedrock Guardrails ID
            aws_region: AWS region
        """
        self.knowledge_base_id = knowledge_base_id
        self.guardrail_id = guardrail_id
        self.aws_region = aws_region
        
        # Initialize AWS clients
        with observability.trace_operation("init_aws_clients", 
                                         knowledge_base_id=knowledge_base_id,
                                         region=aws_region):
            self.bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=aws_region)
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
            self.bedrock_agent = boto3.client('bedrock-agent', region_name=aws_region)
        
        # Query statistics
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0,
            "last_query_time": None
        }
        
        logger.info("Initialized Bedrock clients for region %s", aws_region)
        logger.info("BedrockKnowledgeBaseManager initialized")
        logger.info("Knowledge Base ID: %s", knowledge_base_id)
        logger.info("Guardrail ID: %s", guardrail_id)
    
    @trace_function("check_kb_status")
    def is_ready(self) -> bool:
        """Check if the knowledge base is ready for queries"""
        try:
            with observability.trace_operation("get_knowledge_base_status",
                                             knowledge_base_id=self.knowledge_base_id) as span:
                response = self.bedrock_agent.get_knowledge_base(
                    knowledgeBaseId=self.knowledge_base_id
                )
                
                status = response['knowledgeBase']['status']
                span['metadata']['kb_status'] = status
                span['metadata']['kb_name'] = response['knowledgeBase']['name']
                
                # Log knowledge base status check
                observability.log_knowledge_base_operation(
                    operation_type='status_check',
                    knowledge_base_id=self.knowledge_base_id,
                    query='status_check',
                    results=[],
                    metrics={'status': status}
                )
                
                return status == 'ACTIVE'
                
        except Exception as e:
            logger.error(f"Failed to check knowledge base status: {e}")
            observability.log_error(e, {
                'operation': 'knowledge_base_status_check',
                'knowledge_base_id': self.knowledge_base_id
            })
            return False
    
    @trace_function("query_knowledge_base")
    def query_knowledge_base(self, 
                           query: str, 
                           max_results: int = 5, 
                           similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Query the knowledge base using semantic search
        
        Args:
            query: User query text
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Dict containing search results and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            with observability.trace_operation("bedrock_kb_query",
                                             query=query[:100],
                                             max_results=max_results,
                                             similarity_threshold=similarity_threshold) as span:
                
                logger.info(f"Querying Bedrock KB: '{query[:50]}...'")
                
                # Prepare query parameters
                query_params = {
                    'knowledgeBaseId': self.knowledge_base_id,
                    'retrievalQuery': {
                        'text': query
                    },
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': max_results,
                            'overrideSearchType': 'SEMANTIC'
                        }
                    }
                }
                
                span['metadata']['query_params'] = {
                    'max_results': max_results,
                    'search_type': 'SEMANTIC',
                    'knowledge_base_id': self.knowledge_base_id
                }
                
                # Execute knowledge base query
                response = self.bedrock_agent_runtime.retrieve(
                    **query_params
                )
                
                # Process results
                results = []
                for result in response.get('retrievalResults', []):
                    score = result.get('score', 0.0)
                    
                    # Apply similarity threshold
                    if score >= similarity_threshold:
                        processed_result = {
                            'text': result.get('content', {}).get('text', ''),
                            'score': score,
                            'source': result.get('location', {}).get('s3Location', {}).get('uri', 'unknown'),
                            'metadata': result.get('metadata', {})
                        }
                        results.append(processed_result)
                
                # Calculate metrics
                end_time = datetime.now(timezone.utc)
                duration_seconds = (end_time - start_time).total_seconds()
                
                # Update statistics
                self.query_stats["total_queries"] += 1
                self.query_stats["successful_queries"] += 1
                self.query_stats["last_query_time"] = end_time.isoformat()
                
                if self.query_stats["total_queries"] > 0:
                    # Update rolling average
                    current_avg = self.query_stats["average_response_time"]
                    total_queries = self.query_stats["total_queries"]
                    self.query_stats["average_response_time"] = (
                        (current_avg * (total_queries - 1) + duration_seconds) / total_queries
                    )
                
                span['metadata']['results_count'] = len(results)
                span['metadata']['duration_seconds'] = duration_seconds
                
                logger.info(f"Bedrock KB query successful: {len(results)} results in {duration_seconds:.2f}s")
                
                # Log detailed knowledge base operation
                observability.log_knowledge_base_operation(
                    operation_type='query',
                    knowledge_base_id=self.knowledge_base_id,
                    query=query,
                    results=results,
                    metrics={
                        'duration_seconds': duration_seconds,
                        'total_results_before_threshold': len(response.get('retrievalResults', [])),
                        'results_after_threshold': len(results),
                        'similarity_threshold': similarity_threshold,
                        'max_results': max_results
                    }
                )
                
                # Log vector search details if available
                if results:
                    # Extract vector information for logging
                    vector_metrics = {
                        'search_type': 'semantic',
                        'total_vectors_searched': len(response.get('retrievalResults', [])),
                        'vectors_above_threshold': len(results),
                        'highest_score': max(r['score'] for r in results) if results else 0,
                        'lowest_score': min(r['score'] for r in results) if results else 0,
                        'average_score': sum(r['score'] for r in results) / len(results) if results else 0
                    }
                    
                    observability.log_vector_search(
                        search_type='bedrock_semantic',
                        query_vector=[],  # Bedrock handles embedding internally
                        search_params={
                            'numberOfResults': max_results,
                            'searchType': 'SEMANTIC',
                            'similarity_threshold': similarity_threshold
                        },
                        results=results,
                        opensearch_metrics=vector_metrics
                    )
                
                return {
                    "status": "success",
                    "total_results": len(results),
                    "results": results,
                    "query": query,
                    "duration_seconds": duration_seconds,
                    "knowledge_base_id": self.knowledge_base_id
                }
                
        except Exception as e:
            # Update failure statistics
            self.query_stats["total_queries"] += 1
            self.query_stats["failed_queries"] += 1
            
            logger.error(f"Bedrock KB query failed: {e}")
            
            observability.log_error(e, {
                'operation': 'bedrock_knowledge_base_query',
                'query': query[:100],
                'knowledge_base_id': self.knowledge_base_id,
                'max_results': max_results,
                'similarity_threshold': similarity_threshold
            })
            
            return {
                "status": "error",
                "error": str(e),
                "total_results": 0,
                "results": [],
                "query": query,
                "knowledge_base_id": self.knowledge_base_id
            }
    
    @trace_function("generate_with_knowledge")
    def generate_with_knowledge(self, 
                              user_query: str, 
                              max_tokens: int = 500,
                              temperature: float = 0.1) -> Dict[str, Any]:
        """
        Generate response using knowledge base with Bedrock LLM
        
        Args:
            user_query: User's question
            max_tokens: Maximum tokens to generate
            temperature: Response randomness (0.0 = deterministic)
            
        Returns:
            Dict containing generated response and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            with observability.trace_operation("retrieve_and_generate",
                                             query=user_query[:100],
                                             max_tokens=max_tokens,
                                             temperature=temperature) as span:
                
                logger.info(f"Generating response with knowledge for: '{user_query[:50]}...'")
                
                # First, retrieve relevant documents
                kb_results = self.query_knowledge_base(user_query, max_results=5)
                
                if kb_results["status"] != "success" or not kb_results["results"]:
                    logger.warning("No knowledge base results found")
                
                # Prepare request for Bedrock RetrieveAndGenerate
                request_params = {
                    'input': {
                        'text': user_query
                    },
                    'retrieveAndGenerateConfiguration': {
                        'type': 'KNOWLEDGE_BASE',
                        'knowledgeBaseConfiguration': {
                            'knowledgeBaseId': self.knowledge_base_id,
                            'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0',
                            'retrievalConfiguration': {
                                'vectorSearchConfiguration': {
                                    'numberOfResults': 5
                                }
                            },
                            'generationConfiguration': {
                                'inferenceConfig': {
                                    'textInferenceConfig': {
                                        'maxTokens': max_tokens,
                                        'temperature': temperature
                                    }
                                }
                            }
                        }
                    }
                }
                
                # Add guardrails if configured
                if self.guardrail_id:
                    request_params['retrieveAndGenerateConfiguration']['knowledgeBaseConfiguration']['guardrailConfiguration'] = {
                        'guardrailId': self.guardrail_id,
                        'guardrailVersion': 'DRAFT'
                    }
                
                span['metadata']['request_params'] = {
                    'model': 'amazon.nova-pro-v1:0',
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'guardrail_id': self.guardrail_id,
                    'knowledge_base_id': self.knowledge_base_id
                }
                
                # Log the Bedrock operation
                observability.log_bedrock_operation(
                    operation_type='retrieve_and_generate',
                    model_id='amazon.nova-pro-v1:0',
                    input_data={
                        'text': user_query,
                        'max_tokens': max_tokens,
                        'temperature': temperature
                    }
                )
                
                # Execute RetrieveAndGenerate
                response = self.bedrock_agent_runtime.retrieve_and_generate(
                    **request_params
                )
                
                # Extract response details
                generated_text = response.get('output', {}).get('text', '')
                citations = response.get('citations', [])
                guardrail_action = response.get('guardrailAction')
                
                # Process citations/sources
                sources = []
                for citation in citations:
                    for reference in citation.get('references', []):
                        source_info = reference.get('location', {}).get('s3Location', {})
                        sources.append({
                            'uri': source_info.get('uri', 'unknown'),
                            'text': reference.get('content', {}).get('text', '')[:200]
                        })
                
                # Calculate metrics
                end_time = datetime.now(timezone.utc)
                duration_seconds = (end_time - start_time).total_seconds()
                
                span['metadata']['response_length'] = len(generated_text)
                span['metadata']['sources_count'] = len(sources)
                span['metadata']['guardrail_action'] = guardrail_action
                span['metadata']['duration_seconds'] = duration_seconds
                
                # Log guardrails activation if applicable
                if guardrail_action and guardrail_action != 'NONE':
                    observability.log_guardrails_activation(
                        guardrail_id=self.guardrail_id,
                        input_text=user_query,
                        action=guardrail_action,
                        reason='bedrock_guardrails_check',
                        confidence=1.0
                    )
                
                # Log the Bedrock response
                observability.log_bedrock_operation(
                    operation_type='retrieve_and_generate_response',
                    model_id='amazon.nova-pro-v1:0',
                    input_data={'query': user_query},
                    output_data={
                        'text': generated_text,
                        'sources_count': len(sources),
                        'guardrail_action': guardrail_action
                    },
                    metrics={
                        'duration_seconds': duration_seconds,
                        'response_length': len(generated_text),
                        'citations_count': len(citations)
                    }
                )
                
                logger.info(f"Response generated successfully with {len(sources)} sources")
                
                return {
                    "status": "success",
                    "response": generated_text,
                    "sources": sources,
                    "guardrail_action": guardrail_action,
                    "duration_seconds": duration_seconds,
                    "knowledge_base_id": self.knowledge_base_id
                }
                
        except Exception as e:
            logger.error(f"Generate with knowledge failed: {e}")
            
            observability.log_error(e, {
                'operation': 'retrieve_and_generate',
                'query': user_query[:100],
                'knowledge_base_id': self.knowledge_base_id,
                'max_tokens': max_tokens,
                'temperature': temperature
            })
            
            return {
                "status": "error",
                "error": str(e),
                "response": "",
                "sources": [],
                "knowledge_base_id": self.knowledge_base_id
            }
    
    @trace_function("get_kb_stats")
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base usage statistics"""
        with observability.trace_operation("get_knowledge_base_stats") as span:
            try:
                # Get knowledge base details
                kb_response = self.bedrock_agent.get_knowledge_base(
                    knowledgeBaseId=self.knowledge_base_id
                )
                
                # Get data sources
                ds_response = self.bedrock_agent.list_data_sources(
                    knowledgeBaseId=self.knowledge_base_id
                )
                
                data_sources = []
                for ds in ds_response.get('dataSourceSummaries', []):
                    data_sources.append({
                        'id': ds['dataSourceId'],
                        'name': ds['name'],
                        'status': ds['status'],
                        'updated': ds['updatedAt']
                    })
                
                stats = {
                    "query_statistics": self.query_stats,
                    "configuration": {
                        "knowledge_base_id": self.knowledge_base_id,
                        "guardrail_id": self.guardrail_id,
                        "region": self.aws_region,
                        "default_max_results": 5,
                        "default_similarity_threshold": 0.7
                    },
                    "service_status": {
                        "status": "available",
                        "knowledge_base": {
                            "id": kb_response['knowledgeBase']['knowledgeBaseId'],
                            "name": kb_response['knowledgeBase']['name'],
                            "status": kb_response['knowledgeBase']['status'],
                            "created": kb_response['knowledgeBase']['createdAt'],
                            "updated": kb_response['knowledgeBase']['updatedAt']
                        },
                        "data_sources": data_sources,
                        "total_data_sources": len(data_sources)
                    }
                }
                
                span['metadata']['total_queries'] = self.query_stats["total_queries"]
                span['metadata']['data_sources_count'] = len(data_sources)
                
                return stats
                
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                observability.log_error(e, {
                    'operation': 'get_knowledge_base_stats',
                    'knowledge_base_id': self.knowledge_base_id
                })
                
                return {
                    "status": "error",
                    "message": f"Failed to retrieve stats: {str(e)}"
                }
    
    @trace_function("check_kb_health")
    def check_knowledge_base_status(self) -> Dict[str, Any]:
        """Check detailed knowledge base health status"""
        try:
            with observability.trace_operation("detailed_health_check") as span:
                # Get knowledge base details
                kb_response = self.bedrock_agent.get_knowledge_base(
                    knowledgeBaseId=self.knowledge_base_id
                )
                
                # Get data sources
                ds_response = self.bedrock_agent.list_data_sources(
                    knowledgeBaseId=self.knowledge_base_id
                )
                
                data_sources = []
                for ds in ds_response.get('dataSourceSummaries', []):
                    data_sources.append({
                        'id': ds['dataSourceId'],
                        'name': ds['name'],
                        'status': ds['status'],
                        'updated': ds['updatedAt']
                    })
                
                health_status = {
                    "status": "available",
                    "knowledge_base": {
                        "id": kb_response['knowledgeBase']['knowledgeBaseId'],
                        "name": kb_response['knowledgeBase']['name'],
                        "status": kb_response['knowledgeBase']['status'],
                        "created": kb_response['knowledgeBase']['createdAt'],
                        "updated": kb_response['knowledgeBase']['updatedAt']
                    },
                    "data_sources": data_sources,
                    "total_data_sources": len(data_sources)
                }
                
                span['metadata']['kb_status'] = kb_response['knowledgeBase']['status']
                span['metadata']['data_sources_count'] = len(data_sources)
                
                return health_status
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            observability.log_error(e, {
                'operation': 'knowledge_base_health_check',
                'knowledge_base_id': self.knowledge_base_id
            })
            
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }

# Factory function for easy instantiation
def get_bedrock_knowledge_base(knowledge_base_id: str = None, 
                              guardrail_id: str = None) -> BedrockKnowledgeBaseManager:
    """Factory function to create BedrockKnowledgeBaseManager instance"""
    return BedrockKnowledgeBaseManager(
        knowledge_base_id=knowledge_base_id,
        guardrail_id=guardrail_id
    ) 