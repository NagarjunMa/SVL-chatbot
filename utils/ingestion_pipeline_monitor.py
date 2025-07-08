"""
Comprehensive Ingestion Pipeline Monitoring
Tracks every stage of document processing from S3 to vector storage in OpenSearch
"""

import boto3
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import requests
from requests_aws4auth import AWS4Auth

from utils.observability import observability, trace_function
from utils.logger import get_logger

logger = get_logger("ingestion_pipeline_monitor")

class IngestionPipelineMonitor:
    """Monitor and log every stage of the document ingestion pipeline"""
    
    def __init__(self, knowledge_base_id: str = "7AKY8ACDXE", 
                 data_source_id: str = "FQ0XNYFBAB",
                 s3_bucket: str = "svl-chatbot-knowledge-base-20250708",
                 opensearch_endpoint: str = "https://search-svl-chatbot-vector-search-c2z3p3dkdzmv53gfhucn4uw7cy.us-east-1.es.amazonaws.com"):
        
        self.knowledge_base_id = knowledge_base_id
        self.data_source_id = data_source_id
        self.s3_bucket = s3_bucket
        self.opensearch_endpoint = opensearch_endpoint
        
        # Initialize AWS clients
        self.bedrock_agent = boto3.client('bedrock-agent', region_name='us-east-1')
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.s3_client = boto3.client('s3', region_name='us-east-1')
        
        # OpenSearch authentication
        session = boto3.Session()
        credentials = session.get_credentials()
        self.opensearch_auth = AWS4Auth(
            credentials.access_key, 
            credentials.secret_key, 
            'us-east-1', 
            'es', 
            session_token=credentials.token
        )
    
    @trace_function("monitor_ingestion_job")
    def monitor_ingestion_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor a complete ingestion job from start to finish"""
        
        # Start monitoring trace
        trace = observability.start_trace(
            operation_name="ingestion_pipeline_monitoring",
            session_id=f"ingestion_{job_id}"
        )
        
        try:
            trace.add_metadata('job_id', job_id)
            trace.add_metadata('knowledge_base_id', self.knowledge_base_id)
            trace.add_metadata('data_source_id', self.data_source_id)
            
            logger.info(f"🔍 Starting comprehensive monitoring of ingestion job: {job_id}")
            
            # Stage 1: Monitor S3 document retrieval
            with observability.trace_operation("s3_document_analysis") as span:
                s3_metrics = self._analyze_s3_documents()
                span['metadata'].update(s3_metrics)
                
                observability.log_ingestion_pipeline(
                    stage='s3_document_analysis',
                    document_info=s3_metrics,
                    metrics={
                        'total_documents': s3_metrics.get('total_documents', 0),
                        'total_size_bytes': s3_metrics.get('total_size_bytes', 0)
                    }
                )
            
            # Stage 2: Monitor ingestion job progress
            job_complete = False
            ingestion_metrics = {}
            
            while not job_complete:
                with observability.trace_operation("ingestion_job_status_check") as span:
                    job_status = self._get_ingestion_job_status(job_id)
                    span['metadata'].update(job_status)
                    
                    observability.log_ingestion_pipeline(
                        stage='ingestion_job_progress',
                        document_info={'job_id': job_id},
                        metrics=job_status
                    )
                    
                    if job_status['status'] in ['COMPLETE', 'FAILED']:
                        job_complete = True
                        ingestion_metrics = job_status
                    else:
                        time.sleep(10)  # Wait before checking again
            
            if ingestion_metrics['status'] == 'COMPLETE':
                # Stage 3: Analyze text extraction and chunking
                with observability.trace_operation("text_processing_analysis") as span:
                    text_metrics = self._analyze_text_processing(ingestion_metrics)
                    span['metadata'].update(text_metrics)
                    
                    observability.log_ingestion_pipeline(
                        stage='text_extraction_and_chunking',
                        document_info=text_metrics,
                        metrics={
                            'documents_processed': text_metrics.get('documents_processed', 0),
                            'chunks_created': text_metrics.get('estimated_chunks', 0)
                        }
                    )
                
                # Stage 4: Monitor embedding generation
                with observability.trace_operation("embedding_generation_analysis") as span:
                    embedding_metrics = self._analyze_embedding_generation()
                    span['metadata'].update(embedding_metrics)
                    
                    observability.log_ingestion_pipeline(
                        stage='embedding_generation',
                        document_info=embedding_metrics,
                        metrics={
                            'embedding_model': 'amazon.titan-embed-text-v2:0',
                            'vector_dimension': 1024,
                            'total_embeddings': embedding_metrics.get('total_embeddings', 0)
                        }
                    )
                
                # Stage 5: Monitor vector storage in OpenSearch
                with observability.trace_operation("vector_storage_analysis") as span:
                    vector_metrics = self._analyze_vector_storage()
                    span['metadata'].update(vector_metrics)
                    
                    observability.log_ingestion_pipeline(
                        stage='vector_storage',
                        document_info=vector_metrics,
                        metrics={
                            'vectors_stored': vector_metrics.get('vectors_stored', 0),
                            'index_size_mb': vector_metrics.get('index_size_mb', 0),
                            'storage_engine': 'faiss'
                        }
                    )
                
                # Stage 6: End-to-end validation
                with observability.trace_operation("pipeline_validation") as span:
                    validation_metrics = self._validate_pipeline_completion()
                    span['metadata'].update(validation_metrics)
                    
                    observability.log_ingestion_pipeline(
                        stage='pipeline_validation',
                        document_info=validation_metrics,
                        metrics={
                            'validation_successful': validation_metrics.get('validation_successful', False),
                            'searchable_documents': validation_metrics.get('searchable_documents', 0)
                        }
                    )
            
            # Complete monitoring
            final_metrics = {
                'job_id': job_id,
                'final_status': ingestion_metrics['status'],
                's3_metrics': s3_metrics,
                'ingestion_metrics': ingestion_metrics,
                'pipeline_complete': ingestion_metrics['status'] == 'COMPLETE'
            }
            
            if ingestion_metrics['status'] == 'COMPLETE':
                final_metrics.update({
                    'text_metrics': text_metrics,
                    'embedding_metrics': embedding_metrics,
                    'vector_metrics': vector_metrics,
                    'validation_metrics': validation_metrics
                })
            
            observability.end_trace(trace, status='success' if ingestion_metrics['status'] == 'COMPLETE' else 'partial')
            
            logger.info(f"✅ Ingestion monitoring complete for job {job_id}")
            return final_metrics
            
        except Exception as e:
            logger.error(f"❌ Ingestion monitoring failed: {e}")
            observability.log_error(e, {
                'operation': 'ingestion_pipeline_monitoring',
                'job_id': job_id,
                'knowledge_base_id': self.knowledge_base_id
            })
            observability.end_trace(trace, status='error', error=str(e))
            return {'status': 'error', 'error': str(e)}
    
    @trace_function("analyze_s3_documents")
    def _analyze_s3_documents(self) -> Dict[str, Any]:
        """Analyze documents in S3 bucket"""
        try:
            logger.info("📄 Analyzing S3 documents...")
            
            # List all objects in the research-papers folder
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix='research-papers/'
            )
            
            documents = []
            total_size = 0
            
            for obj in response.get('Contents', []):
                doc_info = {
                    'key': obj['Key'],
                    'size_bytes': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'file_type': obj['Key'].split('.')[-1].lower() if '.' in obj['Key'] else 'unknown'
                }
                documents.append(doc_info)
                total_size += obj['Size']
            
            metrics = {
                'total_documents': len(documents),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'documents': documents,
                'file_types': list(set(doc['file_type'] for doc in documents))
            }
            
            logger.info(f"📊 S3 Analysis: {len(documents)} documents, {metrics['total_size_mb']} MB total")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to analyze S3 documents: {e}")
            return {'error': str(e)}
    
    @trace_function("get_ingestion_job_status")
    def _get_ingestion_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get detailed status of ingestion job"""
        try:
            response = self.bedrock_agent.get_ingestion_job(
                knowledgeBaseId=self.knowledge_base_id,
                dataSourceId=self.data_source_id,
                ingestionJobId=job_id
            )
            
            job = response['ingestionJob']
            
            metrics = {
                'job_id': job_id,
                'status': job['status'],
                'started_at': job['startedAt'].isoformat(),
                'updated_at': job['updatedAt'].isoformat()
            }
            
            if 'statistics' in job:
                stats = job['statistics']
                metrics.update({
                    'documents_scanned': stats.get('numberOfDocumentsScanned', 0),
                    'documents_indexed': stats.get('numberOfNewDocumentsIndexed', 0),
                    'documents_modified': stats.get('numberOfModifiedDocumentsIndexed', 0),
                    'documents_deleted': stats.get('numberOfDocumentsDeleted', 0),
                    'documents_failed': stats.get('numberOfDocumentsFailed', 0)
                })
            
            if 'failureReasons' in job and job['failureReasons']:
                metrics['failure_reasons'] = job['failureReasons']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get ingestion job status: {e}")
            return {'error': str(e)}
    
    @trace_function("analyze_text_processing")
    def _analyze_text_processing(self, ingestion_metrics: Dict) -> Dict[str, Any]:
        """Analyze text extraction and chunking process"""
        try:
            logger.info("📝 Analyzing text processing and chunking...")
            
            # Estimate chunking based on document processing
            documents_processed = ingestion_metrics.get('documents_indexed', 0) + ingestion_metrics.get('documents_modified', 0)
            
            # Typical chunking strategy: 1000 tokens per chunk with 200 token overlap
            # Estimate based on average document sizes
            estimated_chunks_per_doc = 10  # Conservative estimate
            total_estimated_chunks = documents_processed * estimated_chunks_per_doc
            
            # Log the text processing details
            observability.log_bedrock_operation(
                operation_type='text_extraction_chunking',
                model_id='bedrock-internal',
                input_data={
                    'documents_count': documents_processed,
                    'chunking_strategy': 'semantic_chunking',
                    'chunk_size_tokens': 1000,
                    'overlap_tokens': 200
                },
                metrics={
                    'estimated_chunks': total_estimated_chunks,
                    'documents_processed': documents_processed
                }
            )
            
            return {
                'documents_processed': documents_processed,
                'chunking_strategy': 'semantic_chunking_with_overlap',
                'chunk_size_tokens': 1000,
                'overlap_tokens': 200,
                'estimated_chunks': total_estimated_chunks
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze text processing: {e}")
            return {'error': str(e)}
    
    @trace_function("analyze_embedding_generation")
    def _analyze_embedding_generation(self) -> Dict[str, Any]:
        """Analyze embedding generation process"""
        try:
            logger.info("🧮 Analyzing embedding generation...")
            
            # Test embedding generation to understand the process
            test_text = "This is a test text for embedding generation analysis."
            
            start_time = time.time()
            
            # Call Titan embedding model
            response = self.bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({
                    "inputText": test_text,
                    "dimensions": 1024,
                    "normalize": True
                })
            )
            
            end_time = time.time()
            embedding_duration = end_time - start_time
            
            response_body = json.loads(response['body'].read())
            embedding_vector = response_body['embedding']
            
            # Log embedding generation details
            observability.log_bedrock_operation(
                operation_type='embedding_generation',
                model_id='amazon.titan-embed-text-v2:0',
                input_data={
                    'text': test_text,
                    'dimensions': 1024,
                    'normalize': True
                },
                output_data={
                    'embedding_dimension': len(embedding_vector),
                    'embedding_norm': sum(x*x for x in embedding_vector)**0.5
                },
                metrics={
                    'generation_time_seconds': embedding_duration,
                    'vector_dimension': len(embedding_vector)
                }
            )
            
            return {
                'embedding_model': 'amazon.titan-embed-text-v2:0',
                'vector_dimension': len(embedding_vector),
                'normalization': True,
                'sample_generation_time': embedding_duration,
                'total_embeddings': 'calculated_from_chunks'
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze embedding generation: {e}")
            return {'error': str(e)}
    
    @trace_function("analyze_vector_storage")
    def _analyze_vector_storage(self) -> Dict[str, Any]:
        """Analyze vector storage in OpenSearch"""
        try:
            logger.info("🗄️ Analyzing vector storage in OpenSearch...")
            
            index_name = "bedrock-knowledge-base-index"
            
            # Get index stats
            stats_url = f"{self.opensearch_endpoint}/{index_name}/_stats"
            stats_response = requests.get(stats_url, auth=self.opensearch_auth)
            
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                index_stats = stats_data['indices'][index_name]
                
                # Get document count
                count_url = f"{self.opensearch_endpoint}/{index_name}/_count"
                count_response = requests.get(count_url, auth=self.opensearch_auth)
                
                document_count = 0
                if count_response.status_code == 200:
                    count_data = count_response.json()
                    document_count = count_data['count']
                
                # Get index mapping to understand structure
                mapping_url = f"{self.opensearch_endpoint}/{index_name}/_mapping"
                mapping_response = requests.get(mapping_url, auth=self.opensearch_auth)
                
                vector_field_info = {}
                if mapping_response.status_code == 200:
                    mapping_data = mapping_response.json()
                    properties = mapping_data[index_name]['mappings']['properties']
                    
                    if 'bedrock-knowledge-base-default-vector' in properties:
                        vector_field_info = properties['bedrock-knowledge-base-default-vector']
                
                # Log vector storage metrics
                storage_metrics = {
                    'vectors_stored': document_count,
                    'index_size_bytes': index_stats['total']['store']['size_in_bytes'],
                    'index_size_mb': round(index_stats['total']['store']['size_in_bytes'] / (1024 * 1024), 2),
                    'vector_dimension': vector_field_info.get('dimension', 1024),
                    'vector_method': vector_field_info.get('method', {}),
                    'storage_engine': 'faiss'
                }
                
                observability.log_vector_search(
                    search_type='storage_analysis',
                    query_vector=[],
                    search_params={'index': index_name},
                    results=[],
                    opensearch_metrics=storage_metrics
                )
                
                return storage_metrics
            else:
                logger.error(f"Failed to get OpenSearch stats: {stats_response.status_code}")
                return {'error': f'OpenSearch stats request failed: {stats_response.status_code}'}
            
        except Exception as e:
            logger.error(f"Failed to analyze vector storage: {e}")
            return {'error': str(e)}
    
    @trace_function("validate_pipeline_completion")
    def _validate_pipeline_completion(self) -> Dict[str, Any]:
        """Validate that the complete pipeline is working end-to-end"""
        try:
            logger.info("✅ Validating complete pipeline...")
            
            # Test search functionality
            test_queries = [
                "vehicle theft reporting",
                "customer support contact",
                "recovery process"
            ]
            
            successful_searches = 0
            total_results = 0
            
            for query in test_queries:
                try:
                    # Test query through Bedrock Knowledge Base
                    response = boto3.client('bedrock-agent-runtime', region_name='us-east-1').retrieve(
                        knowledgeBaseId=self.knowledge_base_id,
                        retrievalQuery={'text': query},
                        retrievalConfiguration={
                            'vectorSearchConfiguration': {
                                'numberOfResults': 3
                            }
                        }
                    )
                    
                    results_count = len(response.get('retrievalResults', []))
                    if results_count > 0:
                        successful_searches += 1
                        total_results += results_count
                    
                except Exception as e:
                    logger.error(f"Test query '{query}' failed: {e}")
            
            validation_metrics = {
                'validation_successful': successful_searches > 0,
                'successful_test_queries': successful_searches,
                'total_test_queries': len(test_queries),
                'searchable_documents': total_results > 0,
                'average_results_per_query': total_results / len(test_queries) if test_queries else 0
            }
            
            logger.info(f"🎯 Pipeline validation: {successful_searches}/{len(test_queries)} queries successful")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Failed to validate pipeline: {e}")
            return {'error': str(e), 'validation_successful': False}
    
    @trace_function("start_monitored_ingestion")
    def start_monitored_ingestion(self, description: str = None) -> str:
        """Start a new ingestion job with monitoring"""
        try:
            logger.info("🚀 Starting monitored ingestion job...")
            
            # Start ingestion job
            response = self.bedrock_agent.start_ingestion_job(
                knowledgeBaseId=self.knowledge_base_id,
                dataSourceId=self.data_source_id,
                description=description or f"Monitored ingestion job started at {datetime.now().isoformat()}"
            )
            
            job_id = response['ingestionJob']['ingestionJobId']
            logger.info(f"✅ Ingestion job started: {job_id}")
            
            # Log the start of ingestion
            observability.log_ingestion_pipeline(
                stage='ingestion_job_start',
                document_info={'job_id': job_id},
                metrics={
                    'knowledge_base_id': self.knowledge_base_id,
                    'data_source_id': self.data_source_id,
                    'start_time': datetime.now(timezone.utc).isoformat()
                }
            )
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start monitored ingestion: {e}")
            observability.log_error(e, {
                'operation': 'start_monitored_ingestion',
                'knowledge_base_id': self.knowledge_base_id,
                'data_source_id': self.data_source_id
            })
            raise

# Global instance
ingestion_monitor = IngestionPipelineMonitor() 