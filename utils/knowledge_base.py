"""
Main Knowledge Base System for SVL Chatbot
Integrates all components for semantic document search and AI-powered responses
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json

from utils.semantic_search_engine import SemanticSearchEngine
from utils.logger import get_logger

logger = get_logger("knowledge_base")

class KnowledgeBase:
    """
    Main knowledge base system for SVL chatbot
    Provides unified interface for document processing and semantic search
    """
    
    def __init__(self, 
                 s3_bucket_name: str = None,
                 index_storage_path: str = "./knowledge_base_index",
                 aws_region: str = "us-east-1"):
        """
        Initialize Knowledge Base
        
        Args:
            s3_bucket_name: S3 bucket for document storage
            index_storage_path: Local path for index storage
            aws_region: AWS region
        """
        # Set default S3 bucket name
        self.s3_bucket_name = s3_bucket_name or os.getenv(
            'KNOWLEDGE_BASE_S3_BUCKET', 
            'svl-knowledge-base'
        )
        
        self.index_storage_path = index_storage_path
        self.aws_region = aws_region
        
        # Initialize search engine
        try:
            self.search_engine = SemanticSearchEngine(
                s3_bucket_name=self.s3_bucket_name,
                index_storage_path=index_storage_path,
                aws_region=aws_region
            )
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            self.search_engine = None
            raise
        
        # Configuration
        self.config = {
            "auto_initialize": True,
            "cache_enabled": True,
            "default_query_type": "general",
            "max_context_length": 3000,
            "enable_response_enhancement": True
        }
        
        # Status tracking
        self.status = {
            "initialized": False,
            "index_built": False,
            "last_update": None,
            "error_count": 0,
            "total_queries": 0
        }
        
        # Query history for analytics
        self.query_history = []
        self.max_history_size = 1000
    
    def initialize_knowledge_base(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Initialize the knowledge base system
        
        Args:
            force_rebuild: Whether to force rebuild of index
            
        Returns:
            Initialization result
        """
        logger.info("Initializing SVL Knowledge Base...")
        
        try:
            if self.search_engine is None:
                raise Exception("Search engine not available")
            
            # Create data folder structure if it doesn't exist
            self._ensure_data_folder_structure()
            
            # Build or load search index
            build_result = self.search_engine.build_index(force_rebuild=force_rebuild)
            
            if build_result["status"] in ["success", "loaded_existing"]:
                self.status["initialized"] = True
                self.status["index_built"] = True
                self.status["last_update"] = datetime.now(timezone.utc).isoformat()
                
                logger.info("Knowledge base initialized successfully")
                
                return {
                    "status": "success",
                    "message": "Knowledge base initialized successfully",
                    "build_result": build_result,
                    "s3_bucket": self.s3_bucket_name,
                    "index_path": self.index_storage_path
                }
            else:
                self.status["error_count"] += 1
                
                return {
                    "status": "partial_success",
                    "message": "Knowledge base initialized but index build had issues",
                    "build_result": build_result,
                    "warnings": ["Index build incomplete - some features may not work"]
                }
                
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            self.status["error_count"] += 1
            
            return {
                "status": "error",
                "message": f"Failed to initialize knowledge base: {e}",
                "error": str(e)
            }
    
    def query_knowledge_base(self, 
                           user_query: str,
                           query_type: str = None,
                           include_context: bool = True,
                           max_results: int = None,
                           similarity_threshold: float = None) -> Dict[str, Any]:
        """
        Query the knowledge base for relevant information
        
        Args:
            user_query: User's question
            query_type: Type of query (general, sop, faq, process, contact)
            include_context: Whether to include formatted context for AI
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            Query results with context and metadata
        """
        if not self.status["initialized"] or self.search_engine is None:
            return {
                "status": "not_initialized",
                "message": "Knowledge base not initialized",
                "context": "I don't have access to the knowledge base at the moment. Please try again later.",
                "sources": []
            }
        
        # Use default query type if not specified
        query_type = query_type or self.config["default_query_type"]
        
        # Track query
        self.status["total_queries"] += 1
        query_record = {
            "query": user_query,
            "query_type": query_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_query_id": f"q_{self.status['total_queries']}"
        }
        
        try:
            logger.info(f"Processing knowledge base query: '{user_query}' (type: {query_type})")
            
            # Search knowledge base
            search_results = self.search_engine.search_knowledge_base(
                query=user_query,
                query_type=query_type,
                top_k=max_results,
                similarity_threshold=similarity_threshold
            )
            
            # Track search result
            query_record["search_success"] = search_results.get("total_results", 0) > 0
            query_record["results_count"] = search_results.get("total_results", 0)
            query_record["search_time"] = search_results.get("search_time", 0)
            
            # Add to history
            self._add_to_query_history(query_record)
            
            if search_results.get("total_results", 0) == 0:
                return {
                    "status": "no_results",
                    "message": "No relevant information found in the knowledge base",
                    "context": "I couldn't find specific information about your question in our knowledge base. Could you please rephrase your question or provide more details?",
                    "sources": [],
                    "query": user_query,
                    "query_type": query_type,
                    "search_time": search_results.get("search_time", 0)
                }
            
            # Format response
            response = {
                "status": "success",
                "query": user_query,
                "query_type": query_type,
                "results": search_results["results"],
                "total_results": search_results["total_results"],
                "search_time": search_results["search_time"],
                "from_cache": search_results.get("from_cache", False)
            }
            
            # Include formatted context for AI if requested
            if include_context:
                contextual_answer = self.search_engine.get_contextual_answer(
                    query=user_query,
                    search_results=search_results["results"]
                )
                
                response.update({
                    "context": contextual_answer["context"],
                    "sources": contextual_answer["sources"],
                    "confidence": contextual_answer["confidence"]
                })
                
                # Enhance context if enabled
                if self.config["enable_response_enhancement"]:
                    response["context"] = self._enhance_context_for_ai(
                        context=contextual_answer["context"],
                        query=user_query,
                        query_type=query_type
                    )
            
            logger.info(f"Knowledge base query processed: {response['total_results']} results found")
            
            return response
            
        except Exception as e:
            logger.error(f"Knowledge base query failed: {e}")
            self.status["error_count"] += 1
            
            # Add error to history
            query_record["search_success"] = False
            query_record["error"] = str(e)
            self._add_to_query_history(query_record)
            
            return {
                "status": "error",
                "message": f"Knowledge base query failed: {e}",
                "context": "I encountered an error while searching our knowledge base. Please try again or contact support if the issue persists.",
                "sources": [],
                "query": user_query,
                "error": str(e)
            }
    
    def update_knowledge_base(self, new_pdf_path: str = None, s3_key: str = None) -> Dict[str, Any]:
        """
        Add new document to knowledge base
        
        Args:
            new_pdf_path: Local path to PDF file
            s3_key: S3 key of already uploaded PDF
            
        Returns:
            Update result
        """
        if not self.status["initialized"] or self.search_engine is None:
            return {
                "status": "not_initialized",
                "message": "Knowledge base not initialized"
            }
        
        try:
            if s3_key:
                # Use existing S3 document
                result = self.search_engine.update_knowledge_base(s3_key)
            elif new_pdf_path:
                # Upload local file first
                # This would require implementing upload to S3
                return {
                    "status": "not_implemented",
                    "message": "Local file upload not implemented yet"
                }
            else:
                return {
                    "status": "error",
                    "message": "Either new_pdf_path or s3_key must be provided"
                }
            
            if result["status"] == "success":
                self.status["last_update"] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Knowledge base updated: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}")
            self.status["error_count"] += 1
            return {
                "status": "error",
                "message": f"Failed to update knowledge base: {e}",
                "error": str(e)
            }
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get comprehensive knowledge base statistics"""
        if self.search_engine is None:
            return {"status": "not_available", "message": "Search engine not initialized"}
        
        try:
            search_stats = self.search_engine.get_stats()
            
            return {
                "status": self.status,
                "configuration": self.config,
                "search_engine": search_stats,
                "query_history_size": len(self.query_history),
                "recent_queries": self.query_history[-5:] if self.query_history else [],
                "s3_bucket": self.s3_bucket_name,
                "index_storage_path": self.index_storage_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def clear_cache(self):
        """Clear all caches"""
        if self.search_engine:
            self.search_engine.clear_cache()
        logger.info("Cleared knowledge base caches")
    
    def _ensure_data_folder_structure(self):
        """Ensure data folder structure exists"""
        data_path = "./data"
        categories = ["sop", "process_docs", "faq", "contact"]
        
        try:
            for category in categories:
                category_path = os.path.join(data_path, category)
                os.makedirs(category_path, exist_ok=True)
            
            logger.info("Data folder structure ensured")
            
        except Exception as e:
            logger.warning(f"Failed to create data folders: {e}")
    
    def _enhance_context_for_ai(self, 
                               context: str, 
                               query: str, 
                               query_type: str) -> str:
        """
        Enhance context with additional instructions for AI
        
        Args:
            context: Original context
            query: User query
            query_type: Query type
            
        Returns:
            Enhanced context with AI instructions
        """
        # Add query-specific instructions
        query_instructions = {
            "sop": "Please provide step-by-step instructions based on our standard operating procedures.",
            "faq": "Please provide a clear and direct answer as this appears to be a frequently asked question.",
            "process": "Please explain the process clearly with any relevant steps or requirements.",
            "contact": "Please provide the specific contact information requested.",
            "general": "Please provide helpful information based on our knowledge base."
        }
        
        instruction = query_instructions.get(query_type, query_instructions["general"])
        
        # Truncate context if too long
        max_length = self.config["max_context_length"]
        if len(context) > max_length:
            context = context[:max_length] + "...[truncated]"
        
        enhanced_context = f"""
{instruction}

User Question: {query}

Relevant Information from SVL Knowledge Base:
{context}

Please provide a helpful response based on this information. If the information doesn't fully answer the question, please indicate what additional details might be needed.
"""
        
        return enhanced_context.strip()
    
    def _add_to_query_history(self, query_record: Dict[str, Any]):
        """Add query to history with size management"""
        self.query_history.append(query_record)
        
        # Maintain history size
        if len(self.query_history) > self.max_history_size:
            self.query_history = self.query_history[-self.max_history_size:]
    
    def get_document_categories(self) -> List[str]:
        """Get available document categories"""
        return ["sop", "process_docs", "faq", "contact"]
    
    def get_query_types(self) -> List[str]:
        """Get available query types"""
        return ["general", "sop", "faq", "process", "contact"]
    
    def is_ready(self) -> bool:
        """Check if knowledge base is ready for queries"""
        return (
            self.status["initialized"] and 
            self.status["index_built"] and 
            self.search_engine is not None
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring"""
        return {
            "status": "healthy" if self.is_ready() else "unhealthy",
            "initialized": self.status["initialized"],
            "index_built": self.status["index_built"],
            "search_engine_available": self.search_engine is not None,
            "error_count": self.status["error_count"],
            "total_queries": self.status["total_queries"],
            "last_update": self.status["last_update"],
            "s3_bucket": self.s3_bucket_name
        } 