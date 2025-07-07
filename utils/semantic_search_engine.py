"""
Semantic Search Engine for SVL Knowledge Base
Combines PDF processing and embeddings for intelligent document search
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import re
from pathlib import Path

from utils.pdf_document_manager import PDFDocumentManager
from utils.embedding_manager import EmbeddingManager
from utils.logger import get_logger

logger = get_logger("semantic_search_engine")

class SemanticSearchEngine:
    """
    Intelligent search engine for SVL knowledge base
    Combines document processing, embeddings, and semantic search
    """
    
    def __init__(self, 
                 s3_bucket_name: str,
                 index_storage_path: str = "./knowledge_base_index",
                 aws_region: str = "us-east-1",
                 embedding_dimension: int = 1536):
        """
        Initialize Semantic Search Engine
        
        Args:
            s3_bucket_name: S3 bucket for document storage
            index_storage_path: Local path to store/load index files
            aws_region: AWS region
            embedding_dimension: Embedding dimension for Titan
        """
        self.s3_bucket_name = s3_bucket_name
        self.index_storage_path = index_storage_path
        self.aws_region = aws_region
        
        # Initialize components
        self.pdf_manager = PDFDocumentManager(s3_bucket_name, aws_region)
        self.embedding_manager = EmbeddingManager(
            aws_region=aws_region,
            embedding_dimension=embedding_dimension
        )
        
        # Search configuration
        self.default_chunk_size = 800
        self.default_overlap = 100
        self.default_top_k = 5
        self.default_similarity_threshold = 0.7
        
        # Document categories for filtering
        self.category_weights = {
            "sop": 1.2,      # Standard Operating Procedures - higher weight
            "process_docs": 1.1,  # Process Documentation - slightly higher
            "faq": 1.0,      # FAQ - normal weight
            "contact": 0.9   # Contact info - lower weight
        }
        
        # Query type mappings
        self.query_type_config = {
            "general": {
                "top_k": 5,
                "similarity_threshold": 0.7,
                "categories": ["sop", "process_docs", "faq", "contact"]
            },
            "sop": {
                "top_k": 3,
                "similarity_threshold": 0.75,
                "categories": ["sop"]
            },
            "faq": {
                "top_k": 5,
                "similarity_threshold": 0.6,
                "categories": ["faq", "process_docs"]
            },
            "process": {
                "top_k": 4,
                "similarity_threshold": 0.7,
                "categories": ["process_docs", "sop"]
            },
            "contact": {
                "top_k": 2,
                "similarity_threshold": 0.8,
                "categories": ["contact"]
            }
        }
        
        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "average_search_time": 0,
            "total_search_time": 0,
            "cache_hits": 0,
            "last_index_update": None
        }
        
        # Simple search cache
        self.search_cache = {}
        self.max_cache_size = 100
    
    def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build or load the search index from documents
        
        Args:
            force_rebuild: Whether to force rebuild even if index exists
            
        Returns:
            Build statistics and results
        """
        logger.info("Building semantic search index...")
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check if index already exists and load it
            if not force_rebuild and self._load_existing_index():
                return {
                    "status": "loaded_existing",
                    "index_size": self.embedding_manager.get_stats()["index_size"],
                    "load_time": (datetime.now(timezone.utc) - start_time).total_seconds()
                }
            
            # Upload PDFs from data folder to S3
            logger.info("Uploading PDFs from data folder...")
            uploaded_files = self.pdf_manager.upload_pdfs_from_data_folder()
            
            if not any(uploaded_files.values()):
                logger.warning("No PDF files found to process")
                return {"status": "no_files", "uploaded_files": uploaded_files}
            
            # Process all documents
            all_chunks = []
            all_metadata = []
            document_count = 0
            
            for category, file_list in uploaded_files.items():
                if not file_list:
                    continue
                
                logger.info(f"Processing {len(file_list)} files in category: {category}")
                
                for s3_key in file_list:
                    try:
                        # Extract text from PDF
                        extraction_result = self.pdf_manager.extract_text_from_pdf(s3_key=s3_key)
                        
                        if not extraction_result["text"].strip():
                            logger.warning(f"No text extracted from {s3_key}")
                            continue
                        
                        # Chunk the document
                        chunks = self.pdf_manager.chunk_document(
                            extraction_result["text"],
                            chunk_size=self.default_chunk_size,
                            overlap=self.default_overlap
                        )
                        
                        # Add document metadata to each chunk
                        for chunk in chunks:
                            chunk_metadata = {
                                **chunk,
                                "document_source": s3_key,
                                "document_category": category,
                                "extraction_method": extraction_result["extraction_method"],
                                "extraction_quality": extraction_result["quality_score"],
                                "document_filename": s3_key.split('/')[-1],
                                "category_weight": self.category_weights.get(category, 1.0)
                            }
                            
                            all_chunks.append(chunk["text"])
                            all_metadata.append(chunk_metadata)
                        
                        document_count += 1
                        logger.info(f"Processed {s3_key}: {len(chunks)} chunks")
                        
                    except Exception as e:
                        logger.error(f"Failed to process {s3_key}: {e}")
                        continue
            
            if not all_chunks:
                return {"status": "no_chunks_extracted", "document_count": document_count}
            
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            embedding_results = self.embedding_manager.generate_embeddings(
                text_chunks=all_chunks,
                metadata_list=all_metadata
            )
            
            # Store embeddings in FAISS index
            success = self.embedding_manager.store_embeddings(
                embedding_results=embedding_results,
                index_name="svl_knowledge_base"
            )
            
            if not success:
                return {"status": "embedding_storage_failed"}
            
            # Save index to disk
            self._save_index()
            
            # Update stats
            self.search_stats["last_index_update"] = datetime.now(timezone.utc).isoformat()
            
            build_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info(f"Successfully built index in {build_time:.2f} seconds")
            
            return {
                "status": "success",
                "document_count": document_count,
                "chunk_count": len(all_chunks),
                "embedding_count": len(embedding_results),
                "build_time": build_time,
                "categories_processed": list(uploaded_files.keys()),
                "index_size": self.embedding_manager.get_stats()["index_size"]
            }
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return {"status": "error", "error": str(e)}
    
    def search_knowledge_base(self, 
                            query: str,
                            query_type: str = "general",
                            top_k: Optional[int] = None,
                            similarity_threshold: Optional[float] = None,
                            categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant information
        
        Args:
            query: Search query
            query_type: Type of query (general, sop, faq, process, contact)
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            categories: Specific categories to search
            
        Returns:
            Search results with ranked chunks and metadata
        """
        import time
        
        start_time = time.time()
        self.search_stats["total_searches"] += 1
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query, query_type, top_k, similarity_threshold, categories)
            if cache_key in self.search_cache:
                self.search_stats["cache_hits"] += 1
                cached_result = self.search_cache[cache_key]
                cached_result["from_cache"] = True
                return cached_result
            
            # Get configuration for query type
            config = self.query_type_config.get(query_type, self.query_type_config["general"])
            
            # Use provided parameters or defaults from config
            search_top_k = top_k or config["top_k"]
            search_threshold = similarity_threshold or config["similarity_threshold"]
            search_categories = categories or config["categories"]
            
            logger.info(f"Searching knowledge base: '{query}' (type: {query_type})")
            
            # Perform similarity search
            search_results = self.embedding_manager.similarity_search(
                query_text=query,
                top_k=search_top_k * 2,  # Get more results for filtering
                similarity_threshold=search_threshold
            )
            
            if not search_results:
                return {
                    "query": query,
                    "query_type": query_type,
                    "results": [],
                    "total_results": 0,
                    "search_time": time.time() - start_time,
                    "from_cache": False
                }
            
            # Filter by categories and apply weights
            filtered_results = self._filter_and_rank_results(
                search_results, 
                search_categories, 
                search_top_k
            )
            
            # Get contextual information
            enhanced_results = self._enhance_results_with_context(filtered_results)
            
            # Create response
            result = {
                "query": query,
                "query_type": query_type,
                "results": enhanced_results,
                "total_results": len(enhanced_results),
                "search_time": time.time() - start_time,
                "search_config": {
                    "top_k": search_top_k,
                    "similarity_threshold": search_threshold,
                    "categories": search_categories
                },
                "from_cache": False
            }
            
            # Cache the result
            self._cache_search_result(cache_key, result)
            
            # Update stats
            self.search_stats["successful_searches"] += 1
            search_time = time.time() - start_time
            self.search_stats["total_search_time"] += search_time
            self.search_stats["average_search_time"] = (
                self.search_stats["total_search_time"] / self.search_stats["total_searches"]
            )
            
            logger.info(f"Found {len(enhanced_results)} relevant results in {search_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "query": query,
                "query_type": query_type,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "search_time": time.time() - start_time,
                "from_cache": False
            }
    
    def _filter_and_rank_results(self, 
                                search_results: List[Dict[str, Any]], 
                                categories: List[str], 
                                top_k: int) -> List[Dict[str, Any]]:
        """
        Filter results by categories and apply ranking weights
        
        Args:
            search_results: Raw search results
            categories: Categories to include
            top_k: Number of results to return
            
        Returns:
            Filtered and ranked results
        """
        filtered_results = []
        
        for result in search_results:
            metadata = result.get("metadata", {})
            document_category = metadata.get("document_category", "unknown")
            
            # Filter by category
            if document_category not in categories:
                continue
            
            # Apply category weight to similarity score
            category_weight = metadata.get("category_weight", 1.0)
            weighted_score = result["similarity_score"] * category_weight
            
            # Add weighted score
            result["weighted_score"] = weighted_score
            result["category_boost"] = category_weight
            
            filtered_results.append(result)
        
        # Sort by weighted score
        filtered_results.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        # Return top K results
        return filtered_results[:top_k]
    
    def _enhance_results_with_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance search results with additional context and formatting
        
        Args:
            results: Filtered search results
            
        Returns:
            Enhanced results with context
        """
        enhanced_results = []
        
        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            
            # Extract key information
            enhanced_result = {
                "rank": i + 1,
                "relevance_score": round(result["weighted_score"], 3),
                "similarity_score": round(result["similarity_score"], 3),
                "text": result["text"],
                "document": {
                    "filename": metadata.get("document_filename", "Unknown"),
                    "category": metadata.get("document_category", "unknown"),
                    "source": metadata.get("document_source", ""),
                    "extraction_method": metadata.get("extraction_method", ""),
                    "quality_score": metadata.get("extraction_quality", 0)
                },
                "chunk_info": {
                    "chunk_id": metadata.get("chunk_id", 0),
                    "length": metadata.get("length", 0),
                    "word_count": metadata.get("word_count", 0),
                    "position": metadata.get("position_in_document", "")
                },
                "context": {
                    "category_weight": metadata.get("category_weight", 1.0),
                    "text_preview": self._get_text_preview(result["text"]),
                    "key_phrases": self._extract_key_phrases(result["text"])
                },
                "metadata": metadata
            }
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _get_text_preview(self, text: str, max_length: int = 200) -> str:
        """Get a preview of text for display"""
        if len(text) <= max_length:
            return text
        
        # Try to break at a sentence boundary
        truncated = text[:max_length]
        last_sentence = truncated.rfind('.')
        
        if last_sentence > max_length * 0.5:  # If we can get at least half the text
            return truncated[:last_sentence + 1]
        else:
            return truncated + "..."
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text (simple implementation)"""
        # Simple key phrase extraction
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter and deduplicate
        key_phrases = []
        for phrase in words:
            if len(phrase) > 3 and phrase not in key_phrases:
                key_phrases.append(phrase)
        
        return key_phrases[:5]  # Return top 5 key phrases
    
    def get_contextual_answer(self, 
                            query: str, 
                            search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format search results for integration with Bedrock
        
        Args:
            query: Original query
            search_results: Enhanced search results
            
        Returns:
            Formatted context for AI model
        """
        if not search_results:
            return {
                "context": "No relevant information found in the knowledge base.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Build context from top results
        context_parts = []
        sources = []
        total_confidence = 0
        
        for result in search_results[:3]:  # Use top 3 results for context
            document_info = result["document"]
            
            context_part = f"From {document_info['filename']} ({document_info['category']}):\n{result['text']}"
            context_parts.append(context_part)
            
            sources.append({
                "filename": document_info["filename"],
                "category": document_info["category"],
                "relevance_score": result["relevance_score"],
                "chunk_info": result["chunk_info"]
            })
            
            total_confidence += result["relevance_score"]
        
        # Calculate overall confidence
        confidence = min(total_confidence / len(search_results), 1.0)
        
        # Format context
        context = f"Based on the SVL knowledge base, here's relevant information for your query:\n\n"
        context += "\n\n---\n\n".join(context_parts)
        
        return {
            "context": context,
            "sources": sources,
            "confidence": round(confidence, 3),
            "query": query,
            "total_sources": len(search_results)
        }
    
    def _get_cache_key(self, query: str, query_type: str, top_k: Optional[int], 
                       similarity_threshold: Optional[float], categories: Optional[List[str]]) -> str:
        """Generate cache key for search parameters"""
        import hashlib
        
        cache_data = {
            "query": query.lower(),
            "query_type": query_type,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "categories": sorted(categories) if categories else None
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _cache_search_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache search result with size management"""
        if len(self.search_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            items_to_remove = self.max_cache_size // 5
            for _ in range(items_to_remove):
                self.search_cache.pop(next(iter(self.search_cache)))
        
        self.search_cache[cache_key] = result
    
    def _load_existing_index(self) -> bool:
        """Load existing index from disk"""
        try:
            return self.embedding_manager.load_index_from_file(self.index_storage_path)
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            return False
    
    def _save_index(self) -> bool:
        """Save index to disk"""
        try:
            return self.embedding_manager.save_index_to_file(self.index_storage_path)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def update_knowledge_base(self, new_pdf_s3_key: str) -> Dict[str, Any]:
        """
        Add a new PDF to the knowledge base
        
        Args:
            new_pdf_s3_key: S3 key of the new PDF
            
        Returns:
            Update result
        """
        try:
            logger.info(f"Adding new document to knowledge base: {new_pdf_s3_key}")
            
            # Extract category from S3 key
            category = new_pdf_s3_key.split('/')[1] if '/' in new_pdf_s3_key else 'unknown'
            
            # Extract text from new PDF
            extraction_result = self.pdf_manager.extract_text_from_pdf(s3_key=new_pdf_s3_key)
            
            # Chunk the document
            chunks = self.pdf_manager.chunk_document(
                extraction_result["text"],
                chunk_size=self.default_chunk_size,
                overlap=self.default_overlap
            )
            
            # Prepare chunks with metadata
            chunk_texts = []
            chunk_metadata = []
            
            for chunk in chunks:
                chunk_metadata_entry = {
                    **chunk,
                    "document_source": new_pdf_s3_key,
                    "document_category": category,
                    "extraction_method": extraction_result["extraction_method"],
                    "extraction_quality": extraction_result["quality_score"],
                    "document_filename": new_pdf_s3_key.split('/')[-1],
                    "category_weight": self.category_weights.get(category, 1.0)
                }
                
                chunk_texts.append(chunk["text"])
                chunk_metadata.append(chunk_metadata_entry)
            
            # Generate embeddings
            embedding_results = self.embedding_manager.generate_embeddings(
                text_chunks=chunk_texts,
                metadata_list=chunk_metadata
            )
            
            # Store embeddings
            success = self.embedding_manager.store_embeddings(
                embedding_results=embedding_results,
                index_name="svl_knowledge_base"
            )
            
            if success:
                # Save updated index
                self._save_index()
                
                # Clear search cache
                self.search_cache.clear()
                
                logger.info(f"Successfully added {len(chunks)} chunks from {new_pdf_s3_key}")
                
                return {
                    "status": "success",
                    "document": new_pdf_s3_key,
                    "chunks_added": len(chunks),
                    "total_index_size": self.embedding_manager.get_stats()["index_size"]
                }
            else:
                return {"status": "failed", "error": "Failed to store embeddings"}
            
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        embedding_stats = self.embedding_manager.get_stats()
        
        return {
            "search_engine": self.search_stats,
            "embeddings": embedding_stats,
            "index_storage_path": self.index_storage_path,
            "s3_bucket": self.s3_bucket_name,
            "configuration": {
                "default_chunk_size": self.default_chunk_size,
                "default_overlap": self.default_overlap,
                "category_weights": self.category_weights,
                "query_types": list(self.query_type_config.keys())
            }
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.search_cache.clear()
        self.embedding_manager.clear_cache()
        logger.info("Cleared all caches") 