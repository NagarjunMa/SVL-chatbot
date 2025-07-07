"""
Embedding Manager for SVL Knowledge Base
Handles text embeddings using Amazon Titan with FAISS vector storage
"""

import boto3
import json
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    import faiss
except ImportError:
    print("Warning: FAISS not installed. Run: pip install faiss-cpu")

from utils.logger import get_logger

logger = get_logger("embedding_manager")

class EmbeddingManager:
    """
    Manages text embeddings using Amazon Titan Text Embeddings
    Provides efficient batch processing and FAISS vector storage
    """
    
    def __init__(self, 
                 aws_region: str = "us-east-1",
                 embedding_dimension: int = 1536,
                 batch_size: int = 10,
                 max_workers: int = 3):
        """
        Initialize Embedding Manager
        
        Args:
            aws_region: AWS region for Bedrock
            embedding_dimension: Dimension of embeddings (Titan default: 1536)
            batch_size: Number of texts to process in each batch
            max_workers: Maximum number of concurrent workers
        """
        self.aws_region = aws_region
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Amazon Titan Text Embeddings model
        self.titan_model_id = "amazon.titan-embed-text-v1"
        
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
        
        # FAISS index and metadata storage
        self.vector_index = None
        self.metadata_store = {}
        self.index_to_metadata = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance tracking
        self.embedding_stats = {
            "total_embeddings_generated": 0,
            "total_api_calls": 0,
            "total_processing_time": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Simple cache for embeddings
        self.embedding_cache = {}
        self.max_cache_size = 1000
    
    def generate_embeddings(self, 
                          text_chunks: List[str], 
                          metadata_list: List[Dict[str, Any]] = None,
                          use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks using Amazon Titan
        
        Args:
            text_chunks: List of text chunks to embed
            metadata_list: Optional metadata for each chunk
            use_cache: Whether to use caching for embeddings
            
        Returns:
            List of dictionaries containing embeddings and metadata
        """
        if not text_chunks:
            return []
        
        if metadata_list is None:
            metadata_list = [{"chunk_id": i} for i in range(len(text_chunks))]
        
        if len(text_chunks) != len(metadata_list):
            raise ValueError("Number of text chunks must match number of metadata entries")
        
        logger.info(f"Generating embeddings for {len(text_chunks)} text chunks")
        start_time = time.time()
        
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(text_chunks), self.batch_size):
            batch_texts = text_chunks[i:i + self.batch_size]
            batch_metadata = metadata_list[i:i + self.batch_size]
            
            batch_results = self._process_batch(batch_texts, batch_metadata, use_cache)
            results.extend(batch_results)
        
        processing_time = time.time() - start_time
        self.embedding_stats["total_processing_time"] += processing_time
        
        logger.info(f"Generated {len(results)} embeddings in {processing_time:.2f} seconds")
        
        return results
    
    def _process_batch(self, 
                      texts: List[str], 
                      metadata_list: List[Dict[str, Any]], 
                      use_cache: bool) -> List[Dict[str, Any]]:
        """
        Process a batch of texts for embedding generation
        
        Args:
            texts: Batch of texts
            metadata_list: Batch of metadata
            use_cache: Whether to use caching
            
        Returns:
            List of embedding results
        """
        results = []
        texts_to_process = []
        indices_to_process = []
        
        # Check cache first
        for idx, text in enumerate(texts):
            if use_cache:
                text_hash = self._get_text_hash(text)
                if text_hash in self.embedding_cache:
                    cached_embedding = self.embedding_cache[text_hash]
                    result = {
                        "text": text,
                        "embedding": cached_embedding,
                        "metadata": metadata_list[idx],
                        "embedding_dimension": len(cached_embedding),
                        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
                        "from_cache": True
                    }
                    results.append(result)
                    self.embedding_stats["cache_hits"] += 1
                    continue
            
            # Add to processing queue
            texts_to_process.append(text)
            indices_to_process.append(idx)
            self.embedding_stats["cache_misses"] += 1
        
        # Process uncached texts
        if texts_to_process:
            embeddings = self._generate_titan_embeddings(texts_to_process)
            
            for i, embedding in enumerate(embeddings):
                original_idx = indices_to_process[i]
                text = texts_to_process[i]
                
                # Cache the embedding
                if use_cache:
                    self._cache_embedding(text, embedding)
                
                result = {
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata_list[original_idx],
                    "embedding_dimension": len(embedding),
                    "generated_timestamp": datetime.now(timezone.utc).isoformat(),
                    "from_cache": False
                }
                results.append(result)
        
        # Sort results to maintain original order
        results.sort(key=lambda x: texts.index(x["text"]))
        
        return results
    
    def _generate_titan_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Amazon Titan Text Embeddings
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            try:
                # Prepare request for Titan
                body = {
                    "inputText": text
                }
                
                # Call Bedrock
                response = self.bedrock_client.invoke_model(
                    modelId=self.titan_model_id,
                    body=json.dumps(body),
                    contentType="application/json"
                )
                
                # Parse response
                response_body = json.loads(response["body"].read())
                embedding = response_body["embedding"]
                
                embeddings.append(embedding)
                self.embedding_stats["total_api_calls"] += 1
                self.embedding_stats["total_embeddings_generated"] += 1
                
                # Add small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: {text[:50]}... Error: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * self.embedding_dimension)
        
        return embeddings
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding with size management"""
        with self.lock:
            text_hash = self._get_text_hash(text)
            
            # Remove oldest entries if cache is full
            if len(self.embedding_cache) >= self.max_cache_size:
                # Remove 20% of cache (simple LRU approximation)
                items_to_remove = len(self.embedding_cache) // 5
                for _ in range(items_to_remove):
                    self.embedding_cache.pop(next(iter(self.embedding_cache)))
            
            self.embedding_cache[text_hash] = embedding
    
    def store_embeddings(self, 
                        embedding_results: List[Dict[str, Any]], 
                        index_name: str = "svl_knowledge_base") -> bool:
        """
        Store embeddings in FAISS index with metadata
        
        Args:
            embedding_results: List of embedding results from generate_embeddings
            index_name: Name for the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not embedding_results:
                logger.warning("No embeddings to store")
                return False
            
            logger.info(f"Storing {len(embedding_results)} embeddings in FAISS index")
            
            # Extract embeddings and metadata
            embeddings = []
            metadata_entries = []
            
            for result in embedding_results:
                embeddings.append(result["embedding"])
                
                # Create comprehensive metadata
                metadata = {
                    **result["metadata"],
                    "text": result["text"],
                    "text_length": len(result["text"]),
                    "embedding_dimension": result["embedding_dimension"],
                    "generated_timestamp": result["generated_timestamp"],
                    "from_cache": result.get("from_cache", False),
                    "index_name": index_name
                }
                metadata_entries.append(metadata)
            
            # Convert to numpy array
            embedding_matrix = np.array(embeddings, dtype=np.float32)
            
            # Initialize or update FAISS index
            if self.vector_index is None:
                # Create new index
                self.vector_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product (cosine similarity)
                logger.info(f"Created new FAISS index with dimension {self.embedding_dimension}")
            
            # Add embeddings to index
            start_idx = self.vector_index.ntotal
            self.vector_index.add(embedding_matrix)
            
            # Store metadata mapping
            with self.lock:
                for i, metadata in enumerate(metadata_entries):
                    index_position = start_idx + i
                    self.index_to_metadata[index_position] = metadata
                    
                    # Also store in general metadata store with a unique key
                    metadata_key = f"{index_name}_{index_position}"
                    self.metadata_store[metadata_key] = metadata
            
            logger.info(f"Successfully stored {len(embeddings)} embeddings. Total index size: {self.vector_index.ntotal}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            return False
    
    def similarity_search(self, 
                         query_text: str = None,
                         query_embedding: List[float] = None,
                         top_k: int = 5,
                         similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find most similar document chunks using FAISS
        
        Args:
            query_text: Query text (will generate embedding)
            query_embedding: Pre-computed query embedding
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar chunks with metadata and scores
        """
        try:
            if self.vector_index is None or self.vector_index.ntotal == 0:
                logger.warning("No embeddings in index for similarity search")
                return []
            
            # Get query embedding
            if query_embedding is None:
                if query_text is None:
                    raise ValueError("Either query_text or query_embedding must be provided")
                
                # Generate embedding for query
                query_results = self.generate_embeddings([query_text])
                if not query_results:
                    logger.error("Failed to generate embedding for query")
                    return []
                
                query_embedding = query_results[0]["embedding"]
            
            # Convert to numpy array and normalize for cosine similarity
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Perform search
            scores, indices = self.vector_index.search(query_vector, top_k)
            
            # Process results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                # Apply similarity threshold
                if score < similarity_threshold:
                    continue
                
                # Get metadata
                metadata = self.index_to_metadata.get(idx, {})
                
                result = {
                    "rank": i + 1,
                    "similarity_score": float(score),
                    "index_position": int(idx),
                    "text": metadata.get("text", ""),
                    "metadata": metadata,
                    "search_timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks above threshold {similarity_threshold}")
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def save_index_to_file(self, file_path: str) -> bool:
        """
        Save FAISS index and metadata to file
        
        Args:
            file_path: Path to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.vector_index is None:
                logger.warning("No index to save")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.vector_index, f"{file_path}.faiss")
            
            # Save metadata
            metadata_file = f"{file_path}.metadata"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    "index_to_metadata": self.index_to_metadata,
                    "metadata_store": self.metadata_store,
                    "embedding_stats": self.embedding_stats,
                    "embedding_dimension": self.embedding_dimension,
                    "save_timestamp": datetime.now(timezone.utc).isoformat()
                }, f)
            
            logger.info(f"Saved index with {self.vector_index.ntotal} embeddings to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index_from_file(self, file_path: str) -> bool:
        """
        Load FAISS index and metadata from file
        
        Args:
            file_path: Path to load the index from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            faiss_file = f"{file_path}.faiss"
            metadata_file = f"{file_path}.metadata"
            
            if not os.path.exists(faiss_file) or not os.path.exists(metadata_file):
                logger.warning(f"Index files not found at {file_path}")
                return False
            
            # Load FAISS index
            self.vector_index = faiss.read_index(faiss_file)
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.index_to_metadata = saved_data["index_to_metadata"]
                self.metadata_store = saved_data["metadata_store"]
                self.embedding_stats = saved_data.get("embedding_stats", self.embedding_stats)
                self.embedding_dimension = saved_data.get("embedding_dimension", self.embedding_dimension)
            
            logger.info(f"Loaded index with {self.vector_index.ntotal} embeddings from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics"""
        return {
            **self.embedding_stats,
            "index_size": self.vector_index.ntotal if self.vector_index else 0,
            "cache_size": len(self.embedding_cache),
            "metadata_entries": len(self.index_to_metadata),
            "embedding_dimension": self.embedding_dimension
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        with self.lock:
            self.embedding_cache.clear()
            logger.info("Cleared embedding cache")
    
    def rebuild_index(self, 
                     all_embeddings: List[List[float]], 
                     all_metadata: List[Dict[str, Any]]) -> bool:
        """
        Rebuild FAISS index from scratch
        
        Args:
            all_embeddings: List of all embeddings
            all_metadata: List of all metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(all_embeddings) != len(all_metadata):
                raise ValueError("Number of embeddings must match number of metadata entries")
            
            # Create new index
            self.vector_index = faiss.IndexFlatIP(self.embedding_dimension)
            
            # Clear metadata
            with self.lock:
                self.index_to_metadata.clear()
                self.metadata_store.clear()
            
            # Add all embeddings
            if all_embeddings:
                embedding_matrix = np.array(all_embeddings, dtype=np.float32)
                self.vector_index.add(embedding_matrix)
                
                # Store metadata
                with self.lock:
                    for i, metadata in enumerate(all_metadata):
                        self.index_to_metadata[i] = metadata
                        metadata_key = f"rebuilt_{i}"
                        self.metadata_store[metadata_key] = metadata
            
            logger.info(f"Rebuilt index with {len(all_embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            return False 