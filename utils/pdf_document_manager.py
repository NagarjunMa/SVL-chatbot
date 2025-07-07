"""
PDF Document Manager for SVL Knowledge Base
Handles PDF storage, processing, and text extraction with S3 integration
"""

import os
import boto3
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path

try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print("Warning: PDF processing libraries not installed. Run: pip install PyPDF2 pdfplumber")

from utils.logger import get_logger

logger = get_logger("pdf_document_manager")

class PDFDocumentManager:
    """
    Manages PDF documents for the SVL knowledge base
    Handles S3 storage, text extraction, and document chunking
    """
    
    def __init__(self, s3_bucket_name: str, aws_region: str = "us-east-1"):
        """
        Initialize PDF Document Manager
        
        Args:
            s3_bucket_name: S3 bucket for document storage
            aws_region: AWS region for S3 operations
        """
        self.s3_bucket_name = s3_bucket_name
        self.aws_region = aws_region
        
        try:
            self.s3_client = boto3.client('s3', region_name=aws_region)
            self._ensure_bucket_exists()
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
        
        # Document categories and their descriptions
        self.document_categories = {
            "sop": "Standard Operating Procedures",
            "process_docs": "Process Documentation",
            "faq": "Frequently Asked Questions",
            "contact": "Contact Information"
        }
        
    def _ensure_bucket_exists(self):
        """Ensure S3 bucket exists, create if it doesn't"""
        try:
            self.s3_client.head_bucket(Bucket=self.s3_bucket_name)
            logger.info(f"S3 bucket {self.s3_bucket_name} exists")
        except Exception:
            try:
                if self.aws_region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.s3_bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.s3_bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.aws_region}
                    )
                logger.info(f"Created S3 bucket {self.s3_bucket_name}")
            except Exception as e:
                logger.error(f"Failed to create S3 bucket: {e}")
                raise
    
    def upload_pdfs_from_data_folder(self, local_data_path: str = './data') -> Dict[str, List[str]]:
        """
        Upload all PDFs from local data folder to S3 maintaining folder structure
        
        Args:
            local_data_path: Path to local data folder
            
        Returns:
            Dictionary mapping categories to uploaded file lists
        """
        uploaded_files = {}
        
        try:
            data_path = Path(local_data_path)
            if not data_path.exists():
                logger.warning(f"Data folder {local_data_path} does not exist")
                return uploaded_files
            
            for category in self.document_categories.keys():
                category_path = data_path / category
                uploaded_files[category] = []
                
                if not category_path.exists():
                    logger.warning(f"Category folder {category_path} does not exist")
                    continue
                
                pdf_files = list(category_path.glob("*.pdf"))
                logger.info(f"Found {len(pdf_files)} PDF files in {category}")
                
                for pdf_file in pdf_files:
                    try:
                        s3_key = f"documents/{category}/{pdf_file.name}"
                        
                        # Check if file already exists and is unchanged
                        if self._is_file_unchanged(pdf_file, s3_key):
                            logger.info(f"File {pdf_file.name} unchanged, skipping upload")
                            uploaded_files[category].append(s3_key)
                            continue
                        
                        # Upload file to S3
                        self.s3_client.upload_file(
                            str(pdf_file),
                            self.s3_bucket_name,
                            s3_key,
                            ExtraArgs={
                                'Metadata': {
                                    'category': category,
                                    'original_filename': pdf_file.name,
                                    'upload_timestamp': datetime.now(timezone.utc).isoformat(),
                                    'file_size': str(pdf_file.stat().st_size)
                                }
                            }
                        )
                        
                        uploaded_files[category].append(s3_key)
                        logger.info(f"Uploaded {pdf_file.name} to {s3_key}")
                        
                    except Exception as e:
                        logger.error(f"Failed to upload {pdf_file.name}: {e}")
                        continue
            
            # Store upload metadata
            self._store_upload_metadata(uploaded_files)
            
            return uploaded_files
            
        except Exception as e:
            logger.error(f"Error uploading PDFs from data folder: {e}")
            return uploaded_files
    
    def _is_file_unchanged(self, local_file: Path, s3_key: str) -> bool:
        """
        Check if local file is unchanged compared to S3 version
        
        Args:
            local_file: Local file path
            s3_key: S3 object key
            
        Returns:
            True if file is unchanged, False otherwise
        """
        try:
            # Get local file hash
            local_hash = self._calculate_file_hash(local_file)
            
            # Get S3 object metadata
            response = self.s3_client.head_object(
                Bucket=self.s3_bucket_name,
                Key=s3_key
            )
            
            # Compare hashes if available
            s3_hash = response.get('Metadata', {}).get('file_hash')
            if s3_hash and s3_hash == local_hash:
                return True
                
            return False
            
        except Exception:
            # If file doesn't exist in S3 or any error, consider it changed
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _store_upload_metadata(self, uploaded_files: Dict[str, List[str]]):
        """Store metadata about uploaded files"""
        try:
            metadata = {
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "uploaded_files": uploaded_files,
                "total_files": sum(len(files) for files in uploaded_files.values())
            }
            
            metadata_key = f"metadata/upload_log_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Stored upload metadata at {metadata_key}")
            
        except Exception as e:
            logger.error(f"Failed to store upload metadata: {e}")
    
    def extract_text_from_pdf(self, s3_key: str = None, local_path: str = None) -> Dict[str, Any]:
        """
        Extract text from PDF using multiple parsers for robustness
        
        Args:
            s3_key: S3 key for PDF (if stored in S3)
            local_path: Local path to PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Determine source and get file content
            if s3_key:
                pdf_content = self._download_pdf_from_s3(s3_key)
                source = f"s3://{self.s3_bucket_name}/{s3_key}"
            elif local_path:
                with open(local_path, 'rb') as f:
                    pdf_content = f.read()
                source = local_path
            else:
                raise ValueError("Either s3_key or local_path must be provided")
            
            # Try multiple extraction methods
            extraction_results = []
            
            # Method 1: PyPDF2
            try:
                text_pypdf2 = self._extract_with_pypdf2(pdf_content)
                if text_pypdf2.strip():
                    extraction_results.append({
                        "method": "PyPDF2",
                        "text": text_pypdf2,
                        "quality_score": self._assess_text_quality(text_pypdf2)
                    })
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
            
            # Method 2: pdfplumber
            try:
                text_pdfplumber = self._extract_with_pdfplumber(pdf_content)
                if text_pdfplumber.strip():
                    extraction_results.append({
                        "method": "pdfplumber",
                        "text": text_pdfplumber,
                        "quality_score": self._assess_text_quality(text_pdfplumber)
                    })
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
            
            if not extraction_results:
                raise Exception("All PDF extraction methods failed")
            
            # Choose best extraction result
            best_result = max(extraction_results, key=lambda x: x["quality_score"])
            
            # Clean and preprocess text
            cleaned_text = self.clean_and_preprocess_text(best_result["text"])
            
            return {
                "source": source,
                "text": cleaned_text,
                "extraction_method": best_result["method"],
                "quality_score": best_result["quality_score"],
                "text_length": len(cleaned_text),
                "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                "all_methods_tried": [r["method"] for r in extraction_results]
            }
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def _download_pdf_from_s3(self, s3_key: str) -> bytes:
        """Download PDF content from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket_name,
                Key=s3_key
            )
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to download PDF from S3: {e}")
            raise
    
    def _extract_with_pypdf2(self, pdf_content: bytes) -> str:
        """Extract text using PyPDF2"""
        import io
        
        text_parts = []
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_parts.append(page.extract_text())
        
        return "\n".join(text_parts)
    
    def _extract_with_pdfplumber(self, pdf_content: bytes) -> str:
        """Extract text using pdfplumber"""
        import io
        
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        
        return "\n".join(text_parts)
    
    def _assess_text_quality(self, text: str) -> float:
        """
        Assess the quality of extracted text
        
        Args:
            text: Extracted text
            
        Returns:
            Quality score between 0 and 1
        """
        if not text.strip():
            return 0.0
        
        score = 0.0
        
        # Check for readable words
        words = text.split()
        if len(words) > 0:
            score += 0.3
        
        # Check for sentence structure
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            score += 0.2
        
        # Check for reasonable character distribution
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text)
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio > 0.6:  # Good ratio of alphabetic characters
                score += 0.3
        
        # Check for minimal strange characters
        strange_chars = sum(1 for c in text if ord(c) > 127 and not c.isspace())
        if strange_chars / total_chars < 0.1:  # Less than 10% strange characters
            score += 0.2
        
        return min(score, 1.0)
    
    def clean_and_preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Page numbers
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between merged words
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation
        
        # Remove non-printable characters except common ones
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        return text.strip()
    
    def chunk_document(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        Split document text into semantic chunks with metadata
        
        Args:
            text: Document text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        if not text.strip():
            return []
        
        chunks = []
        
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk_metadata(current_chunk, chunk_id, text))
                chunk_id += 1
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append(self._create_chunk_metadata(current_chunk, chunk_id, text))
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _create_chunk_metadata(self, chunk_text: str, chunk_id: int, full_text: str) -> Dict[str, Any]:
        """Create metadata for a document chunk"""
        return {
            "chunk_id": chunk_id,
            "text": chunk_text.strip(),
            "length": len(chunk_text),
            "word_count": len(chunk_text.split()),
            "position_in_document": chunk_text[:50] + "..." if len(chunk_text) > 50 else chunk_text,
            "created_timestamp": datetime.now(timezone.utc).isoformat(),
            "text_hash": hashlib.md5(chunk_text.encode()).hexdigest()
        }
    
    def get_document_list(self, category: str = None) -> List[Dict[str, Any]]:
        """
        Get list of documents in S3 bucket
        
        Args:
            category: Filter by document category (optional)
            
        Returns:
            List of document metadata
        """
        try:
            documents = []
            prefix = f"documents/{category}/" if category else "documents/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket_name,
                Prefix=prefix
            )
            
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.pdf'):
                    # Get object metadata
                    try:
                        head_response = self.s3_client.head_object(
                            Bucket=self.s3_bucket_name,
                            Key=obj['Key']
                        )
                        
                        documents.append({
                            "s3_key": obj['Key'],
                            "filename": obj['Key'].split('/')[-1],
                            "size": obj['Size'],
                            "last_modified": obj['LastModified'].isoformat(),
                            "category": obj['Key'].split('/')[1] if len(obj['Key'].split('/')) > 1 else 'unknown',
                            "metadata": head_response.get('Metadata', {})
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get metadata for {obj['Key']}: {e}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get document list: {e}")
            return []
    
    def delete_document(self, s3_key: str) -> bool:
        """
        Delete document from S3
        
        Args:
            s3_key: S3 key of document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.s3_bucket_name,
                Key=s3_key
            )
            logger.info(f"Deleted document {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {s3_key}: {e}")
            return False 