# SVL Knowledge Base System

A comprehensive semantic search and document processing system for the SVL (Stolen Vehicle Locator) chatbot, featuring PDF processing, Amazon Titan embeddings, and FAISS vector search.

## 🌟 Features

### Core Capabilities
- **PDF Document Processing**: Automatic extraction and processing of PDFs from organized folders
- **Semantic Search**: Advanced similarity search using Amazon Titan Text Embeddings
- **Vector Database**: FAISS-powered vector storage with metadata support
- **Multi-Category Support**: Organized document categories (SOP, FAQ, Process Docs, Contact)
- **Intelligent Ranking**: Context-aware result ranking with category weights
- **Caching System**: Performance optimization with intelligent caching
- **AI Integration**: Seamless integration with Amazon Nova Pro and Claude models

### Architecture Components
- **PDFDocumentManager**: S3 integration and robust PDF text extraction
- **EmbeddingManager**: Amazon Titan embeddings with batch processing
- **SemanticSearchEngine**: FAISS vector search with relevance ranking
- **KnowledgeBase**: Main interface for AI-powered document queries
- **Lambda Integration**: Serverless deployment support

## 📁 Document Organization

The system processes PDFs organized in the following structure:

```
data/
├── sop/                    # Standard Operating Procedures
│   ├── stolen_vehicle_report_sop.pdf
│   └── emergency_procedures.pdf
├── process_docs/           # Process Documentation  
│   ├── recovery_process_explanation.pdf
│   └── mobile_app_instructions.pdf
├── faq/                   # Frequently Asked Questions
│   ├── customer_faq.pdf
│   └── pricing_information.pdf
└── contact/               # Contact Information
    └── contact_information.pdf
```

**S3 Bucket Structure:**
```
svl-knowledge-base/
├── documents/             # PDF storage maintaining folder structure
├── embeddings/           # Vector embeddings and metadata
└── metadata/             # Document processing logs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `PyPDF2>=3.0.1` - PDF text extraction
- `pdfplumber>=0.10.0` - Advanced PDF processing
- `faiss-cpu>=1.7.4` - Vector similarity search
- `numpy>=1.24.0` - Numerical operations
- `boto3` - AWS SDK
- `streamlit` - Web interface

### 2. Environment Setup

Create a `.env` file or set environment variables:

```bash
# Required AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Knowledge Base Configuration
KNOWLEDGE_BASE_S3_BUCKET=svl-knowledge-base
KNOWLEDGE_BASE_INDEX_PATH=./knowledge_base_index

# Bedrock Configuration
BEDROCK_MODEL_ID=amazon.nova-pro-v1:0
```

### 3. Initialize Knowledge Base

```bash
# Quick initialization
python test_knowledge_base.py --init

# Full initialization with performance testing
python test_knowledge_base.py --init --performance --report kb_report.txt

# Force rebuild of index
python test_knowledge_base.py --init --force-rebuild
```

### 4. Test the System

```bash
# Health check
python test_knowledge_base.py --health

# Test specific query
python test_knowledge_base.py --query "How to report stolen vehicle" --query-type process

# Run functional tests
python test_knowledge_base.py --test
```

## 💻 Usage Examples

### Basic Python Usage

```python
from utils.knowledge_base import KnowledgeBase

# Initialize knowledge base
kb = KnowledgeBase()

# Initialize if needed
if not kb.is_ready():
    result = kb.initialize_knowledge_base()
    print(f"Initialization: {result['status']}")

# Query the knowledge base
result = kb.query_knowledge_base(
    user_query="How do I report a stolen vehicle?",
    query_type="process",
    include_context=True
)

print(f"Found {result['total_results']} results")
print(f"Context: {result['context']}")
```

### Advanced Query Types

```python
# Standard Operating Procedures
sop_result = kb.query_knowledge_base(
    user_query="Emergency procedures for stolen vehicle",
    query_type="sop",
    similarity_threshold=0.75
)

# Frequently Asked Questions
faq_result = kb.query_knowledge_base(
    user_query="What does the service cost?",
    query_type="faq",
    max_results=3
)

# Contact Information
contact_result = kb.query_knowledge_base(
    user_query="Who can I call for help?",
    query_type="contact",
    similarity_threshold=0.8
)
```

### Integration with Conversation Manager

The knowledge base automatically integrates with the conversation manager:

```python
from utils.conversation_manager import ConversationManager

# Conversation manager automatically uses knowledge base
manager = ConversationManager(session_id="test_session")

# Knowledge base context is automatically included
response = manager.process_user_input(
    user_input="How does the recovery process work?",
    phase="process_explanation"
)
```

## 🏗️ Architecture Deep Dive

### Document Processing Pipeline

1. **PDF Upload**: Local PDFs → S3 bucket with organized structure
2. **Text Extraction**: Multiple extraction methods (PyPDF2, pdfplumber) with quality assessment
3. **Chunking**: Semantic segmentation (800 chars with 100 char overlap)
4. **Embedding Generation**: Amazon Titan Text Embeddings with batch processing
5. **Vector Storage**: FAISS index with metadata mapping
6. **Search Interface**: Semantic similarity search with relevance ranking

### Query Processing Flow

1. **Query Analysis**: Determine query type based on keywords and context
2. **Embedding Generation**: Convert query to vector using Amazon Titan
3. **Vector Search**: FAISS similarity search with configurable thresholds
4. **Result Filtering**: Category-based filtering and weight application
5. **Context Formation**: Combine results into AI-ready context
6. **Response Enhancement**: Add query-specific instructions for AI models

### Performance Optimizations

- **Embedding Caching**: LRU cache for frequently accessed embeddings
- **Search Result Caching**: Query result caching with intelligent invalidation
- **Batch Processing**: Efficient batch embedding generation
- **Index Persistence**: Save/load FAISS indices to disk
- **Lazy Loading**: Load components only when needed

## 🔧 Configuration

### Knowledge Base Settings

```python
# Default configuration in KnowledgeBase class
config = {
    "auto_initialize": True,
    "cache_enabled": True,
    "default_query_type": "general",
    "max_context_length": 3000,
    "enable_response_enhancement": True
}
```

### Query Type Configuration

```python
query_type_config = {
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
    }
}
```

### Category Weights

```python
category_weights = {
    "sop": 1.2,          # Standard Operating Procedures - higher priority
    "process_docs": 1.1,  # Process Documentation - slightly higher
    "faq": 1.0,          # FAQ - normal priority
    "contact": 0.9       # Contact info - lower priority
}
```

## 🚀 Lambda Deployment

### Serverless Architecture

The knowledge base supports serverless deployment with Lambda functions:

```bash
# Deploy Lambda functions
cd lambda
./deploy.sh

# Configure environment variables
export USE_LAMBDA_BACKEND=true
export API_GATEWAY_URL=https://your-api-gateway-url
```

### Lambda Function Structure

- **bedrock_orchestrator.py**: Main conversation processing
- **knowledge_base_query.py**: Semantic search with fallback
- **request_preprocessor.py**: Input validation and sanitization
- **response_formatter.py**: Output formatting
- **lambda_orchestrator.py**: Multi-function workflow coordination

## 📊 Monitoring and Analytics

### Performance Metrics

```python
# Get comprehensive statistics
stats = kb.get_knowledge_base_stats()

print(f"Total queries: {stats['status']['total_queries']}")
print(f"Index size: {stats['search_engine']['embeddings']['index_size']}")
print(f"Cache hit rate: {stats['search_engine']['embeddings']['cache_hits']}")
```

### Health Monitoring

```python
# Check system health
health = kb.get_health_status()

print(f"Status: {health['status']}")
print(f"Ready: {health['initialized']}")
print(f"Error count: {health['error_count']}")
```

## 🔍 Testing and Validation

### Automated Testing

```bash
# Run comprehensive test suite
python test_knowledge_base.py --init --performance --report detailed_report.txt

# Quick functional tests
python test_knowledge_base.py --test

# Performance benchmarking
python -c "
from utils.knowledge_base_initializer import KnowledgeBaseInitializer
initializer = KnowledgeBaseInitializer()
results = initializer.initialize_and_test(run_performance_tests=True)
print(f'Overall status: {results[\"overall_status\"]}')
"
```

### Test Scenarios

The system includes comprehensive test scenarios:

- **Document Processing**: PDF extraction quality and chunking accuracy
- **Embedding Generation**: Vector quality and similarity calculations
- **Search Relevance**: Query result accuracy and ranking
- **Performance**: Query speed, cache efficiency, concurrent handling
- **Integration**: End-to-end conversation flow testing

## 🛠️ Troubleshooting

### Common Issues

**1. Knowledge Base Not Ready**
```bash
# Check initialization status
python test_knowledge_base.py --health

# Force reinitialize
python test_knowledge_base.py --init --force-rebuild
```

**2. Poor Search Results**
```python
# Adjust similarity threshold
result = kb.query_knowledge_base(
    user_query="your query",
    similarity_threshold=0.6  # Lower threshold for more results
)

# Try different query types
result = kb.query_knowledge_base(
    user_query="your query",
    query_type="general"  # Broader category search
)
```

**3. Slow Performance**
```python
# Clear caches
kb.clear_cache()

# Check index size and optimize
stats = kb.get_knowledge_base_stats()
if stats['search_engine']['embeddings']['index_size'] > 10000:
    # Consider rebuilding with smaller chunks
    pass
```

**4. AWS Connection Issues**
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check Bedrock access
aws bedrock list-foundation-models --region us-east-1
```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use initializer for detailed diagnostics
from utils.knowledge_base_initializer import KnowledgeBaseInitializer
initializer = KnowledgeBaseInitializer()
results = initializer.initialize_and_test(force_rebuild=True)
print(initializer.generate_report(results))
```

## 📈 Performance Benchmarks

### Expected Performance Metrics

- **Initialization Time**: 30-120 seconds (depending on document count)
- **Query Response Time**: < 2 seconds for cached queries, < 5 seconds for new queries
- **Index Build Time**: 2-5 minutes for 10-50 documents
- **Memory Usage**: 100-500 MB depending on index size
- **Cache Hit Rate**: 60-80% for repeated queries

### Optimization Recommendations

1. **Document Optimization**: Keep PDFs under 50 pages for optimal processing
2. **Chunk Size Tuning**: Balance between context (larger chunks) and precision (smaller chunks)
3. **Cache Configuration**: Increase cache size for high-traffic scenarios
4. **Batch Processing**: Process multiple documents in batches for efficiency
5. **Index Persistence**: Save indices to disk to avoid rebuilding

## 🔮 Future Enhancements

### Planned Features

- **Multi-language Support**: Process documents in multiple languages
- **Advanced Document Types**: Support for Word docs, PowerPoint, and web pages
- **Real-time Updates**: Live document monitoring and automatic reindexing
- **Advanced Analytics**: Query analytics and search optimization insights
- **Bedrock Agents Integration**: Native integration with Amazon Bedrock Agents

### Migration Path to Bedrock Agents

The architecture is designed for easy migration to Amazon Bedrock Agents:

1. **Phase 1**: Current semantic search system (implemented)
2. **Phase 2**: Bedrock Knowledge Base integration
3. **Phase 3**: Full Bedrock Agents deployment
4. **Phase 4**: Advanced agent capabilities and workflows

## 📚 API Reference

### KnowledgeBase Class

```python
class KnowledgeBase:
    def __init__(self, s3_bucket_name=None, index_storage_path="./knowledge_base_index", aws_region="us-east-1")
    def initialize_knowledge_base(self, force_rebuild=False) -> Dict[str, Any]
    def query_knowledge_base(self, user_query, query_type="general", include_context=True, max_results=None, similarity_threshold=None) -> Dict[str, Any]
    def update_knowledge_base(self, new_pdf_path=None, s3_key=None) -> Dict[str, Any]
    def get_knowledge_base_stats(self) -> Dict[str, Any]
    def get_health_status(self) -> Dict[str, Any]
    def is_ready(self) -> bool
```

### Query Response Format

```python
{
    "status": "success",
    "query": "user query",
    "query_type": "general",
    "results": [
        {
            "rank": 1,
            "relevance_score": 0.95,
            "similarity_score": 0.92,
            "text": "relevant document chunk",
            "document": {
                "filename": "document.pdf",
                "category": "sop",
                "source": "s3://bucket/documents/sop/document.pdf"
            },
            "chunk_info": {
                "chunk_id": 0,
                "length": 800,
                "word_count": 120
            }
        }
    ],
    "total_results": 3,
    "context": "formatted context for AI",
    "sources": ["document1.pdf", "document2.pdf"],
    "confidence": 0.87,
    "search_time": 0.45
}
```

## 📄 License

This knowledge base system is part of the SVL Chatbot project. See the main project LICENSE file for details.

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility when possible
5. Test performance impact of changes

---

**Need Help?** Check the troubleshooting section above or run `python test_knowledge_base.py --help` for available testing options. 