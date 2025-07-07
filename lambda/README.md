# SVL Chatbot - Lambda Architecture

## Overview

This directory contains a streamlined, serverless Lambda-based architecture for the SVL chatbot, specifically designed for **Bedrock orchestration** and **future migration to Bedrock Agents**. The architecture emphasizes minimal business logic in Lambda functions, clean separation of concerns, and efficient AWS service integration.

## Design Principles

### 1. **Lightweight Lambda Functions**
- **Minimal Business Logic**: Complex business logic is delegated to Bedrock models
- **Single Responsibility**: Each function has one clear purpose
- **Fast Cold Starts**: Minimal dependencies and optimized for quick initialization
- **Easy Testing**: Simple input/output patterns for reliable testing

### 2. **Bedrock-First Architecture**
- **AI-Driven Processing**: Bedrock handles conversation logic, decision making, and complex reasoning
- **Prompt Engineering**: Structured prompts guide Bedrock rather than hardcoded business rules
- **Model Flexibility**: Support for multiple Bedrock models (Nova Pro, Claude)
- **Context Management**: Lightweight context passing to Bedrock for informed responses

### 3. **Bedrock Agents Migration Readiness**
- **Clean API Interfaces**: Functions expose clear interfaces suitable for agent integration
- **Stateless Design**: Functions are stateless and easily replaceable by agents
- **Minimal Dependencies**: Reduced external dependencies for easier migration
- **Service-Oriented**: Each function can be independently replaced by Bedrock Agents

## Architecture Components

### Core Lambda Functions

#### 1. **Bedrock Orchestrator** (`bedrock_orchestrator.py`)
**Purpose**: Main entry point for conversation processing
- Coordinates conversation flow
- Calls Bedrock models with structured prompts
- Manages conversation context
- Handles model-specific API formats (Nova Pro, Claude)

**Migration Path**: This function is the **primary candidate** for Bedrock Agent replacement

#### 2. **Request Preprocessor** (`request_preprocessor.py`)
**Purpose**: Minimal input validation and sanitization
- Basic input validation (length, format)
- Simple PII detection (warning only)
- SQL injection prevention (basic)
- Input sanitization (minimal)

**Migration Path**: Keep as Lambda - basic validation better handled at the edge

#### 3. **Ticket Generator** (`ticket_generator.py`)
**Purpose**: Generate unique SVL ticket IDs and store basic ticket data
- Unique ID generation: `SVL-YYYYMMDD-HHMMSS-XXX`
- Simple ticket structure creation
- DynamoDB storage
- Minimal validation

**Migration Path**: **Good candidate** for Bedrock Agent with function calling

#### 4. **Knowledge Base Query** (`knowledge_base_query.py`)
**Purpose**: Context retrieval for Bedrock processing
- FAQ context retrieval
- Process information lookup
- Conversation history extraction
- Basic context formatting

**Migration Path**: **Excellent candidate** - will be replaced by Bedrock Knowledge Bases

#### 5. **Response Formatter** (`response_formatter.py`)
**Purpose**: Basic response formatting and structure
- Simple text cleaning
- Action item extraction (basic)
- Response type routing
- Minimal formatting

**Migration Path**: **Good candidate** for agent-based formatting

### Supporting Components

#### **Lambda Orchestrator** (`lambda_orchestrator.py`)
**Purpose**: Coordinates complex workflows between Lambda functions
- Parallel function execution
- Workflow management
- Error handling and fallbacks
- Performance optimization

**Migration Path**: Will be replaced by Bedrock Agent orchestration capabilities

## DynamoDB Design

### Tables

#### **Conversations Table** (`svl-{stage}-conversations`)
```
Primary Key: conversation_id (HASH) + timestamp (RANGE)
GSI: user_id-index (user_id HASH + timestamp RANGE)
```
- Stores conversation turns
- Session management
- Message history
- Optimized for recent message retrieval

#### **Tickets Table** (`svl-{stage}-tickets`)
```
Primary Key: ticket_id (HASH) + created_at (RANGE)
GSI: user_id-index, status-index
```
- Stores vehicle theft reports
- Unique SVL ticket IDs
- Status tracking
- User and status-based queries

### Optimization Features
- **Pay-per-request billing** for cost efficiency
- **Global Secondary Indexes** for flexible querying
- **DynamoDB Streams** for change tracking (future use)
- **Lightweight schema** for fast reads/writes

## Deployment

### Prerequisites
- Node.js and npm (for Serverless Framework)
- Python 3.9+
- AWS CLI configured
- Bedrock access in your AWS account

### Quick Deployment
```bash
cd lambda
./deploy.sh dev us-east-1
```

### Manual Deployment
```bash
cd infrastructure
npm install
serverless deploy --stage dev --region us-east-1
```

### Environment Variables
```bash
export BEDROCK_MODEL_ID="amazon.nova-pro-v1:0"
export KNOWLEDGE_BASE_ID=""  # Optional
```

## API Endpoints

### Chat Endpoint
```
POST /chat
{
  "user_input": "My car was stolen",
  "session_id": "session-abc123",
  "phase": "greeting"
}
```

### Ticket Creation
```
POST /ticket
{
  "session_id": "session-abc123",
  "vehicle_info": {
    "make": "Toyota",
    "model": "Camry",
    "year": 2020,
    "license_plate": "ABC123"
  }
}
```

## Bedrock Agents Migration Strategy

### Phase 1: Knowledge Base Integration (Immediate)
1. **Replace Knowledge Base Query Lambda** with Bedrock Knowledge Bases
2. **Update prompts** to reference knowledge base context
3. **Test knowledge retrieval** accuracy and performance

### Phase 2: Agent-Based Conversation (Short-term)
1. **Create Bedrock Agent** to replace Bedrock Orchestrator
2. **Implement function calling** for ticket generation
3. **Migrate conversation logic** to agent instructions
4. **Test agent conversation** quality and consistency

### Phase 3: Full Agent Architecture (Medium-term)
1. **Replace remaining Lambda functions** with agent actions
2. **Implement agent-to-agent** communication patterns
3. **Optimize agent prompts** and instructions
4. **Performance tuning** and cost optimization

### Phase 4: Advanced Capabilities (Long-term)
1. **Multi-modal processing** (images, documents)
2. **Complex workflow orchestration** via agents
3. **Real-time collaboration** between agents
4. **Advanced reasoning** and decision making

## Migration Readiness Assessment

### High Readiness (Easy Migration)
- ✅ **Bedrock Orchestrator**: Pure AI orchestration
- ✅ **Knowledge Base Query**: Natural fit for Knowledge Bases
- ✅ **Response Formatter**: Simple text processing

### Medium Readiness (Some Refactoring)
- ⚠️ **Ticket Generator**: Need function calling setup
- ⚠️ **Lambda Orchestrator**: Workflow migration needed

### Low Readiness (Keep as Lambda)
- 🔄 **Request Preprocessor**: Edge validation better as Lambda

## Performance Considerations

### Lambda Optimizations
- **Minimal dependencies**: Fast cold starts
- **Parallel execution**: Where possible using ThreadPoolExecutor
- **Connection reuse**: Boto3 client reuse
- **Memory optimization**: Right-sized memory allocation

### Cost Optimization
- **Pay-per-request DynamoDB**: Cost-effective for variable load
- **Lambda provisioned concurrency**: For production high-frequency usage
- **Bedrock model selection**: Choose appropriate model for cost/performance

### Monitoring
- **CloudWatch Logs**: Comprehensive logging for debugging
- **CloudWatch Metrics**: Performance and error tracking
- **X-Ray Tracing**: Request flow visualization (optional)

## Testing

### Unit Tests
```bash
cd lambda
python -m pytest tests/
```

### Integration Tests
```bash
cd lambda
python test_integration.py
```

### Load Testing
```bash
# Use artillery or similar tool
artillery run load-test.yml
```

## Security Considerations

### IAM Permissions
- **Least privilege principle**: Minimal required permissions
- **Function-specific roles**: Each Lambda has its own role
- **Resource-based policies**: Restrict access to specific resources

### Data Protection
- **PII Detection**: Basic PII detection in preprocessing
- **Encryption in transit**: HTTPS/TLS for all communications
- **Encryption at rest**: DynamoDB encryption enabled

### Network Security
- **VPC deployment**: Optional for enhanced security
- **Security groups**: Restrict Lambda network access
- **Private endpoints**: VPC endpoints for AWS services

## Troubleshooting

### Common Issues

#### Cold Start Performance
- **Solution**: Use provisioned concurrency for high-frequency functions
- **Monitoring**: Track cold start metrics in CloudWatch

#### Bedrock Throttling
- **Solution**: Implement exponential backoff and retry logic
- **Monitoring**: Monitor throttling metrics and adjust concurrency

#### DynamoDB Hot Partitions
- **Solution**: Design keys to distribute load evenly
- **Monitoring**: Use DynamoDB Insights to identify hot partitions

### Debugging Steps
1. **Check CloudWatch Logs** for error details
2. **Verify IAM permissions** for AWS service access
3. **Test individual functions** using AWS Lambda console
4. **Validate DynamoDB table** schemas and indexes
5. **Check Bedrock model availability** in your region

## Future Enhancements

### Planned Features
- **Bedrock Knowledge Base integration**
- **Multi-language support**
- **Advanced PII handling**
- **Real-time notifications**
- **Analytics and reporting**

### Agent Capabilities Roadmap
- **Function calling** for external integrations
- **Code interpretation** for data analysis
- **File processing** for document analysis
- **Multi-agent collaboration** for complex workflows

## Contributing

### Development Guidelines
1. **Keep functions lightweight** - minimal business logic
2. **Use type hints** for better code documentation
3. **Follow error handling patterns** established in existing functions
4. **Add comprehensive logging** for debugging
5. **Test thoroughly** before deployment

### Code Standards
- **PEP 8 compliance** for Python code
- **Docstring documentation** for all functions
- **Error handling** with proper logging
- **Type annotations** for function signatures

---

This architecture provides a solid foundation for the SVL chatbot while maintaining a clear path toward Bedrock Agents migration. The design emphasizes simplicity, performance, and migration readiness to ensure smooth evolution as AWS Bedrock capabilities expand. 