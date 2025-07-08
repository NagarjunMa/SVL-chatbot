# SVL Chatbot - Production Ready

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CloudWatch Monitoring
- **Dashboard**: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=SVL-Chatbot-Comprehensive-Monitoring
- **Conversations**: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/%2Faws%2Fsvl%2Fconversation
- **Setup**: `python infrastructure/cloudwatch_setup.py`

## Features
- Stolen vehicle reporting and tracking
- AWS Bedrock AI integration
- Knowledge base with PDF processing
- Comprehensive CloudWatch logging with PII masking
- Real-time conversation monitoring
- Security and compliance features

## Key Components
- `app.py` - Main Streamlit application
- `utils/` - Core functionality (conversation, security, observability)
- `data/` - Knowledge base documents and models
- `infrastructure/` - AWS setup scripts
- `config/` - Configuration files

## Environment Variables
```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret  
AWS_DEFAULT_REGION=us-east-1
```

## Production Deployment

### Prerequisites
- Python 3.9+ installed
- AWS CLI configured with your credentials
- AWS account with Bedrock access enabled

### Step-by-Step Setup
1. **Clone and Install Dependencies**
   ```bash
   git clone <your-repo-url>
   cd svl-chatbot
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials and configuration
   ```

3. **Set up AWS Infrastructure**
   ```bash
   python infrastructure/aws_setup.py
   python infrastructure/cloudwatch_setup.py
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

### Required AWS Permissions
Your AWS credentials need permissions for:
- Bedrock (invoke models, create knowledge bases)
- DynamoDB (create tables, read/write)
- S3 (create buckets, upload files)
- OpenSearch (create domains)
- CloudWatch (create log groups, metrics)
- IAM (create roles for services)

### Deployment Verification
After setup, verify everything works:

1. **Test Application Launch**
   ```bash
   streamlit run app.py
   # Should start without errors on http://localhost:8501
   ```

2. **Verify AWS Services**
   ```bash
   # Check if DynamoDB tables exist
   aws dynamodb list-tables --region us-east-1
   
   # Check CloudWatch log groups
   aws logs describe-log-groups --region us-east-1 | grep svl
   ```

3. **Test Basic Functionality**
   - Open http://localhost:8501
   - Send a test message like "Hello"
   - Check if response is generated
   - Verify logs appear in CloudWatch

### Troubleshooting Common Issues

#### 1. AWS Credentials Issues
```bash
# Error: Unable to locate credentials
# Solution: Configure AWS credentials
aws configure
# OR set environment variables in .env file
```

#### 2. Missing Dependencies
```bash
# Error: ModuleNotFoundError
# Solution: Install all dependencies
pip install -r requirements.txt
```

#### 3. Bedrock Access Issues
```bash
# Error: Access denied for Bedrock
# Solution: Request Bedrock access in AWS Console
# Go to AWS Bedrock console -> Model access -> Request access
```

#### 4. DynamoDB Permission Issues
```bash
# Error: User not authorized to perform dynamodb operations
# Solution: Add DynamoDB permissions to your IAM user/role
```

#### 5. CloudWatch Setup Issues
```bash
# Error: Log group creation failed
# Solution: Ensure CloudWatch permissions and run setup
python infrastructure/cloudwatch_setup.py
```

### Team Deployment Checklist

Before sharing with your team, ensure:
- [ ] `.env.example` file exists with all required variables
- [ ] `requirements.txt` contains all dependencies
- [ ] Infrastructure setup scripts work
- [ ] README has clear deployment instructions
- [ ] AWS permissions are documented
- [ ] Troubleshooting guide is complete

## Monitoring Commands
```bash
# Live conversation logs
aws logs tail "/aws/svl/conversation" --follow --region us-east-1

# Search logs
aws logs filter-log-events --log-group-name "/aws/svl/conversation" --filter-pattern "{ $.query_type = \"pricing_inquiry\" }" --region us-east-1
```

## Model Support

The application supports multiple AWS Bedrock models with automatic format detection:

### Amazon Nova Pro (Default)
- **Model ID**: `amazon.nova-pro-v1:0`
- **Features**: High-quality text generation, multimodal capabilities
- **Best for**: Production use, complex conversations

### Anthropic Claude Models
- **Claude 3.5 Sonnet**: `anthropic.claude-3-5-sonnet-20241022-v2:0`
- **Claude 3 Haiku**: `anthropic.claude-3-haiku-20240307-v1:0`
- **Features**: Excellent reasoning, safety, and helpfulness

To switch models, simply update the `BEDROCK_MODEL_ID` environment variable and restart the application.

## Project Structure
```
svl-chatbot/
├── app.py
├── config/
│   ├── aws_config.py
│   └── database_config.py
├── utils/
│   ├── aws_bedrock.py
│   ├── conversation_manager.py
│   ├── database_manager.py
│   ├── data_utils.py
│   ├── chat_utils.py
│   ├── logger.py
│   └── custom.css
├── data/
│   ├── models.py
│   └── knowledge_base.json
├── requirements.txt
├── .env.example
└── README.md
```

## AWS Services Setup
- **AWS Bedrock**: Ensure you have access to AWS Bedrock and the necessary permissions.
- **DynamoDB**: Tables will be created automatically with proper schemas and indexes.
- **IAM Permissions**: Ensure your AWS credentials have permissions for:
  - DynamoDB (create, read, update, delete)
  - Bedrock (invoke model)

## Data Management Features
- **Conversation Storage**: All chat interactions are stored in DynamoDB with conversation history.
- **Ticket Generation**: Unique ticket IDs (SVL-YYYYMMDD-HHMMSS-XXX) for each stolen vehicle report.
- **PII Protection**: Automatic detection and masking of sensitive information.
- **Data Validation**: Comprehensive validation for VIN, phone numbers, emails, etc.
- **Encryption**: Sensitive data (phone, email, address) is encrypted at rest.
- **Audit Logging**: All data access and modifications are logged for compliance.
- **Data Export**: Export conversations and tickets in CSV/JSON formats.

## Security & Compliance
- No secrets are committed; all sensitive data is managed via environment variables.
- PII detection and masking for privacy protection.
- Encryption for sensitive data storage.
- Comprehensive audit logging for compliance.
- Data retention policies configurable via environment variables.

## Database Schema
- **Conversations Table**: Stores chat history with user sessions
- **Tickets Table**: Stores stolen vehicle reports with encrypted sensitive data
- **Indexes**: Optimized for querying by user, status, and date ranges

## Further Reading
- [Streamlit Documentation](https://docs.streamlit.io/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
- [Python Logging](https://docs.python.org/3/library/logging.html)

## Observability & Monitoring

### CloudWatch Log Groups
- `/aws/svl/conversation` - All user interactions (PII-masked)
- `/aws/svl/api-requests` - API call metrics  
- `/aws/svl/errors` - Error tracking

### Quick Access
- **Dashboard**: [SVL CloudWatch Dashboard](https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=SVL-Chatbot-Comprehensive-Monitoring)
- **Live Logs**: `aws logs tail "/aws/svl/conversation" --follow --region us-east-1`

All logs automatically mask PII (SSN, phone, email, names, addresses). 