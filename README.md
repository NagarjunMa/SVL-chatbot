# Stolen Vehicle Locator (SVL) Chatbot

A professional AI-powered chatbot for reporting and tracking stolen vehicles, built with Streamlit and AWS Bedrock.

## Features

- **Professional Chat Interface**: Clean, responsive design with typing animations
- **Multi-Step Vehicle Reporting**: Guided form for comprehensive incident documentation
- **AI-Powered Conversations**: Integrated with AWS Bedrock supporting multiple models:
  - Amazon Nova Pro (default)
  - Anthropic Claude 3.5 Sonnet
  - Anthropic Claude 3 Haiku
- **Comprehensive Data Management**: DynamoDB integration with encrypted sensitive data
- **PII Protection**: Automatic detection, masking, and secure handling of sensitive information
- **Audit Trails**: Complete logging and compliance tracking
- **Data Export**: CSV and JSON export capabilities
- **Session Management**: Persistent conversations with memory
- **Error Handling**: Robust error handling with retry logic

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd svl-chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```env
   # AWS Configuration
   AWS_ACCESS_KEY_ID=your_aws_access_key_here
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
   AWS_REGION=us-east-1

   # Bedrock Model Configuration
   # For Amazon Nova Pro (default): amazon.nova-pro-v1:0
   # For Claude 3.5 Sonnet: anthropic.claude-3-5-sonnet-20241022-v2:0
   # For Claude 3 Haiku: anthropic.claude-3-haiku-20240307-v1:0
   BEDROCK_MODEL_ID=amazon.nova-pro-v1:0

   # Database Configuration
   DYNAMODB_TABLE_PREFIX=svl

   # Encryption Configuration (32 character key)
   ENCRYPTION_KEY=your_32_character_encryption_key

   # Application Configuration
   DEBUG=false
   LOG_LEVEL=INFO
   ```

4. **Set up AWS credentials**:
   Ensure your AWS credentials have access to:
   - Amazon Bedrock (for AI model access)
   - DynamoDB (for data storage)

5. **Run the application**:
   ```bash
   streamlit run app.py
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