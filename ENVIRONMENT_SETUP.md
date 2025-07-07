# Environment Variables Setup

To set up the SVL Chatbot, you'll need to create a .env file with the following variables:

## Required Variables:
- AWS_ACCESS_KEY_ID=your_aws_access_key
- AWS_SECRET_ACCESS_KEY=your_aws_secret_key  
- AWS_REGION=us-east-1
- BEDROCK_MODEL_ID=amazon.nova-pro-v1:0
- DYNAMODB_TABLE_PREFIX=svl
- ENCRYPTION_KEY=your_32_character_key

## Knowledge Base Variables:
- KNOWLEDGE_BASE_S3_BUCKET=svl-knowledge-base
- KNOWLEDGE_BASE_INDEX_PATH=./knowledge_base_index

## Lambda Architecture Variables (Optional):
- USE_LAMBDA_BACKEND=false  # Set to true to use Lambda backend
- API_GATEWAY_URL=https://your-api-gateway-url  # For HTTP API calls
- LAMBDA_STAGE=dev  # Deployment stage
- KNOWLEDGE_BASE_ID=your_knowledge_base_id  # For Bedrock Knowledge Bases

See README.md for detailed setup instructions.
