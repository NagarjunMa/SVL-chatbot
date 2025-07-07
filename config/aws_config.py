import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"]
    )

def get_model_id():
    """
    Get the Bedrock model ID from environment variables.
    Defaults to Amazon Nova Pro if not specified.
    
    Supported models:
    - amazon.nova-pro-v1:0 (Amazon Nova Pro)
    - anthropic.claude-3-5-sonnet-20241022-v2:0 (Claude 3.5 Sonnet)
    - anthropic.claude-3-haiku-20240307-v1:0 (Claude 3 Haiku)
    """
    return os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

def is_nova_model(model_id: str = None) -> bool:
    """Check if the current model is Amazon Nova."""
    if model_id is None:
        model_id = get_model_id()
    return "amazon.nova" in model_id

def is_claude_model(model_id: str = None) -> bool:
    """Check if the current model is Anthropic Claude."""
    if model_id is None:
        model_id = get_model_id()
    return "anthropic.claude" in model_id 