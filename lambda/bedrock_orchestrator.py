"""
AWS Lambda: Bedrock Orchestration Handler
Main entry point for SVL chatbot requests - lightweight orchestration for Bedrock
Designed for easy migration to Bedrock Agents
"""

import json
import boto3
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
DYNAMODB_TABLE_PREFIX = os.environ.get("DYNAMODB_TABLE_PREFIX", "svl")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# AWS clients
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

def lambda_handler(event, context):
    """
    Main Lambda handler for Bedrock orchestration
    Lightweight entry point for conversation processing
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Parse request
        request_data = parse_request(event)
        session_id = request_data.get("session_id")
        user_input = request_data.get("user_input")
        conversation_phase = request_data.get("phase", "greeting")
        
        # Validate input
        if not user_input or not session_id:
            return create_error_response(400, "Missing required parameters: user_input, session_id")
        
        # Process conversation
        response = process_conversation(
            session_id=session_id,
            user_input=user_input,
            phase=conversation_phase,
            event_context=request_data
        )
        
        logger.info(f"Generated response: {response[:100]}...")
        
        return create_success_response(response, session_id)
        
    except Exception as e:
        logger.error(f"Error in bedrock orchestrator: {str(e)}")
        return create_error_response(500, "Internal server error")

def parse_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse incoming Lambda event - supports API Gateway and direct invocation
    """
    # Handle API Gateway event
    if "body" in event:
        if isinstance(event["body"], str):
            body = json.loads(event["body"])
        else:
            body = event["body"]
    else:
        # Direct invocation
        body = event
    
    return {
        "session_id": body.get("session_id") or generate_session_id(),
        "user_input": body.get("user_input", "").strip(),
        "phase": body.get("phase", "greeting"),
        "user_id": body.get("user_id"),
        "context": body.get("context", {})
    }

def process_conversation(session_id: str, user_input: str, phase: str, event_context: Dict[str, Any]) -> str:
    """
    Orchestrate conversation processing
    Lightweight orchestration - delegates complex logic to Bedrock
    """
    
    # 1. Retrieve conversation context (minimal processing)
    conversation_context = get_conversation_context(session_id)
    
    # 2. Build prompt for Bedrock (delegate complexity to AI)
    prompt = build_bedrock_prompt(
        user_input=user_input,
        phase=phase,
        context=conversation_context,
        additional_context=event_context.get("context", {})
    )
    
    # 3. Call Bedrock (main processing)
    bedrock_response = invoke_bedrock_model(prompt)
    
    # 4. Update conversation history (minimal storage)
    update_conversation_history(session_id, user_input, bedrock_response, phase)
    
    return bedrock_response

def get_conversation_context(session_id: str) -> Dict[str, Any]:
    """
    Retrieve minimal conversation context from DynamoDB
    Lightweight - just recent messages for context
    """
    try:
        conversations_table = dynamodb.Table(f"{DYNAMODB_TABLE_PREFIX}-conversations")
        
        response = conversations_table.query(
            KeyConditionExpression="conversation_id = :session_id",
            ExpressionAttributeValues={":session_id": session_id},
            Limit=10,  # Only recent messages for context
            ScanIndexForward=False  # Most recent first
        )
        
        items = response.get("Items", [])
        if not items:
            return {"messages": [], "is_new": True}
        
        # Extract recent messages (lightweight processing)
        messages = []
        for item in reversed(items):  # Chronological order
            if "messages" in item:
                messages.extend(item["messages"][-5:])  # Last 5 messages
        
        return {
            "messages": messages[-10:],  # Keep last 10 total
            "is_new": False,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.warning(f"Could not retrieve conversation context: {e}")
        return {"messages": [], "is_new": True}

def build_bedrock_prompt(user_input: str, phase: str, context: Dict[str, Any], additional_context: Dict[str, Any]) -> str:
    """
    Build prompt for Bedrock - delegate complex logic to AI model
    Minimal prompt engineering - let Bedrock handle complexity
    """
    
    # Base system prompt (minimal instructions)
    system_prompt = """You are SVL, a professional assistant for stolen vehicle recovery.
Be empathetic, concise, and helpful. Guide users through vehicle reporting when needed.
Never request sensitive personal information in chat."""
    
    # Phase-specific guidance (minimal)
    phase_guidance = {
        "greeting": "Greet warmly and offer assistance.",
        "collect_info": "Guide through vehicle reporting process.",
        "process_explanation": "Explain next steps clearly.",
        "faq": "Answer questions about vehicle theft recovery.",
        "confirmation": "Confirm details and provide next steps."
    }.get(phase, "Respond helpfully and professionally.")
    
    # Build conversation history (lightweight)
    conversation_history = ""
    if context.get("messages"):
        history_lines = []
        for msg in context["messages"][-5:]:  # Last 5 messages
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            if role and content:
                history_lines.append(f"{role}: {content}")
        conversation_history = "\n".join(history_lines)
    
    # Construct prompt (delegate complexity to Bedrock)
    prompt = f"""{system_prompt}

Current phase: {phase}
Guidance: {phase_guidance}

Recent conversation:
{conversation_history or "This is the start of the conversation."}

User's message: {user_input}

SVL Assistant:"""
    
    return prompt

def invoke_bedrock_model(prompt: str) -> str:
    """
    Call Bedrock model - main AI processing happens here
    Lightweight wrapper around Bedrock API
    """
    try:
        # Determine model format
        if "amazon.nova" in BEDROCK_MODEL_ID:
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        elif "anthropic.claude" in BEDROCK_MODEL_ID:
            body = {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": 500,
                "temperature": 0.7,
                "top_p": 0.9
            }
        else:
            body = {
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.7
            }
        
        # Call Bedrock
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body),
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response["body"].read())
        
        # Extract text based on model
        if "amazon.nova" in BEDROCK_MODEL_ID:
            return response_body["output"]["message"]["content"][0]["text"]
        elif "anthropic.claude" in BEDROCK_MODEL_ID:
            return response_body["completion"]
        else:
            return response_body.get("text", response_body.get("completion", ""))
            
    except Exception as e:
        logger.error(f"Bedrock invocation failed: {e}")
        return "I'm having trouble processing your request right now. Please try again."

def update_conversation_history(session_id: str, user_input: str, assistant_response: str, phase: str):
    """
    Update conversation history in DynamoDB
    Lightweight storage - minimal processing
    """
    try:
        conversations_table = dynamodb.Table(f"{DYNAMODB_TABLE_PREFIX}-conversations")
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Simple message structure
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        }
        
        assistant_message = {
            "role": "assistant", 
            "content": assistant_response,
            "timestamp": timestamp
        }
        
        # Store conversation turn
        conversations_table.put_item(
            Item={
                "conversation_id": session_id,
                "timestamp": timestamp,
                "messages": [user_message, assistant_message],
                "phase": phase,
                "updated_at": timestamp
            }
        )
        
        logger.info(f"Updated conversation history for session: {session_id}")
        
    except Exception as e:
        logger.warning(f"Failed to update conversation history: {e}")

def generate_session_id() -> str:
    """Generate unique session ID"""
    return f"session-{uuid.uuid4().hex[:12]}"

def create_success_response(response_text: str, session_id: str) -> Dict[str, Any]:
    """Create successful Lambda response"""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "success": True,
            "response": response_text,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    }

def create_error_response(status_code: int, error_message: str) -> Dict[str, Any]:
    """Create error Lambda response"""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "success": False,
            "error": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    } 