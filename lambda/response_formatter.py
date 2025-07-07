"""
AWS Lambda: Response Formatter
Basic response formatting for SVL chatbot
Lightweight formatting - complex logic delegated to Bedrock
Designed for easy Bedrock Agents integration
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import logging
import re

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Lambda handler for response formatting
    Lightweight formatting - minimal processing
    """
    try:
        logger.info(f"Formatting response: {json.dumps(event)}")
        
        # Parse request
        request_data = parse_request(event)
        raw_response = request_data.get("response", "")
        response_type = request_data.get("type", "chat")
        
        if not raw_response:
            return create_error_response(400, "Response content is required")
        
        # Format based on type (minimal processing)
        if response_type == "chat":
            formatted_response = format_chat_response(raw_response, request_data)
        elif response_type == "ticket":
            formatted_response = format_ticket_response(raw_response, request_data)
        elif response_type == "error":
            formatted_response = format_error_response(raw_response, request_data)
        else:
            formatted_response = format_generic_response(raw_response, request_data)
        
        logger.info("Response formatting completed")
        
        return create_success_response(formatted_response)
        
    except Exception as e:
        logger.error(f"Error in response formatter: {str(e)}")
        return create_error_response(500, "Response formatting failed")

def parse_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Lambda event for response formatting"""
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
        "response": body.get("response", ""),
        "type": body.get("type", "chat"),
        "session_id": body.get("session_id", ""),
        "context": body.get("context", {}),
        "metadata": body.get("metadata", {})
    }

def format_chat_response(response: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format chat response - basic formatting only
    Complex formatting delegated to Bedrock
    """
    
    # Basic text cleaning (minimal)
    cleaned_response = clean_response_text(response)
    
    # Add basic formatting markers (lightweight)
    formatted_text = add_basic_formatting(cleaned_response)
    
    # Extract action items if present (simple pattern matching)
    action_items = extract_action_items(formatted_text)
    
    return {
        "type": "chat",
        "content": formatted_text,
        "actions": action_items,
        "session_id": context.get("session_id", ""),
        "metadata": {
            "formatted_at": datetime.now(timezone.utc).isoformat(),
            "length": len(formatted_text),
            "has_actions": len(action_items) > 0
        }
    }

def format_ticket_response(response: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format ticket-related response - minimal structure
    """
    
    cleaned_response = clean_response_text(response)
    
    # Extract ticket ID if present (basic regex)
    ticket_id_match = re.search(r'SVL-\d{8}-\d{6}-\d{3}', cleaned_response)
    ticket_id = ticket_id_match.group(0) if ticket_id_match else None
    
    return {
        "type": "ticket",
        "content": cleaned_response,
        "ticket_id": ticket_id,
        "session_id": context.get("session_id", ""),
        "next_steps": extract_next_steps(cleaned_response),
        "metadata": {
            "formatted_at": datetime.now(timezone.utc).isoformat(),
            "has_ticket_id": ticket_id is not None
        }
    }

def format_error_response(response: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format error response - basic error structure
    """
    
    return {
        "type": "error",
        "content": clean_response_text(response),
        "session_id": context.get("session_id", ""),
        "suggested_actions": [
            "Please try rephrasing your question",
            "Contact support if the issue persists",
            "Check your internet connection"
        ],
        "metadata": {
            "formatted_at": datetime.now(timezone.utc).isoformat(),
            "is_error": True
        }
    }

def format_generic_response(response: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format generic response - minimal processing
    """
    
    return {
        "type": "generic",
        "content": clean_response_text(response),
        "session_id": context.get("session_id", ""),
        "metadata": {
            "formatted_at": datetime.now(timezone.utc).isoformat(),
            "length": len(response)
        }
    }

def clean_response_text(text: str) -> str:
    """
    Basic text cleaning - minimal processing
    Complex text processing delegated to Bedrock
    """
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Basic character cleaning (minimal)
    cleaned = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/]', '', cleaned)
    
    # Ensure proper sentence ending
    if cleaned and not cleaned.endswith(('.', '!', '?')):
        cleaned += '.'
    
    return cleaned

def add_basic_formatting(text: str) -> str:
    """
    Add basic formatting markers - lightweight
    Complex formatting delegated to UI/Bedrock
    """
    
    # Add emphasis for caps words (minimal)
    text = re.sub(r'\b[A-Z]{2,}\b', r'**\g<0>**', text)
    
    # Format phone numbers (basic)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', r'**\g<0>**', text)
    
    # Format ticket IDs (basic)
    text = re.sub(r'\bSVL-\d{8}-\d{6}-\d{3}\b', r'**\g<0>**', text)
    
    return text

def extract_action_items(text: str) -> List[str]:
    """
    Extract action items from response - simple pattern matching
    Complex extraction delegated to Bedrock
    """
    
    action_items = []
    
    # Look for numbered lists (basic)
    numbered_items = re.findall(r'\d+\.\s*([^.]+)\.?', text)
    action_items.extend(numbered_items)
    
    # Look for "please" items (basic)
    please_items = re.findall(r'[Pp]lease\s+([^.]+)\.?', text)
    action_items.extend([f"Please {item}" for item in please_items])
    
    # Look for "you should" items (basic)
    should_items = re.findall(r'[Yy]ou should\s+([^.]+)\.?', text)
    action_items.extend([f"You should {item}" for item in should_items])
    
    return action_items[:5]  # Limit to 5 items

def extract_next_steps(text: str) -> List[str]:
    """
    Extract next steps from text - basic extraction
    """
    
    next_steps = []
    
    # Look for "next" related items
    next_patterns = [
        r'[Nn]ext[,:]?\s*([^.]+)\.?',
        r'[Ff]ollow(?:ing)?\s+steps?[,:]?\s*([^.]+)\.?',
        r'[Ww]hat happens next[,:]?\s*([^.]+)\.?'
    ]
    
    for pattern in next_patterns:
        matches = re.findall(pattern, text)
        next_steps.extend(matches)
    
    return next_steps[:3]  # Limit to 3 steps

def create_success_response(formatted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create successful formatting response"""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "success": True,
            "formatted_response": formatted_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    }

def create_error_response(status_code: int, error_message: str) -> Dict[str, Any]:
    """Create error response"""
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

# Future enhancement placeholder
def apply_advanced_formatting(text: str, context: Dict[str, Any]) -> str:
    """
    Placeholder for advanced formatting
    Will be implemented with Bedrock Agents for complex formatting
    """
    # TODO: Implement with Bedrock Agents
    # - Rich text formatting
    # - Context-aware formatting
    # - Multi-language support
    # - Accessibility features
    
    return text 