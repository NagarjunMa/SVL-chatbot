"""
AWS Lambda: Request Preprocessor
Minimal input validation and sanitization for SVL chatbot
Designed for easy replacement by Bedrock Agents
"""

import json
import re
import boto3
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# PII patterns for basic detection
PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
}

def lambda_handler(event, context):
    """
    Lambda handler for request preprocessing
    Minimal validation - let Bedrock handle complex logic
    """
    try:
        logger.info(f"Preprocessing request: {json.dumps(event)}")
        
        # Parse input
        request_data = parse_input(event)
        
        # Basic validation (minimal)
        validation_result = validate_request(request_data)
        if not validation_result["valid"]:
            return create_error_response(400, validation_result["error"])
        
        # Sanitize input (basic)
        sanitized_data = sanitize_input(request_data)
        
        # PII detection (warning only)
        pii_warnings = detect_pii(sanitized_data.get("user_input", ""))
        if pii_warnings:
            sanitized_data["pii_warnings"] = pii_warnings
            logger.warning(f"PII detected in input: {pii_warnings}")
        
        logger.info("Request preprocessing completed")
        
        return create_success_response(sanitized_data)
        
    except Exception as e:
        logger.error(f"Error in request preprocessor: {str(e)}")
        return create_error_response(500, "Preprocessing failed")

def parse_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Lambda event to extract request data
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
        "user_input": body.get("user_input", ""),
        "session_id": body.get("session_id", ""),
        "phase": body.get("phase", "greeting"),
        "user_id": body.get("user_id", ""),
        "context": body.get("context", {}),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def validate_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basic request validation - minimal checks only
    Complex validation delegated to Bedrock
    """
    
    # Check required fields
    user_input = data.get("user_input", "").strip()
    if not user_input:
        return {"valid": False, "error": "User input is required"}
    
    # Basic length check
    if len(user_input) > 10000:
        return {"valid": False, "error": "Input too long (max 10000 characters)"}
    
    # Basic content check (minimal)
    if len(user_input) < 1:
        return {"valid": False, "error": "Input too short"}
    
    # Session ID format (basic)
    session_id = data.get("session_id", "")
    if session_id and not re.match(r"^[a-zA-Z0-9-_]+$", session_id):
        return {"valid": False, "error": "Invalid session ID format"}
    
    return {"valid": True, "error": None}

def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basic input sanitization - minimal processing
    Complex sanitization delegated to Bedrock
    """
    
    # Clean user input (basic)
    user_input = data.get("user_input", "")
    
    # Remove excessive whitespace
    user_input = re.sub(r'\s+', ' ', user_input).strip()
    
    # Remove potentially harmful characters (minimal)
    user_input = re.sub(r'[<>]', '', user_input)
    
    # Basic SQL injection prevention (minimal)
    dangerous_patterns = [
        r'(?i)(drop\s+table)',
        r'(?i)(delete\s+from)',
        r'(?i)(insert\s+into)',
        r'(?i)(update\s+.*set)',
        r'(?i)(union\s+select)'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, user_input):
            logger.warning(f"Potentially dangerous input detected: {pattern}")
            user_input = re.sub(pattern, '[FILTERED]', user_input)
    
    return {
        **data,
        "user_input": user_input,
        "sanitized": True,
        "processed_at": datetime.now(timezone.utc).isoformat()
    }

def detect_pii(text: str) -> List[str]:
    """
    Basic PII detection - warning only
    Complex PII handling delegated to Bedrock/business logic
    """
    detected_pii = []
    
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            detected_pii.append(pii_type)
    
    return detected_pii

def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create successful preprocessing response"""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "success": True,
            "data": data,
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