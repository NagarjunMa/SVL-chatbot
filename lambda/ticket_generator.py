"""
AWS Lambda: Ticket Generator
Simple ticket generation system for SVL chatbot
Lightweight ID generation and storage for Bedrock Agents compatibility
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
DYNAMODB_TABLE_PREFIX = os.environ.get("DYNAMODB_TABLE_PREFIX", "svl")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# AWS clients
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

def lambda_handler(event, context):
    """
    Lambda handler for ticket generation
    Simple ticket creation with minimal processing
    """
    try:
        logger.info(f"Generating ticket: {json.dumps(event)}")
        
        # Parse request
        request_data = parse_request(event)
        
        # Validate ticket data (minimal)
        validation_result = validate_ticket_data(request_data)
        if not validation_result["valid"]:
            return create_error_response(400, validation_result["error"])
        
        # Generate ticket ID
        ticket_id = generate_ticket_id()
        
        # Create ticket record (lightweight)
        ticket_data = create_ticket_record(ticket_id, request_data)
        
        # Store ticket
        storage_result = store_ticket(ticket_data)
        if not storage_result["success"]:
            return create_error_response(500, storage_result["error"])
        
        logger.info(f"Ticket generated successfully: {ticket_id}")
        
        return create_success_response(ticket_data)
        
    except Exception as e:
        logger.error(f"Error in ticket generator: {str(e)}")
        return create_error_response(500, "Ticket generation failed")

def parse_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Lambda event for ticket generation
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
        "session_id": body.get("session_id", ""),
        "user_id": body.get("user_id", ""),
        "vehicle_info": body.get("vehicle_info", {}),
        "incident_info": body.get("incident_info", {}),
        "contact_info": body.get("contact_info", {}),
        "additional_notes": body.get("additional_notes", ""),
        "priority": body.get("priority", "medium"),
        "source": body.get("source", "chatbot")
    }

def validate_ticket_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basic ticket data validation - minimal checks
    Complex validation delegated to Bedrock/business logic
    """
    
    # Check basic required fields
    if not data.get("session_id"):
        return {"valid": False, "error": "Session ID is required"}
    
    # Basic vehicle info check (minimal)
    vehicle_info = data.get("vehicle_info", {})
    if not any([
        vehicle_info.get("make"),
        vehicle_info.get("model"), 
        vehicle_info.get("license_plate"),
        vehicle_info.get("vin")
    ]):
        return {"valid": False, "error": "At least one vehicle identifier is required"}
    
    return {"valid": True, "error": None}

def generate_ticket_id() -> str:
    """
    Generate unique ticket ID in format: SVL-YYYYMMDD-HHMMSS-XXX
    Simple deterministic format for easy tracking
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    
    # Generate 3-digit sequence from UUID (last 3 digits)
    sequence = str(uuid.uuid4().int % 1000).zfill(3)
    
    return f"SVL-{date_str}-{time_str}-{sequence}"

def create_ticket_record(ticket_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create ticket record with minimal processing
    Lightweight structure for easy Bedrock Agent integration
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    return {
        "ticket_id": ticket_id,
        "session_id": request_data.get("session_id"),
        "user_id": request_data.get("user_id"),
        "status": "open",
        "priority": request_data.get("priority", "medium"),
        "source": request_data.get("source", "chatbot"),
        "created_at": timestamp,
        "updated_at": timestamp,
        
        # Vehicle information (basic structure)
        "vehicle_info": {
            "make": request_data.get("vehicle_info", {}).get("make", ""),
            "model": request_data.get("vehicle_info", {}).get("model", ""),
            "year": request_data.get("vehicle_info", {}).get("year", ""),
            "color": request_data.get("vehicle_info", {}).get("color", ""),
            "license_plate": request_data.get("vehicle_info", {}).get("license_plate", ""),
            "vin": request_data.get("vehicle_info", {}).get("vin", "")
        },
        
        # Incident information (basic structure)
        "incident_info": {
            "date_reported": timestamp,
            "location": request_data.get("incident_info", {}).get("location", ""),
            "circumstances": request_data.get("incident_info", {}).get("circumstances", ""),
            "date_occurred": request_data.get("incident_info", {}).get("date_occurred", "")
        },
        
        # Contact information (minimal - avoid PII in Lambda)
        "contact_info": {
            "contact_method": request_data.get("contact_info", {}).get("contact_method", "session"),
            "session_reference": request_data.get("session_id")
        },
        
        # Additional data
        "notes": [request_data.get("additional_notes", "")] if request_data.get("additional_notes") else [],
        "metadata": {
            "lambda_version": context.aws_request_id if 'context' in globals() else "unknown",
            "processing_time": timestamp
        }
    }

def store_ticket(ticket_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store ticket in DynamoDB
    Simple storage with minimal error handling
    """
    try:
        tickets_table = dynamodb.Table(f"{DYNAMODB_TABLE_PREFIX}-tickets")
        
        # Store ticket
        tickets_table.put_item(Item=ticket_data)
        
        logger.info(f"Ticket stored successfully: {ticket_data['ticket_id']}")
        
        return {"success": True, "error": None}
        
    except Exception as e:
        logger.error(f"Failed to store ticket: {str(e)}")
        return {"success": False, "error": f"Storage failed: {str(e)}"}

def create_success_response(ticket_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create successful ticket generation response"""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "success": True,
            "ticket_id": ticket_data["ticket_id"],
            "status": ticket_data["status"],
            "created_at": ticket_data["created_at"],
            "summary": {
                "vehicle": ticket_data["vehicle_info"],
                "incident": ticket_data["incident_info"]
            },
            "next_steps": [
                "Your stolen vehicle report has been recorded",
                f"Reference number: {ticket_data['ticket_id']}",
                "Local authorities will be notified",
                "You will receive updates on this case"
            ]
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