"""
AWS Lambda: Knowledge Base Query Handler
Context retrieval for SVL chatbot - minimal processing
Designed for easy integration with Bedrock Agents and Knowledge Bases
"""

import json
import boto3
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
DYNAMODB_TABLE_PREFIX = os.environ.get("DYNAMODB_TABLE_PREFIX", "svl")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")

# AWS clients
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
# bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)  # For future Bedrock Agents

# Import semantic knowledge base with graceful fallback
SEMANTIC_SEARCH_AVAILABLE = False
try:
    # These would be packaged in a Lambda layer
    from utils.knowledge_base import KnowledgeBase
    from utils.semantic_search_engine import SemanticSearchEngine
    SEMANTIC_SEARCH_AVAILABLE = True
    logger.info("Semantic search modules imported successfully")
except ImportError as e:
    logger.warning(f"Semantic search not available: {e}")
    SEMANTIC_SEARCH_AVAILABLE = False

def lambda_handler(event, context):
    """
    Lambda handler for knowledge base queries
    Lightweight context retrieval - delegate complex search to Bedrock
    """
    try:
        logger.info(f"Processing knowledge query: {json.dumps(event)}")
        
        # Parse request
        request_data = parse_request(event)
        query = request_data.get("query", "")
        query_type = request_data.get("type", "general")
        
        if not query:
            return create_error_response(400, "Query is required")
        
        # Route query based on availability of semantic search
        if SEMANTIC_SEARCH_AVAILABLE:
            context = get_semantic_context(query, query_type, request_data)
        else:
            # Fallback to basic routing
            if query_type == "faq":
                context = get_faq_context(query)
            elif query_type == "process":
                context = get_process_context(query)
            elif query_type == "conversation":
                context = get_conversation_context(request_data.get("session_id", ""))
            else:
                context = get_general_context(query)
        
        logger.info(f"Retrieved context items: {len(context.get('results', []))}")
        
        return create_success_response(context)
        
    except Exception as e:
        logger.error(f"Error in knowledge base query: {str(e)}")
        return create_error_response(500, "Knowledge query failed")

def parse_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Lambda event for knowledge query"""
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
        "query": body.get("query", "").strip(),
        "type": body.get("type", "general"),
        "session_id": body.get("session_id", ""),
        "max_results": body.get("max_results", 5),
        "context": body.get("context", {}),
        "similarity_threshold": body.get("similarity_threshold", 0.7)
    }

def get_semantic_context(query: str, query_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get context using semantic search with embeddings
    Enhanced knowledge base with PDF processing and FAISS search
    """
    try:
        logger.info(f"Using semantic search for query: {query}")
        
        # Get S3 bucket from environment
        s3_bucket = os.environ.get('KNOWLEDGE_BASE_S3_BUCKET', 'svl-knowledge-base')
        
        # Initialize knowledge base
        knowledge_base = KnowledgeBase(s3_bucket_name=s3_bucket)
        
        # Check if knowledge base is ready
        if not knowledge_base.is_ready():
            logger.info("Initializing knowledge base...")
            init_result = knowledge_base.initialize_knowledge_base()
            
            if init_result["status"] not in ["success", "loaded_existing"]:
                logger.warning(f"Knowledge base initialization issues: {init_result}")
                # Fall back to basic context
                return get_general_context(query)
        
        # Query the semantic knowledge base
        result = knowledge_base.query_knowledge_base(
            user_query=query,
            query_type=query_type,
            include_context=True,
            max_results=request_data.get("max_results", 5),
            similarity_threshold=request_data.get("similarity_threshold", 0.7)
        )
        
        if result["status"] == "success" and result.get("total_results", 0) > 0:
            # Format semantic search results for Lambda response
            formatted_results = []
            for search_result in result["results"]:
                formatted_results.append({
                    "content": search_result["text"],
                    "source": search_result["document"]["filename"],
                    "category": search_result["document"]["category"],
                    "relevance_score": search_result["relevance_score"],
                    "similarity_score": search_result["similarity_score"],
                    "rank": search_result["rank"],
                    "semantic_search": True
                })
            
            return {
                "type": "semantic",
                "query": query,
                "results": formatted_results,
                "total_found": result["total_results"],
                "context": result.get("context", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "search_time": result.get("search_time", 0),
                "semantic_search": True
            }
        else:
            # No semantic results found, return basic context
            logger.info("No semantic results found, falling back to basic context")
            return get_general_context(query)
            
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        # Fall back to basic context
        return get_general_context(query)

def get_faq_context(query: str) -> Dict[str, Any]:
    """
    Get FAQ context - minimal hardcoded responses
    In production, this would query a proper knowledge base
    """
    
    # Basic FAQ mapping (lightweight)
    faq_responses = {
        "how": [
            {
                "question": "How does the stolen vehicle recovery process work?",
                "answer": "When you report a stolen vehicle through SVL, we immediately alert local authorities and add your vehicle to national databases. Our network of recovery specialists begins searching within your area.",
                "relevance": 0.9
            }
        ],
        "what": [
            {
                "question": "What information do I need to report a stolen vehicle?",
                "answer": "You'll need your vehicle's make, model, year, color, license plate number, and VIN. Also provide the location and time when you discovered it was stolen.",
                "relevance": 0.9
            }
        ],
        "when": [
            {
                "question": "When should I report my vehicle stolen?",
                "answer": "Report your vehicle stolen immediately after confirming it's missing. The sooner you report, the better chances of recovery.",
                "relevance": 0.9
            }
        ],
        "where": [
            {
                "question": "Where can I track my case status?",
                "answer": "You can check your case status using your ticket ID. Updates are provided through the system and local law enforcement.",
                "relevance": 0.8
            }
        ]
    }
    
    # Simple keyword matching (minimal processing)
    query_lower = query.lower()
    matching_context = []
    
    for keyword, responses in faq_responses.items():
        if keyword in query_lower:
            matching_context.extend(responses)
    
    return {
        "type": "faq",
        "query": query,
        "results": matching_context[:3],  # Limit results
        "total_found": len(matching_context)
    }

def get_process_context(query: str) -> Dict[str, Any]:
    """
    Get process information - minimal process guidance
    Lightweight responses for common process questions
    """
    
    process_info = [
        {
            "process": "Vehicle Reporting",
            "steps": [
                "1. Provide vehicle details (make, model, VIN, license plate)",
                "2. Describe incident circumstances and location",
                "3. Receive ticket ID for tracking",
                "4. Authorities are automatically notified",
                "5. Recovery process begins immediately"
            ],
            "relevance": 0.9
        },
        {
            "process": "Case Tracking", 
            "steps": [
                "1. Use your SVL ticket ID for updates",
                "2. Check with local law enforcement",
                "3. Monitor for system notifications",
                "4. Recovery team provides regular updates"
            ],
            "relevance": 0.8
        }
    ]
    
    return {
        "type": "process",
        "query": query,
        "results": process_info,
        "total_found": len(process_info)
    }

def get_conversation_context(session_id: str) -> Dict[str, Any]:
    """
    Get conversation context from DynamoDB
    Lightweight retrieval of recent conversation history
    """
    if not session_id:
        return {"type": "conversation", "results": [], "total_found": 0}
    
    try:
        conversations_table = dynamodb.Table(f"{DYNAMODB_TABLE_PREFIX}-conversations")
        
        response = conversations_table.query(
            KeyConditionExpression="conversation_id = :session_id",
            ExpressionAttributeValues={":session_id": session_id},
            Limit=5,  # Recent context only
            ScanIndexForward=False  # Most recent first
        )
        
        items = response.get("Items", [])
        
        # Extract conversation summary (minimal processing)
        conversation_summary = []
        for item in items:
            if "messages" in item:
                for msg in item["messages"]:
                    conversation_summary.append({
                        "role": msg.get("role", ""),
                        "content": msg.get("content", "")[:200],  # Truncate
                        "timestamp": msg.get("timestamp", "")
                    })
        
        return {
            "type": "conversation",
            "session_id": session_id,
            "results": conversation_summary[-10:],  # Last 10 messages
            "total_found": len(conversation_summary)
        }
        
    except Exception as e:
        logger.warning(f"Could not retrieve conversation context: {e}")
        return {"type": "conversation", "results": [], "total_found": 0}

def get_general_context(query: str) -> Dict[str, Any]:
    """
    Get general context - basic information
    Minimal context for general queries
    """
    
    general_info = [
        {
            "topic": "SVL Service Overview",
            "content": "SVL (Stolen Vehicle Locator) provides 24/7 stolen vehicle recovery services. We work with law enforcement and use advanced tracking technology to locate and recover stolen vehicles.",
            "relevance": 0.7
        },
        {
            "topic": "Emergency Contacts",
            "content": "For immediate assistance: Police Emergency (911), SVL Hotline (1-800-555-0123), Insurance Support (1-800-555-0199)",
            "relevance": 0.8
        },
        {
            "topic": "Recovery Success",
            "content": "SVL has a high success rate in vehicle recovery when reported within the first 24 hours. Early reporting significantly improves recovery chances.",
            "relevance": 0.6
        }
    ]
    
    return {
        "type": "general",
        "query": query,
        "results": general_info,
        "total_found": len(general_info)
    }

def create_success_response(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create successful knowledge query response"""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "success": True,
            "context": context_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "Context provided for Bedrock processing"
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

# Future Bedrock Agents integration placeholder
def query_bedrock_knowledge_base(query: str, knowledge_base_id: str) -> Dict[str, Any]:
    """
    Placeholder for Bedrock Knowledge Base integration
    This function will be implemented when migrating to Bedrock Agents
    """
    # TODO: Implement Bedrock Knowledge Base query
    # bedrock_response = bedrock_agent.retrieve(
    #     knowledgeBaseId=knowledge_base_id,
    #     retrievalQuery={"text": query}
    # )
    # return process_bedrock_response(bedrock_response)
    
    return {"results": [], "message": "Bedrock Knowledge Base integration pending"} 