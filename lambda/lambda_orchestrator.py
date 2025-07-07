"""
Lambda Orchestrator: Coordinates Lambda functions for complex workflows
Lightweight orchestration for SVL chatbot - prepares for Bedrock Agents migration
"""

import json
import boto3
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
lambda_client = boto3.client("lambda")

class LambdaOrchestrator:
    """
    Orchestrates Lambda functions for complex SVL chatbot workflows
    Designed for easy migration to Bedrock Agents
    """
    
    def __init__(self, stage: str = "dev"):
        self.stage = stage
        self.function_prefix = f"svl-chatbot-lambda-{stage}"
        
    async def process_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate complete chat processing workflow
        Lightweight coordination of Lambda functions
        """
        try:
            logger.info(f"Orchestrating chat request: {request_data.get('user_input', '')[:50]}...")
            
            # Step 1: Preprocess request (parallel with knowledge query)
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Run preprocessing and knowledge query in parallel
                preprocess_future = executor.submit(
                    self._invoke_lambda,
                    "requestPreprocessor",
                    request_data
                )
                
                knowledge_future = executor.submit(
                    self._invoke_lambda,
                    "knowledgeBaseQuery",
                    {
                        "query": request_data.get("user_input", ""),
                        "type": "general",
                        "session_id": request_data.get("session_id", "")
                    }
                )
                
                # Wait for both to complete
                preprocess_result = preprocess_future.result()
                knowledge_result = knowledge_future.result()
            
            if not preprocess_result.get("success"):
                return preprocess_result
            
            # Step 2: Enhance request with knowledge context
            enhanced_request = self._merge_context(
                preprocess_result["data"],
                knowledge_result.get("context", {})
            )
            
            # Step 3: Process with Bedrock
            bedrock_result = self._invoke_lambda(
                "bedrockOrchestrator",
                enhanced_request
            )
            
            if not bedrock_result.get("success"):
                return bedrock_result
            
            # Step 4: Format response
            format_result = self._invoke_lambda(
                "responseFormatter",
                {
                    "response": bedrock_result["response"],
                    "type": "chat",
                    "session_id": enhanced_request.get("session_id", ""),
                    "context": enhanced_request
                }
            )
            
            logger.info("Chat request orchestration completed successfully")
            
            return {
                "success": True,
                "response": format_result.get("formatted_response", {}),
                "session_id": enhanced_request.get("session_id", ""),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Chat orchestration failed: {str(e)}")
            return {
                "success": False,
                "error": f"Orchestration failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def process_ticket_creation(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate ticket creation workflow
        Coordinates validation, creation, and notification
        """
        try:
            logger.info(f"Orchestrating ticket creation for session: {ticket_data.get('session_id', '')}")
            
            # Step 1: Validate ticket data
            validation_result = self._invoke_lambda(
                "requestPreprocessor",
                ticket_data
            )
            
            if not validation_result.get("success"):
                return validation_result
            
            # Step 2: Generate ticket
            ticket_result = self._invoke_lambda(
                "ticketGenerator",
                validation_result["data"]
            )
            
            if not ticket_result.get("success"):
                return ticket_result
            
            # Step 3: Format ticket response
            format_result = self._invoke_lambda(
                "responseFormatter",
                {
                    "response": f"Ticket {ticket_result['ticket_id']} created successfully",
                    "type": "ticket",
                    "session_id": ticket_data.get("session_id", ""),
                    "context": ticket_result
                }
            )
            
            logger.info(f"Ticket creation orchestration completed: {ticket_result['ticket_id']}")
            
            return {
                "success": True,
                "ticket_id": ticket_result["ticket_id"],
                "formatted_response": format_result.get("formatted_response", {}),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ticket orchestration failed: {str(e)}")
            return {
                "success": False,
                "error": f"Ticket creation failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _invoke_lambda(self, function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke Lambda function with error handling
        Lightweight wrapper for function calls
        """
        try:
            full_function_name = f"{self.function_prefix}-{function_name}"
            
            response = lambda_client.invoke(
                FunctionName=full_function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload)
            )
            
            # Parse response
            response_payload = json.loads(response["Payload"].read())
            
            # Handle Lambda errors
            if response.get("FunctionError"):
                logger.error(f"Lambda function {function_name} failed: {response_payload}")
                return {
                    "success": False,
                    "error": f"Function {function_name} failed",
                    "details": response_payload
                }
            
            # Parse body if it's an API Gateway response
            if "body" in response_payload:
                if isinstance(response_payload["body"], str):
                    return json.loads(response_payload["body"])
                else:
                    return response_payload["body"]
            
            return response_payload
            
        except Exception as e:
            logger.error(f"Failed to invoke {function_name}: {str(e)}")
            return {
                "success": False,
                "error": f"Function invocation failed: {str(e)}"
            }
    
    def _merge_context(self, request_data: Dict[str, Any], knowledge_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge request data with knowledge context
        Lightweight context enhancement
        """
        enhanced_request = request_data.copy()
        
        # Add knowledge context
        if "context" not in enhanced_request:
            enhanced_request["context"] = {}
        
        enhanced_request["context"]["knowledge"] = knowledge_context
        enhanced_request["context"]["enhanced_at"] = datetime.now(timezone.utc).isoformat()
        
        return enhanced_request

# Standalone Lambda handler for orchestration
def lambda_handler(event, context):
    """
    Lambda handler for orchestration requests
    Routes to appropriate workflow
    """
    try:
        # Parse request
        if "body" in event:
            if isinstance(event["body"], str):
                body = json.loads(event["body"])
            else:
                body = event["body"]
        else:
            body = event
        
        workflow_type = body.get("workflow", "chat")
        orchestrator = LambdaOrchestrator(stage=body.get("stage", "dev"))
        
        # Route to appropriate workflow
        if workflow_type == "chat":
            result = asyncio.run(orchestrator.process_chat_request(body))
        elif workflow_type == "ticket":
            result = asyncio.run(orchestrator.process_ticket_creation(body))
        else:
            result = {
                "success": False,
                "error": f"Unknown workflow type: {workflow_type}"
            }
        
        return {
            "statusCode": 200 if result.get("success") else 400,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Orchestration handler failed: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "success": False,
                "error": "Orchestration failed",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        }

# Utility functions for Bedrock Agents migration
class BedrockAgentPreparation:
    """
    Preparation utilities for migrating to Bedrock Agents
    These functions will help identify migration opportunities
    """
    
    @staticmethod
    def analyze_lambda_complexity(function_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Lambda function complexity for Bedrock Agent migration
        Identifies functions that can be replaced by agents
        """
        complexity_score = 0
        migration_readiness = "high"
        
        # Factors that increase complexity (harder to migrate)
        if function_metrics.get("business_logic_lines", 0) > 50:
            complexity_score += 2
            migration_readiness = "medium"
        
        if function_metrics.get("external_dependencies", 0) > 3:
            complexity_score += 1
        
        if function_metrics.get("stateful_operations", False):
            complexity_score += 2
            migration_readiness = "low"
        
        # Factors that favor migration
        if function_metrics.get("ai_focused", False):
            complexity_score -= 1
        
        if function_metrics.get("simple_io", True):
            complexity_score -= 1
        
        return {
            "complexity_score": complexity_score,
            "migration_readiness": migration_readiness,
            "recommendations": BedrockAgentPreparation._get_migration_recommendations(complexity_score)
        }
    
    @staticmethod
    def _get_migration_recommendations(complexity_score: int) -> List[str]:
        """Get migration recommendations based on complexity"""
        if complexity_score <= 0:
            return [
                "Excellent candidate for Bedrock Agent replacement",
                "Can be migrated with minimal changes",
                "Consider migrating in first phase"
            ]
        elif complexity_score <= 2:
            return [
                "Good candidate for Bedrock Agent with some refactoring",
                "Extract business logic to separate components",
                "Plan for second migration phase"
            ]
        else:
            return [
                "Keep as Lambda function for now",
                "Focus on simplifying business logic",
                "Consider hybrid approach with Bedrock Agents"
            ] 