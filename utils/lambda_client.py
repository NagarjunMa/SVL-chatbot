"""
Lambda Client for SVL Chatbot
Integrates existing Streamlit app with new Lambda architecture
Provides seamless transition between local and serverless processing
"""

import json
import boto3
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import os
import logging

from utils.logger import get_logger

logger = get_logger("lambda_client")

class SVLLambdaClient:
    """
    Client for interacting with SVL Lambda functions
    Provides both direct Lambda invocation and HTTP API Gateway calls
    """
    
    def __init__(self, 
                 use_lambda: bool = False,
                 api_gateway_url: Optional[str] = None,
                 stage: str = "dev",
                 region: str = "us-east-1"):
        """
        Initialize Lambda client
        
        Args:
            use_lambda: If True, use direct Lambda invocation. If False, use API Gateway
            api_gateway_url: API Gateway URL for HTTP requests
            stage: Deployment stage (dev, prod, etc.)
            region: AWS region
        """
        self.use_lambda = use_lambda
        self.api_gateway_url = api_gateway_url
        self.stage = stage
        self.region = region
        
        # Initialize AWS clients if using direct Lambda invocation
        if self.use_lambda:
            self.lambda_client = boto3.client("lambda", region_name=region)
            self.function_prefix = f"svl-chatbot-lambda-{stage}"
    
    async def process_chat_message(self, user_input: str, session_id: str, phase: str = "greeting") -> Dict[str, Any]:
        """
        Process chat message through Lambda architecture
        
        Args:
            user_input: User's message
            session_id: Session identifier
            phase: Conversation phase
            
        Returns:
            Processed response from Lambda
        """
        try:
            request_data = {
                "user_input": user_input,
                "session_id": session_id,
                "phase": phase,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if self.use_lambda:
                return await self._invoke_lambda_direct("bedrockOrchestrator", request_data)
            else:
                return await self._call_api_gateway("/chat", request_data)
                
        except Exception as e:
            logger.error(f"Chat processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"Chat processing failed: {str(e)}",
                "fallback_response": "I'm having trouble processing your request right now. Please try again."
            }
    
    async def create_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create ticket through Lambda architecture
        
        Args:
            ticket_data: Ticket information including vehicle details
            
        Returns:
            Ticket creation result
        """
        try:
            if self.use_lambda:
                return await self._invoke_lambda_direct("ticketGenerator", ticket_data)
            else:
                return await self._call_api_gateway("/ticket", ticket_data)
                
        except Exception as e:
            logger.error(f"Ticket creation failed: {str(e)}")
            return {
                "success": False,
                "error": f"Ticket creation failed: {str(e)}"
            }
    
    async def query_knowledge_base(self, query: str, query_type: str = "general", session_id: str = "") -> Dict[str, Any]:
        """
        Query knowledge base through Lambda
        
        Args:
            query: Search query
            query_type: Type of query (faq, process, general)
            session_id: Session identifier
            
        Returns:
            Knowledge base results
        """
        try:
            request_data = {
                "query": query,
                "type": query_type,
                "session_id": session_id
            }
            
            return await self._invoke_lambda_direct("knowledgeBaseQuery", request_data)
            
        except Exception as e:
            logger.error(f"Knowledge base query failed: {str(e)}")
            return {
                "success": False,
                "error": f"Knowledge query failed: {str(e)}",
                "context": {"results": [], "total_found": 0}
            }
    
    async def preprocess_request(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """
        Preprocess user request through Lambda
        
        Args:
            user_input: User's input
            session_id: Session identifier
            
        Returns:
            Preprocessed request data
        """
        try:
            request_data = {
                "user_input": user_input,
                "session_id": session_id
            }
            
            return await self._invoke_lambda_direct("requestPreprocessor", request_data)
            
        except Exception as e:
            logger.error(f"Request preprocessing failed: {str(e)}")
            return {
                "success": True,  # Don't block on preprocessing failure
                "data": request_data,
                "warning": f"Preprocessing failed: {str(e)}"
            }
    
    async def format_response(self, response: str, response_type: str = "chat", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format response through Lambda
        
        Args:
            response: Raw response text
            response_type: Type of response (chat, ticket, error)
            context: Additional context
            
        Returns:
            Formatted response
        """
        try:
            request_data = {
                "response": response,
                "type": response_type,
                "context": context or {}
            }
            
            return await self._invoke_lambda_direct("responseFormatter", request_data)
            
        except Exception as e:
            logger.error(f"Response formatting failed: {str(e)}")
            return {
                "success": True,  # Don't block on formatting failure
                "formatted_response": {
                    "type": response_type,
                    "content": response
                },
                "warning": f"Formatting failed: {str(e)}"
            }
    
    async def _invoke_lambda_direct(self, function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke Lambda function directly
        
        Args:
            function_name: Name of the Lambda function
            payload: Request payload
            
        Returns:
            Lambda response
        """
        try:
            full_function_name = f"{self.function_prefix}-{function_name}"
            
            response = self.lambda_client.invoke(
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
    
    async def _call_api_gateway(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call API Gateway endpoint
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            
        Returns:
            API response
        """
        try:
            if not self.api_gateway_url:
                raise ValueError("API Gateway URL not configured")
            
            url = f"{self.api_gateway_url.rstrip('/')}{endpoint}"
            
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API Gateway call failed: {str(e)}")
            return {
                "success": False,
                "error": f"API call failed: {str(e)}"
            }

# Enhanced ConversationManager that can use Lambda backend
class LambdaEnhancedConversationManager:
    """
    Enhanced ConversationManager that can seamlessly switch between
    local processing and Lambda-based processing
    """
    
    def __init__(self, session_id: str, use_lambda: bool = False, **lambda_kwargs):
        """
        Initialize enhanced conversation manager
        
        Args:
            session_id: Session identifier
            use_lambda: Whether to use Lambda backend
            **lambda_kwargs: Additional arguments for Lambda client
        """
        self.session_id = session_id
        self.use_lambda = use_lambda
        
        if self.use_lambda:
            self.lambda_client = SVLLambdaClient(**lambda_kwargs)
        else:
            # Fall back to existing local conversation manager
            from utils.conversation_manager import ConversationManager
            self.local_manager = ConversationManager(session_id)
    
    async def process_user_input(self, user_input: str, phase: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """
        Process user input through either Lambda or local processing
        
        Args:
            user_input: User's message
            phase: Conversation phase
            extra: Additional context
            
        Returns:
            AI response
        """
        try:
            if self.use_lambda:
                # Use Lambda backend
                result = await self.lambda_client.process_chat_message(
                    user_input=user_input,
                    session_id=self.session_id,
                    phase=phase
                )
                
                if result.get("success"):
                    formatted_response = result.get("response", {})
                    if isinstance(formatted_response, dict):
                        return formatted_response.get("content", result.get("fallback_response", ""))
                    else:
                        return str(formatted_response)
                else:
                    # Fall back to local processing on Lambda failure
                    logger.warning("Lambda processing failed, falling back to local")
                    return self.local_manager.process_user_input(user_input, phase, extra)
            else:
                # Use local processing
                return self.local_manager.process_user_input(user_input, phase, extra)
                
        except Exception as e:
            logger.error(f"Enhanced conversation processing failed: {str(e)}")
            if hasattr(self, 'local_manager'):
                return self.local_manager.process_user_input(user_input, phase, extra)
            else:
                return "I'm having trouble processing your request right now. Please try again."

# Configuration helper
def get_lambda_config() -> Dict[str, Any]:
    """
    Get Lambda configuration from environment variables
    
    Returns:
        Lambda configuration dictionary
    """
    return {
        "use_lambda": os.environ.get("USE_LAMBDA_BACKEND", "false").lower() == "true",
        "api_gateway_url": os.environ.get("API_GATEWAY_URL"),
        "stage": os.environ.get("LAMBDA_STAGE", "dev"),
        "region": os.environ.get("AWS_REGION", "us-east-1")
    }

# Factory function for easy integration
def get_enhanced_conversation_manager(session_id: str) -> LambdaEnhancedConversationManager:
    """
    Factory function to create enhanced conversation manager
    
    Args:
        session_id: Session identifier
        
    Returns:
        Enhanced conversation manager instance
    """
    lambda_config = get_lambda_config()
    return LambdaEnhancedConversationManager(session_id, **lambda_config) 