"""
Enhanced Conversation Manager with Comprehensive Observability
Tracks every aspect of conversation flow from user input to final response
"""

import json
import time
import boto3
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import re

from utils.logger import get_logger
from utils.database_manager import DatabaseManager
from config.aws_config import get_bedrock_client
from utils.observability import observability, trace_function

# Knowledge base import - using new Bedrock Knowledge Base
try:
    from utils.bedrock_knowledge_base import BedrockKnowledgeBaseManager
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Bedrock Knowledge base not available: {e}")
    KNOWLEDGE_BASE_AVAILABLE = False

logger = get_logger("conversation_manager")

class ConversationManagerWithObservability:
    """Enhanced conversation manager with comprehensive observability and tracing"""
    
    @trace_function("conversation_manager_init")
    def __init__(self, session_id: str):
        """Initialize conversation manager with observability"""
        self.session_id = session_id
        self.conversation_history = []
        self.context_memory = {}
        self.user_preferences = {}
        
        # Initialize Bedrock client
        with observability.trace_operation("init_bedrock_client") as span:
            self.bedrock_client = get_bedrock_client()
            span['metadata']['bedrock_client_initialized'] = True
        
        # Initialize database
        with observability.trace_operation("init_database") as span:
            self.db = DatabaseManager()
            span['metadata']['session_id'] = session_id
        
        # Initialize knowledge base if available
        self.knowledge_base = None
        if KNOWLEDGE_BASE_AVAILABLE:
            try:
                with observability.trace_operation("init_knowledge_base") as span:
                    self.knowledge_base = BedrockKnowledgeBaseManager()
                    span['metadata']['kb_id'] = self.knowledge_base.knowledge_base_id
                    span['metadata']['guardrail_id'] = self.knowledge_base.guardrail_id
                    
                logger.info("Bedrock Knowledge Base manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock Knowledge Base: {e}")
                observability.log_error(e, {
                    'operation': 'knowledge_base_initialization',
                    'session_id': session_id
                })
                self.knowledge_base = None
        
        # Load conversation history
        self._load_conversation_history()
    
    @trace_function("process_user_message")
    async def process_user_message(self, user_input: str) -> str:
        """Process user message with full observability"""
        
        start_time = time.time()
        
        with observability.trace_operation("user_message_processing") as span:
            try:
                span['metadata'] = {
                    'user_input_length': len(user_input),
                    'session_id': self.session_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Log API request
                observability.log_api_request({
                    'method': 'POST',
                    'endpoint': '/chat',
                    'user_id': 'anonymous',
                    'session_id': self.session_id,
                    'request_size': len(user_input),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                # Add to conversation history
                self.conversation_history.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                # Save user message to database
                await self.db.add_message_to_conversation(
                    self.session_id, 
                    'user', 
                    user_input
                )
                
                # Get knowledge base context
                kb_context = await self._get_knowledge_context(user_input)
                
                # Build prompt
                prompt = self._build_prompt(user_input, "active", kb_context)
                
                # Check for guardrails
                guardrails_result = self._check_content_guardrails(user_input)
                if guardrails_result.get('blocked', False):
                    # Log guardrails activation
                    observability.log_guardrails_activity({
                        'conversation_id': self.session_id,
                        'user_message': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                        'filter_triggered': True,
                        'filter_type': guardrails_result.get('reason', 'content_filter'),
                        'action_taken': 'blocked',
                        'blocked_content': True
                    })
                    
                    response = "I understand you may have various thoughts, but let's focus on gathering the necessary details to report and locate your stolen vehicle."
                else:
                    # Log guardrails passed
                    observability.log_guardrails_activity({
                        'conversation_id': self.session_id,
                        'user_message': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                        'filter_triggered': False,
                        'action_taken': 'approved',
                        'blocked_content': False
                    })
                    
                    # Generate response using Bedrock
                    response = await self._call_bedrock_model(prompt)
                
                # Add to conversation history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                # Save assistant response to database
                await self.db.add_message_to_conversation(
                    self.session_id, 
                    'assistant', 
                    response
                )
                
                # Calculate response time
                response_time_ms = int((time.time() - start_time) * 1000)
                
                # Log EVERY conversation activity (not just pricing) with PII masking
                observability.log_conversation({
                    'conversation_id': self.session_id,
                    'user_id': 'anonymous',
                    'session_id': self.session_id,
                    'user_message': user_input,  # Will be PII-masked in logging
                    'bot_response': response,     # Will be PII-masked in logging
                    'response_time_ms': response_time_ms,
                    'guardrails_triggered': guardrails_result.get('blocked', False),
                    'knowledge_base_used': bool(kb_context.get('results')),
                    'bedrock_model_used': 'amazon.nova-pro-v1:0',
                    'conversation_phase': 'active',
                    'metadata': {
                        'kb_results_count': len(kb_context.get('results', [])),
                        'prompt_length': len(prompt),
                        'user_input_length': len(user_input),
                        'response_length': len(response),
                        'guardrails_reason': guardrails_result.get('reason') if guardrails_result.get('blocked') else None,
                        'processing_steps': [
                            'input_validation',
                            'knowledge_retrieval', 
                            'guardrails_check',
                            'llm_generation',
                            'response_formatting'
                        ]
                    }
                })
                
                # Log API response for EVERY request
                observability.log_api_request({
                    'method': 'POST',
                    'endpoint': '/chat',
                    'user_id': 'anonymous',
                    'session_id': self.session_id,
                    'response_time_ms': response_time_ms,
                    'status_code': 200,
                    'request_size': len(user_input),
                    'response_size': len(response),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                # Log guardrails activity for EVERY request
                observability.log_guardrails_activity({
                    'conversation_id': self.session_id,
                    'user_message': user_input[:50] + "..." if len(user_input) > 50 else user_input,  # Truncated for logging
                    'filter_triggered': guardrails_result.get('blocked', False),
                    'filter_type': guardrails_result.get('reason', 'content_filter'),
                    'confidence_score': guardrails_result.get('confidence', 0.0),
                    'action_taken': 'blocked' if guardrails_result.get('blocked') else 'approved',
                    'blocked_content': guardrails_result.get('blocked', False),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                span['metadata']['response_time_ms'] = response_time_ms
                span['metadata']['response_length'] = len(response)
                
                return response
                
            except Exception as e:
                error_time_ms = int((time.time() - start_time) * 1000)
                
                # Log error
                observability.log_error(e, {
                    'session_id': self.session_id,
                    'user_input': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                    'error_time_ms': error_time_ms
                })
                
                # Log failed API request
                observability.log_api_request({
                    'method': 'POST',
                    'endpoint': '/chat',
                    'user_id': 'anonymous',
                    'session_id': self.session_id,
                    'response_time_ms': error_time_ms,
                    'status_code': 500,
                    'error': str(e)
                })
                
                span['error'] = str(e)
                raise
    
    @trace_function("call_bedrock_with_retries")
    def _call_bedrock_with_retries(self, prompt: str, phase: str, original_query: str, 
                                  max_retries: int = 3) -> str:
        """Call Bedrock with retries and comprehensive logging"""
        
        logger.info("Calling Bedrock with retries...")
        
        for attempt in range(max_retries):
            try:
                with observability.trace_operation("bedrock_api_call",
                                                 attempt=attempt + 1,
                                                 model="amazon.nova-pro-v1:0") as span:
                    
                    logger.info("=== Starting Bedrock query ===")
                    logger.info(f"Using model: amazon.nova-pro-v1:0")
                    
                    # Prepare request
                    request_body = {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        "inferenceConfig": {
                            "max_new_tokens": 1000,
                            "temperature": 0.1,
                            "top_p": 0.9
                        }
                    }
                    
                    span['metadata']['request_body_size'] = len(json.dumps(request_body))
                    span['metadata']['max_tokens'] = 1000
                    span['metadata']['temperature'] = 0.1
                    
                    # Log Bedrock operation
                    observability.log_bedrock_operation(
                        operation_type='llm_invoke',
                        model_id='amazon.nova-pro-v1:0',
                        input_data={
                            'prompt': prompt,
                            'max_tokens': 1000,
                            'temperature': 0.1,
                            'original_query': original_query
                        }
                    )
                    
                    logger.info("Using Nova Pro format")
                    logger.info(f"Request body prepared: {json.dumps(request_body, indent=2)[:500]}...")
                    logger.info("Making API call to Bedrock...")
                    
                    # Make the API call
                    start_time = time.time()
                    response = self.bedrock_client.invoke_model(
                        modelId="amazon.nova-pro-v1:0",
                        body=json.dumps(request_body),
                        contentType="application/json"
                    )
                    end_time = time.time()
                    
                    api_duration = end_time - start_time
                    span['metadata']['api_duration_seconds'] = api_duration
                    
                    logger.info("Bedrock API call successful, parsing response...")
                    
                    # Parse response
                    response_body = json.loads(response['body'].read())
                    logger.info(f"Raw response body: {json.dumps(response_body, indent=2)[:500]}...")
                    
                    logger.info("Parsing Nova Pro response...")
                    
                    # Extract text from Nova Pro response
                    if 'output' in response_body and 'message' in response_body['output']:
                        message = response_body['output']['message']
                        if 'content' in message and len(message['content']) > 0:
                            response_text = message['content'][0].get('text', '')
                        else:
                            response_text = ''
                    else:
                        response_text = ''
                    
                    logger.info(f"Extracted Nova Pro text: {response_text[:100]}...")
                    
                    if response_text.strip():
                        logger.info(f"Bedrock response received: {response_text[:100]}...")
                        
                        # Log successful Bedrock operation
                        observability.log_bedrock_operation(
                            operation_type='llm_invoke_response',
                            model_id='amazon.nova-pro-v1:0',
                            input_data={'original_query': original_query},
                            output_data={'text': response_text},
                            metrics={
                                'api_duration_seconds': api_duration,
                                'response_length': len(response_text),
                                'attempt': attempt + 1
                            }
                        )
                        
                        span['metadata']['response_length'] = len(response_text)
                        span['metadata']['success'] = True
                        
                        return response_text
                    else:
                        logger.warning("Empty response from Bedrock")
                        span['metadata']['success'] = False
                        span['metadata']['error'] = 'empty_response'
                        
                        if attempt < max_retries - 1:
                            continue
                        else:
                            return "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."
            
            except Exception as e:
                logger.error(f"Bedrock call attempt {attempt + 1} failed: {e}")
                
                observability.log_error(e, {
                    'operation': 'bedrock_api_call',
                    'attempt': attempt + 1,
                    'model': 'amazon.nova-pro-v1:0',
                    'original_query': original_query[:100]
                })
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return "I'm sorry, I'm experiencing technical difficulties. Please try again later."
    
    @trace_function("check_guardrails")
    def _check_guardrails(self, user_input: str) -> Dict[str, Any]:
        """Check input against guardrails with logging"""
        
        # Basic content filtering
        inappropriate_patterns = [
            r'\b(hack|attack|exploit)\b',
            r'\b(personal\s+information|ssn|social\s+security)\b',
            r'\b(credit\s+card|password|login)\b'
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, user_input.lower()):
                return {
                    'passed': False,
                    'reason': f'Pattern matched: {pattern}',
                    'confidence': 0.9,
                    'message': "I can't help with that request. Please ask something related to vehicle recovery services."
                }
        
        # Length check
        if len(user_input) > 4000:
            return {
                'passed': False,
                'reason': 'Input too long',
                'confidence': 1.0,
                'message': "Your message is too long. Please keep it under 4000 characters."
            }
        
        return {
            'passed': True,
            'reason': 'All checks passed',
            'confidence': 1.0,
            'message': None
        }
    
    @trace_function("build_prompt")
    def _build_prompt(self, user_input: str, phase: str, knowledge_context: Dict = None) -> str:
        """Build prompt with knowledge context"""
        
        base_prompt = """You are SVL, an empathetic but professional assistant helping users report and locate stolen vehicles. Always protect user privacy, never request or reveal sensitive PERSONAL information (like SSNs, credit card numbers, passwords, etc.). However, you can and should discuss pricing, costs, and service fees as these are legitimate business topics that help customers understand our services."""
        
        # Add pricing-specific context if user is asking about pricing
        pricing_keywords = ["price", "cost", "fee", "charge", "payment", "billing", "offer", "pricing", "expensive", "cheap", "money"]
        contact_keywords = ["contact", "phone", "email", "call", "reach", "support", "help", "toll free", "number"]
        
        # Add specific guidance for pricing queries
        if any(keyword in user_input.lower() for keyword in pricing_keywords):
            base_prompt += """

PRICING ASSISTANCE: You are authorized and encouraged to discuss SVL service pricing and costs. This is normal business information that helps customers make informed decisions. You can provide general pricing information, direct users to our sales team for detailed quotes, and discuss payment options. Pricing discussions are completely appropriate and expected for a business service."""

        # Add contact information guidance
        if any(keyword in user_input.lower() for keyword in contact_keywords):
            base_prompt += """

CONTACT INFORMATION: Always provide these SVL contact details when asked:

**Customer Support:**
- Phone: 1-800-555-0123 (24/7)
- Email: support@svlservices.com

**Emergency Hotline:**
- Phone: 1-800-911-FIND (1-800-911-3463) (24/7)
- Email: emergency@svlservices.com

**Technical Support:**
- Phone: 1-800-555-TECH (1-800-555-8324)
- Email: tech@svlservices.com

**Sales & Pricing:**
- Phone: 1-800-555-SALE (1-800-555-7253)
- Email: sales@svlservices.com

**Billing Support:**
- Phone: 1-800-555-BILL (1-800-555-2455)
- Email: billing@svlservices.com"""
        
        if knowledge_context and knowledge_context.get("context"):
            context_prompt = f"""
            
Based on the following relevant information from our knowledge base:

{knowledge_context["context"]}

Please answer the user's question: {user_input}

Use the provided context to give accurate, helpful information. If the context doesn't fully answer the question, provide general guidance while noting what specific information might need to be obtained through other channels."""
        else:
            context_prompt = f"""

User question: {user_input}

Please provide helpful information about vehicle theft reporting and recovery services. If you need specific information that's not in your knowledge base, guide the user appropriately."""
        
        return base_prompt + context_prompt
    
    @trace_function("save_conversation_to_db")
    def _save_conversation_to_db(self, user_input: str, response: str, phase: str, 
                                knowledge_context: Dict = None):
        """Save conversation to database with observability"""
        try:
            conversation_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'user_input': user_input,
                'ai_response': response,
                'phase': phase,
                'knowledge_base_used': knowledge_context is not None,
                'sources': knowledge_context.get("sources", []) if knowledge_context else []
            }
            
            # Save to database (implement based on your database schema)
            # self.db.save_conversation(conversation_data)
            
        except Exception as e:
            logger.error(f"Failed to save conversation to database: {e}")
            observability.log_error(e, {
                'operation': 'save_conversation_to_db',
                'session_id': self.session_id
            })
    
    @trace_function("load_conversation_history")
    def _load_conversation_history(self):
        """Load conversation history from database"""
        try:
            # Load from database (implement based on your database schema)
            # self.conversation_history = self.db.load_conversation_history(self.session_id)
            pass
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            observability.log_error(e, {
                'operation': 'load_conversation_history',
                'session_id': self.session_id
            })
    
    @trace_function("get_knowledge_base_stats")
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics with observability"""
        if self.knowledge_base:
            return self.knowledge_base.get_stats()
        else:
            return {"status": "not_available", "message": "Bedrock Knowledge base not initialized"}
    
    @trace_function("get_knowledge_base_health")
    def get_knowledge_base_health(self) -> Dict[str, Any]:
        """Get knowledge base health status with observability"""
        if self.knowledge_base:
            return self.knowledge_base.check_knowledge_base_status()
        else:
            return {"status": "unavailable", "message": "Bedrock Knowledge base not initialized"} 