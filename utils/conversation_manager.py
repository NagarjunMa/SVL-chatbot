import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from config.aws_config import get_bedrock_client, get_model_id, is_nova_model, is_claude_model
from utils.database_manager import DatabaseManager
from utils.logger import get_logger

logger = get_logger("conversation_manager")

# --- Guardrails ---
PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"\b\d{16}\b",              # Credit card
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",  # Email
    r"\b\d{10}\b",              # Phone number
]

FORBIDDEN_CONTENT = [
    "violence", "threat", "abuse", "illegal", "offensive"
]

# --- Conversation Memory ---
class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_context(self) -> List[Dict[str, str]]:
        return self.history.copy()

# --- Prompt Engineering ---
def build_prompt(phase: str, user_input: str, memory: ConversationMemory, extra: Optional[Dict[str, Any]] = None) -> str:
    """
    Build a prompt for the LLM based on the conversation phase and memory.
    """
    base_instructions = (
        "You are SVL, an empathetic but professional assistant helping users report and locate stolen vehicles. "
        "Always protect user privacy, never request or reveal sensitive information. "
        "Be concise, clear, and supportive."
    )
    
    phase_instructions = {
        "greeting": "Greet the user warmly, express empathy for their situation, and offer help with their stolen vehicle report.",
        "collect_info": "Ask for and confirm details about the stolen vehicle and incident. Guide them through the reporting process step by step.",
        "process_explanation": "Explain the SVL process and next steps clearly. Let them know what happens after they submit their report.",
        "faq": "Answer frequently asked questions clearly and concisely. Provide helpful information about vehicle theft and recovery.",
        "confirmation": "Confirm the ticket/report details and reassure the user that help is on the way. Provide next steps and contact information.",
    }
    
    # Build conversation context
    context_lines = []
    for msg in memory.get_context():
        role = msg['role'].capitalize()
        content = msg['content']
        context_lines.append(f"{role}: {content}")
    
    context = "\n".join(context_lines) if context_lines else "This is the start of the conversation."
    
    # Add extra context if provided
    extra_str = f"\nAdditional context: {json.dumps(extra)}" if extra else ""
    
    # Construct the full prompt
    prompt = f"""{base_instructions}

Current conversation phase: {phase}
Phase guidance: {phase_instructions.get(phase, 'Respond helpfully and professionally.')}

Conversation history:
{context}

{extra_str}

User's current message: {user_input}

Please respond as SVL Assistant:"""
    
    return prompt

# --- Guardrails ---
def contains_pii(text: str) -> bool:
    for pattern in PII_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def contains_forbidden_content(text: str) -> bool:
    for word in FORBIDDEN_CONTENT:
        if word in text.lower():
            return True
    return False

# --- Response Validation ---
def validate_response(response: str) -> bool:
    if not response or len(response.strip()) < 2:
        return False
    if contains_forbidden_content(response):
        return False
    return True

# --- Retry Logic ---
def call_with_retries(fn, max_retries=3, backoff=1.5, *args, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
            else:
                logger.error(f"All {max_retries} attempts failed.")
                raise

# --- Conversation Manager ---
class ConversationManager:
    def __init__(self, session_id: str):
        """
        Initialize conversation manager with session ID and AWS Bedrock client
        """
        self.session_id = session_id
        self.bedrock_client = get_bedrock_client()
        self.memory = ConversationMemory()
        self.db_manager = DatabaseManager()
        self.current_phase = "greeting"

    def process_user_input(self, user_input: str, phase: str, extra: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"=== Processing user input: '{user_input}' in phase: '{phase}' ===")
        
        # Guardrails: block PII or forbidden content
        if contains_pii(user_input):
            logger.warning("PII detected in user input. Blocking response.")
            return "For your safety, please do not share sensitive personal information."
        if contains_forbidden_content(user_input):
            logger.warning("Forbidden content detected in user input. Blocking response.")
            return "I'm unable to assist with that request."
        
        logger.info("Guardrails passed, building prompt...")
        
        # Build prompt
        prompt = build_prompt(phase, user_input, self.memory, extra)
        logger.info(f"Prompt for Bedrock: {prompt[:200]}...")
        
        logger.info("Calling Bedrock with retries...")
        
        # Call Bedrock with retries
        try:
            response = call_with_retries(self._query_bedrock, 3, 1.5, prompt)
            logger.info(f"Bedrock response received: {response[:100]}...")
        except Exception as e:
            logger.error(f"Bedrock API failed after retries: {e}")
            return "Sorry, I'm having trouble connecting to our AI service. Please try again later."
        
        # Validate response
        if not validate_response(response):
            logger.warning("Invalid or unsafe response from Bedrock.")
            return "Sorry, I couldn't generate a safe response."
        
        logger.info("Response validated successfully, adding to memory...")
        
        # Add to memory
        self.memory.add("user", user_input)
        self.memory.add("assistant", response)
        
        logger.info(f"=== Returning response: {response[:100]}... ===")
        return response

    def _query_bedrock(self, prompt: str) -> str:
        """
        Query AWS Bedrock with proper API format
        """
        try:
            logger.info("=== Starting Bedrock query ===")
            # Prepare the request body based on the model type
            model_id = get_model_id()
            logger.info(f"Using model: {model_id}")
            
            # Different models have different input formats
            if is_claude_model(model_id):
                logger.info("Using Claude format")
                # Claude format
                body = {
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": 500,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            elif is_nova_model(model_id):
                logger.info("Using Nova Pro format")
                # Amazon Nova Pro format
                body = {
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
                        "max_new_tokens": 500,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }
            else:
                logger.info("Using generic format")
                # Generic format
                body = {
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.7
                }
            
            logger.info(f"Request body prepared: {json.dumps(body, indent=2)[:300]}...")
            
            # Convert to JSON bytes as required by Bedrock
            logger.info("Making API call to Bedrock...")
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body).encode('utf-8'),
                contentType='application/json'
            )
            
            logger.info("Bedrock API call successful, parsing response...")
            
            # Parse response
            response_body = json.loads(response['body'].read())
            logger.info(f"Raw response body: {json.dumps(response_body, indent=2)[:500]}...")
            
            # Extract completion based on model type
            if is_claude_model(model_id):
                result = response_body.get('completion', '[No response from model]')
                logger.info(f"Extracted Claude completion: {result[:100]}...")
                return result
            elif is_nova_model(model_id):
                logger.info("Parsing Nova Pro response...")
                # Nova Pro response format
                if 'output' in response_body and 'message' in response_body['output']:
                    message = response_body['output']['message']
                    if 'content' in message and len(message['content']) > 0:
                        result = message['content'][0].get('text', '[No response from model]')
                        logger.info(f"Extracted Nova Pro text: {result[:100]}...")
                        return result
                logger.warning("Nova Pro response structure unexpected")
                return '[No response from model]'
            else:
                result = response_body.get('text', response_body.get('completion', '[No response from model]'))
                logger.info(f"Extracted generic response: {result[:100]}...")
                return result
                
        except Exception as e:
            logger.error(f"Bedrock query error: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_memory(self) -> List[Dict[str, str]]:
        return self.memory.get_context() 