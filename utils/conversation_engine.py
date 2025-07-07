"""
Advanced Conversation Engine for SVL Chatbot
Main orchestrator for conversation flow with state machine management,
response generation, and intelligent routing.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import traceback

from utils.conversation_flow import (
    ConversationState, Intent, Priority, ConversationContext,
    IntentRecognizer, WorkflowOrchestrator, ContextManager
)
from utils.conversation_manager import ConversationManager
from utils.database_manager import DatabaseManager
from utils.logger import get_logger
from data.models import Message

logger = get_logger("conversation_engine")

class StateTransitionEngine:
    """Manages state transitions and flow control"""
    
    def __init__(self):
        self.transition_rules = {
            ConversationState.INITIAL: {
                Intent.REPORT_THEFT: ConversationState.ASSESSMENT,
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT,
                Intent.EMERGENCY: ConversationState.ESCALATION,
                Intent.UNKNOWN: ConversationState.GREETING
            },
            ConversationState.GREETING: {
                Intent.REPORT_THEFT: ConversationState.ASSESSMENT,
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT,
                Intent.EMERGENCY: ConversationState.ESCALATION,
                Intent.REQUEST_HELP: ConversationState.PROCESS_EXPLANATION
            },
            ConversationState.ASSESSMENT: {
                Intent.CONFIRM_ACTION: ConversationState.INFO_COLLECTION,
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT,
                Intent.ESCALATE: ConversationState.ESCALATION,
                Intent.START_OVER: ConversationState.GREETING
            },
            ConversationState.INFO_COLLECTION: {
                Intent.PROVIDE_INFO: ConversationState.VEHICLE_DETAILS,
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT,
                Intent.REQUEST_HELP: ConversationState.PROCESS_EXPLANATION
            },
            ConversationState.VEHICLE_DETAILS: {
                Intent.PROVIDE_INFO: ConversationState.OWNER_DETAILS,
                Intent.MODIFY_INFO: ConversationState.VEHICLE_DETAILS,
                Intent.CONTINUE_PROCESS: ConversationState.OWNER_DETAILS,
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT
            },
            ConversationState.OWNER_DETAILS: {
                Intent.PROVIDE_INFO: ConversationState.INCIDENT_DETAILS,
                Intent.MODIFY_INFO: ConversationState.OWNER_DETAILS,
                Intent.CONTINUE_PROCESS: ConversationState.INCIDENT_DETAILS,
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT
            },
            ConversationState.INCIDENT_DETAILS: {
                Intent.PROVIDE_INFO: ConversationState.INSURANCE_DETAILS,
                Intent.CONTINUE_PROCESS: ConversationState.INSURANCE_DETAILS,
                Intent.MODIFY_INFO: ConversationState.INCIDENT_DETAILS,
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT
            },
            ConversationState.INSURANCE_DETAILS: {
                Intent.PROVIDE_INFO: ConversationState.VERIFICATION,
                Intent.CONTINUE_PROCESS: ConversationState.VERIFICATION,
                Intent.MODIFY_INFO: ConversationState.INSURANCE_DETAILS,
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT
            },
            ConversationState.VERIFICATION: {
                Intent.CONFIRM_ACTION: ConversationState.TICKET_CREATION,
                Intent.MODIFY_INFO: ConversationState.INFO_COLLECTION,
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT
            },
            ConversationState.TICKET_CREATION: {
                Intent.UNKNOWN: ConversationState.CONFIRMATION,
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT
            },
            ConversationState.CONFIRMATION: {
                Intent.ASK_QUESTION: ConversationState.QA_SUPPORT,
                Intent.REQUEST_HELP: ConversationState.PROCESS_EXPLANATION,
                Intent.ESCALATE: ConversationState.ESCALATION,
                Intent.START_OVER: ConversationState.GREETING,
                Intent.UNKNOWN: ConversationState.FOLLOW_UP
            },
            ConversationState.QA_SUPPORT: {
                Intent.REPORT_THEFT: ConversationState.ASSESSMENT,
                Intent.CONTINUE_PROCESS: ConversationState.INFO_COLLECTION,
                Intent.ESCALATE: ConversationState.ESCALATION,
                Intent.START_OVER: ConversationState.GREETING
            },
            ConversationState.ESCALATION: {
                Intent.CONTINUE_PROCESS: ConversationState.FOLLOW_UP,
                Intent.START_OVER: ConversationState.GREETING
            },
            ConversationState.ERROR_RECOVERY: {
                Intent.CONTINUE_PROCESS: ConversationState.INFO_COLLECTION,
                Intent.START_OVER: ConversationState.GREETING,
                Intent.ESCALATE: ConversationState.ESCALATION
            }
        }
    
    def get_next_state(self, current_state: ConversationState, intent: Intent) -> ConversationState:
        """Determine next state based on current state and intent"""
        if current_state in self.transition_rules:
            return self.transition_rules[current_state].get(intent, current_state)
        return current_state

class ResponseGenerator:
    """Generates contextual responses based on conversation state and intent"""
    
    def __init__(self, conversation_manager: ConversationManager):
        self.conversation_manager = conversation_manager
        self.response_templates = {
            ConversationState.GREETING: {
                Intent.REPORT_THEFT: "I'm here to help you report your stolen vehicle. Let's start by gathering some information about what happened.",
                Intent.ASK_QUESTION: "I'm happy to answer your questions about our vehicle recovery services. What would you like to know?",
                Intent.UNKNOWN: "Hello! I'm SVL Assistant, here to help you with stolen vehicle reports and recovery services. How can I assist you today?"
            },
            ConversationState.ASSESSMENT: {
                Intent.CONFIRM_ACTION: "Thank you for confirming. Let's begin collecting the necessary information for your stolen vehicle report.",
                Intent.ASK_QUESTION: "I understand you have questions. I'm here to help clarify anything about the reporting process."
            },
            ConversationState.VEHICLE_DETAILS: {
                Intent.PROVIDE_INFO: "Thank you for providing the vehicle information. Now I need to collect your contact details.",
                Intent.MODIFY_INFO: "I'll help you update the vehicle information. What would you like to change?"
            },
            ConversationState.VERIFICATION: {
                Intent.CONFIRM_ACTION: "Perfect! I'll now create your stolen vehicle report ticket.",
                Intent.MODIFY_INFO: "No problem, let's update the information that needs to be corrected."
            },
            ConversationState.ESCALATION: {
                Intent.UNKNOWN: "I understand you need additional assistance. Let me connect you with our specialized support team."
            },
            ConversationState.ERROR_RECOVERY: {
                Intent.CONTINUE_PROCESS: "Let's try that again. I'm here to help you complete your report.",
                Intent.START_OVER: "No problem, let's start fresh with your stolen vehicle report."
            }
        }
    
    async def generate_response(self, context: ConversationContext, intent: Intent, user_message: str) -> str:
        """Generate contextual response"""
        try:
            # Get template response
            template_response = self._get_template_response(context.current_state, intent)
            
            # For QA support, use knowledge base
            if context.current_state == ConversationState.QA_SUPPORT:
                if self.conversation_manager.knowledge_base and self.conversation_manager.knowledge_base.is_ready():
                    kb_result = self.conversation_manager.knowledge_base.query_knowledge_base(
                        user_query=user_message,
                        query_type=self._determine_query_type(user_message),
                        include_context=True
                    )
                    
                    if kb_result["status"] == "success" and kb_result.get("total_results", 0) > 0:
                        # Use conversation manager to generate AI response with knowledge base context
                        ai_response = self.conversation_manager.process_user_input(
                            user_input=user_message,
                            phase="faq",
                            extra={"knowledge_base_context": kb_result["context"]}
                        )
                        return ai_response
            
            # Use template or generate contextual response
            if template_response:
                return self._enhance_template_response(template_response, context, user_message)
            else:
                return await self._generate_ai_response(context, intent, user_message)
        
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I'm experiencing some technical difficulties. Let me connect you with a human agent who can assist you."
    
    def _get_template_response(self, state: ConversationState, intent: Intent) -> Optional[str]:
        """Get template response for state and intent"""
        return self.response_templates.get(state, {}).get(intent)
    
    def _enhance_template_response(self, template: str, context: ConversationContext, user_message: str) -> str:
        """Enhance template response with context"""
        # Add personalization based on context
        if context.collected_data.get("owner_name"):
            template = template.replace("you", f"you, {context.collected_data['owner_name']}")
        
        # Add specific guidance based on missing information
        if context.current_state == ConversationState.VEHICLE_DETAILS:
            missing_fields = self._get_missing_vehicle_fields(context)
            if missing_fields:
                template += f" I'll need the following information: {', '.join(missing_fields)}."
        
        return template
    
    async def _generate_ai_response(self, context: ConversationContext, intent: Intent, user_message: str) -> str:
        """Generate AI response using conversation manager"""
        phase = self._map_state_to_phase(context.current_state)
        
        extra_context = {
            "conversation_state": context.current_state.value,
            "intent": intent.value,
            "collected_data": context.collected_data,
            "conversation_memory": list(context.conversation_memory)
        }
        
        return self.conversation_manager.process_user_input(
            user_input=user_message,
            phase=phase,
            extra=extra_context
        )
    
    def _determine_query_type(self, message: str) -> str:
        """Determine query type for knowledge base"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["process", "procedure", "step", "how to"]):
            return "process"
        elif any(word in message_lower for word in ["price", "cost", "fee", "payment"]):
            return "faq"
        elif any(word in message_lower for word in ["contact", "phone", "email", "reach"]):
            return "contact"
        elif any(word in message_lower for word in ["policy", "rule", "requirement"]):
            return "sop"
        else:
            return "general"
    
    def _map_state_to_phase(self, state: ConversationState) -> str:
        """Map conversation state to conversation manager phase"""
        state_phase_map = {
            ConversationState.GREETING: "greeting",
            ConversationState.ASSESSMENT: "collect_info",
            ConversationState.INFO_COLLECTION: "collect_info",
            ConversationState.VEHICLE_DETAILS: "collect_info",
            ConversationState.OWNER_DETAILS: "collect_info",
            ConversationState.INCIDENT_DETAILS: "collect_info",
            ConversationState.INSURANCE_DETAILS: "collect_info",
            ConversationState.VERIFICATION: "confirmation",
            ConversationState.CONFIRMATION: "confirmation",
            ConversationState.QA_SUPPORT: "faq",
            ConversationState.PROCESS_EXPLANATION: "process_explanation"
        }
        return state_phase_map.get(state, "greeting")
    
    def _get_missing_vehicle_fields(self, context: ConversationContext) -> List[str]:
        """Get missing vehicle information fields"""
        required_fields = ["make", "model", "year", "color", "vin", "license_plate"]
        return [field for field in required_fields if field not in context.collected_data]

class ConversationEngine:
    """Main conversation engine orchestrator"""
    
    def __init__(self, db_manager: DatabaseManager, conversation_manager: ConversationManager):
        self.db_manager = db_manager
        self.conversation_manager = conversation_manager
        self.intent_recognizer = IntentRecognizer()
        self.workflow_orchestrator = WorkflowOrchestrator(db_manager)
        self.context_manager = ContextManager()
        self.state_engine = StateTransitionEngine()
        self.response_generator = ResponseGenerator(conversation_manager)
        self.performance_metrics = {
            "total_conversations": 0,
            "successful_completions": 0,
            "escalations": 0,
            "errors": 0,
            "average_response_time": 0.0
        }
    
    async def process_message(self, user_id: str, conversation_id: str, message: str) -> Dict[str, Any]:
        """Main message processing pipeline"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get or create conversation context
            context = self.context_manager.get_context(conversation_id)
            if not context:
                context = self.context_manager.create_context(user_id, conversation_id)
                self.performance_metrics["total_conversations"] += 1
            
            # Add user message to memory
            self.context_manager.add_to_memory(conversation_id, "user", message)
            
            # Recognize intent
            intent = self.intent_recognizer.recognize_intent(message, context)
            context.intent_history.append(intent)
            
            # Handle emergency intents immediately
            if intent == Intent.EMERGENCY:
                return await self._handle_emergency(context, message)
            
            # Determine next state
            previous_state = context.current_state
            next_state = self.state_engine.get_next_state(context.current_state, intent)
            
            # Update context
            context.previous_state = previous_state
            context.current_state = next_state
            
            # Process state-specific logic
            processing_result = await self._process_state_logic(context, intent, message)
            
            # Generate response
            response = await self.response_generator.generate_response(context, intent, message)
            
            # Add assistant response to memory
            self.context_manager.add_to_memory(conversation_id, "assistant", response)
            
            # Calculate metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._update_performance_metrics(processing_time, True)
            
            return {
                "status": "success",
                "response": response,
                "state": next_state.value,
                "intent": intent.value,
                "processing_time": processing_time,
                "context_data": processing_result
            }
        
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            logger.error(traceback.format_exc())
            
            # Error recovery
            error_response = await self._handle_error(conversation_id, str(e))
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._update_performance_metrics(processing_time, False)
            
            return {
                "status": "error",
                "response": error_response,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def _process_state_logic(self, context: ConversationContext, intent: Intent, message: str) -> Dict[str, Any]:
        """Process state-specific business logic"""
        if context.current_state == ConversationState.TICKET_CREATION:
            return await self._handle_ticket_creation(context)
        elif context.current_state == ConversationState.ESCALATION:
            return await self._handle_escalation(context, message)
        elif context.current_state in [ConversationState.VEHICLE_DETAILS, 
                                      ConversationState.OWNER_DETAILS,
                                      ConversationState.INCIDENT_DETAILS,
                                      ConversationState.INSURANCE_DETAILS]:
            return await self._handle_information_collection(context, intent, message)
        elif context.current_state == ConversationState.VERIFICATION:
            return await self._handle_verification(context, intent)
        
        return {"status": "processed"}
    
    async def _handle_information_collection(self, context: ConversationContext, intent: Intent, message: str) -> Dict[str, Any]:
        """Handle information collection states"""
        if intent == Intent.PROVIDE_INFO:
            # Extract information from message
            extracted_info = self._extract_information(message, context.current_state)
            context.collected_data.update(extracted_info)
            
            return {
                "status": "info_collected",
                "extracted_info": extracted_info,
                "total_collected": len(context.collected_data)
            }
        
        return {"status": "awaiting_info"}
    
    async def _handle_ticket_creation(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle ticket creation workflow"""
        try:
            result = await self.workflow_orchestrator.execute_workflow("theft_report", context)
            
            if result.get("ticket_creation", {}).get("status") == "success":
                ticket_id = result["ticket_creation"]["ticket_id"]
                context.current_state = ConversationState.CONFIRMATION
                self.performance_metrics["successful_completions"] += 1
                
                return {
                    "status": "ticket_created",
                    "ticket_id": ticket_id,
                    "workflow_result": result
                }
            else:
                context.current_state = ConversationState.ERROR_RECOVERY
                return {
                    "status": "ticket_creation_failed",
                    "workflow_result": result
                }
        
        except Exception as e:
            logger.error(f"Ticket creation failed: {e}")
            context.current_state = ConversationState.ERROR_RECOVERY
            return {"status": "error", "error": str(e)}
    
    async def _handle_escalation(self, context: ConversationContext, message: str) -> Dict[str, Any]:
        """Handle escalation procedures"""
        try:
            result = await self.workflow_orchestrator.execute_workflow("escalation", context)
            self.performance_metrics["escalations"] += 1
            
            return {
                "status": "escalated",
                "escalation_result": result,
                "priority": context.priority.value
            }
        
        except Exception as e:
            logger.error(f"Escalation failed: {e}")
            return {"status": "escalation_failed", "error": str(e)}
    
    async def _handle_verification(self, context: ConversationContext, intent: Intent) -> Dict[str, Any]:
        """Handle information verification"""
        if intent == Intent.CONFIRM_ACTION:
            return {"status": "verified", "action": "proceed_to_ticket_creation"}
        elif intent == Intent.MODIFY_INFO:
            context.current_state = ConversationState.INFO_COLLECTION
            return {"status": "modification_requested"}
        
        return {"status": "awaiting_verification"}
    
    async def _handle_emergency(self, context: ConversationContext, message: str) -> Dict[str, Any]:
        """Handle emergency situations"""
        context.priority = Priority.EMERGENCY
        context.current_state = ConversationState.ESCALATION
        
        emergency_response = """
🚨 **EMERGENCY DETECTED** 🚨

If this is a life-threatening emergency, please call 911 immediately.

For immediate SVL emergency assistance:
📞 **Emergency Hotline**: 1-800-SVL-911
📧 **Emergency Email**: emergency@svlservices.com

I'm also escalating your case to our emergency response team.
"""
        
        return {
            "status": "emergency",
            "response": emergency_response,
            "priority": "EMERGENCY",
            "auto_escalated": True
        }
    
    async def _handle_error(self, conversation_id: str, error_message: str) -> str:
        """Handle errors with graceful degradation"""
        self.performance_metrics["errors"] += 1
        
        # Update context to error recovery state
        context = self.context_manager.get_context(conversation_id)
        if context:
            context.current_state = ConversationState.ERROR_RECOVERY
            context.errors.append(error_message)
            context.retry_count += 1
        
        if context and context.retry_count > 3:
            return "I'm experiencing persistent technical difficulties. Let me connect you with a human agent who can assist you immediately."
        
        return "I apologize for the technical issue. Let's try again. How can I help you with your stolen vehicle report?"
    
    def _extract_information(self, message: str, state: ConversationState) -> Dict[str, Any]:
        """Extract relevant information from user message"""
        extracted = {}
        message_lower = message.lower()
        
        # Vehicle information extraction
        if state == ConversationState.VEHICLE_DETAILS:
            # Extract make/model patterns
            for make in ["toyota", "honda", "ford", "chevrolet", "nissan", "bmw", "mercedes", "audi"]:
                if make in message_lower:
                    extracted["make"] = make.capitalize()
                    break
            
            # Extract year patterns
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', message)
            if year_match:
                extracted["year"] = int(year_match.group())
            
            # Extract license plate patterns
            plate_match = re.search(r'\b[A-Z0-9]{2,8}\b', message.upper())
            if plate_match:
                extracted["license_plate"] = plate_match.group()
        
        # Owner information extraction
        elif state == ConversationState.OWNER_DETAILS:
            # Extract phone numbers
            phone_match = re.search(r'\b\d{10}\b|\b\d{3}-\d{3}-\d{4}\b', message)
            if phone_match:
                phone = re.sub(r'\D', '', phone_match.group())
                if len(phone) == 10:
                    extracted["phone"] = phone
            
            # Extract email addresses
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message)
            if email_match:
                extracted["email"] = email_match.group()
        
        return extracted
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        if success:
            # Update average response time
            current_avg = self.performance_metrics["average_response_time"]
            total_processed = (self.performance_metrics["successful_completions"] + 
                             self.performance_metrics["errors"])
            
            if total_processed > 0:
                self.performance_metrics["average_response_time"] = (
                    (current_avg * (total_processed - 1) + processing_time) / total_processed
                )
            else:
                self.performance_metrics["average_response_time"] = processing_time
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation status and context"""
        context = self.context_manager.get_context(conversation_id)
        if not context:
            return {"status": "not_found"}
        
        return {
            "status": "active",
            "current_state": context.current_state.value,
            "priority": context.priority.value,
            "collected_data_count": len(context.collected_data),
            "conversation_length": len(context.conversation_memory),
            "last_activity": context.last_activity.isoformat(),
            "retry_count": context.retry_count,
            "escalation_reason": context.escalation_reason
        } 