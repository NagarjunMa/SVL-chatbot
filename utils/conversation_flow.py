"""
Advanced Conversation Flow System for SVL Chatbot
Comprehensive state machine with multi-turn conversation handling, 
intelligent routing, and workflow orchestration.
"""

import json
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import re
import logging
from collections import deque

from utils.logger import get_logger
from utils.database_manager import DatabaseManager
from data.models import Message, VehicleInfo, OwnerInfo, IncidentInfo, InsuranceInfo

logger = get_logger("conversation_flow")

class ConversationState(Enum):
    """Conversation state definitions"""
    INITIAL = "initial"
    GREETING = "greeting" 
    ASSESSMENT = "assessment"
    INFO_COLLECTION = "info_collection"
    VEHICLE_DETAILS = "vehicle_details"
    OWNER_DETAILS = "owner_details"
    INCIDENT_DETAILS = "incident_details"
    INSURANCE_DETAILS = "insurance_details"
    VERIFICATION = "verification"
    TICKET_CREATION = "ticket_creation"
    CONFIRMATION = "confirmation"
    PROCESS_EXPLANATION = "process_explanation"
    QA_SUPPORT = "qa_support"
    ESCALATION = "escalation"
    FOLLOW_UP = "follow_up"
    COMPLETED = "completed"
    ERROR_RECOVERY = "error_recovery"

class Intent(Enum):
    """User intent classifications"""
    REPORT_THEFT = "report_theft"
    ASK_QUESTION = "ask_question"
    GET_STATUS = "get_status"
    PROVIDE_INFO = "provide_info"
    CONFIRM_ACTION = "confirm_action"
    REQUEST_HELP = "request_help"
    ESCALATE = "escalate"
    MODIFY_INFO = "modify_info"
    CONTINUE_PROCESS = "continue_process"
    START_OVER = "start_over"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"

class Priority(Enum):
    """Case priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    EMERGENCY = "emergency"

@dataclass
class ConversationContext:
    """Enhanced conversation context with memory and state tracking"""
    user_id: str
    conversation_id: str
    current_state: ConversationState
    previous_state: Optional[ConversationState] = None
    intent_history: List[Intent] = field(default_factory=list)
    collected_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    retry_count: int = 0
    priority: Priority = Priority.MEDIUM
    escalation_reason: Optional[str] = None
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_memory: deque = field(default_factory=lambda: deque(maxlen=10))
    metadata: Dict[str, Any] = field(default_factory=dict)

class IntentRecognizer:
    """Advanced intent recognition with contextual awareness"""
    
    def __init__(self):
        self.intent_patterns = {
            Intent.REPORT_THEFT: [
                r"stolen|theft|missing|taken|car.*gone|vehicle.*missing",
                r"report.*theft|file.*report|stolen.*vehicle",
                r"my.*car.*stolen|someone.*took.*my"
            ],
            Intent.ASK_QUESTION: [
                r"what.*is|how.*does|can.*you.*tell|explain|info.*about",
                r"question|ask|wondering|curious|help.*understand"
            ],
            Intent.GET_STATUS: [
                r"status|update|progress|what.*happening|any.*news",
                r"check.*on|follow.*up|case.*status"
            ],
            Intent.PROVIDE_INFO: [
                r"here.*is|this.*is|my.*\w+.*is|the.*\w+.*is",
                r"license.*plate|vin.*number|phone.*number"
            ],
            Intent.CONFIRM_ACTION: [
                r"yes|yeah|correct|right|that.*right|confirm",
                r"proceed|continue|go.*ahead|next.*step"
            ],
            Intent.REQUEST_HELP: [
                r"help|assist|support|don.*know|confused|stuck"
            ],
            Intent.ESCALATE: [
                r"manager|supervisor|human|person|speak.*to.*someone",
                r"escalate|transfer|not.*working|frustrated"
            ],
            Intent.EMERGENCY: [
                r"emergency|urgent|immediate|911|help.*now|crisis"
            ],
            Intent.START_OVER: [
                r"start.*over|begin.*again|restart|reset|new.*report"
            ]
        }
    
    def recognize_intent(self, message: str, context: ConversationContext) -> Intent:
        """Recognize user intent with contextual awareness"""
        message_lower = message.lower().strip()
        
        # Emergency detection (highest priority)
        if any(re.search(pattern, message_lower) for pattern in self.intent_patterns[Intent.EMERGENCY]):
            return Intent.EMERGENCY
        
        # Context-aware intent recognition
        if context.current_state == ConversationState.VERIFICATION:
            if any(re.search(r"yes|correct|right|confirm", message_lower) for _ in [1]):
                return Intent.CONFIRM_ACTION
            elif any(re.search(r"no|wrong|incorrect|change", message_lower) for _ in [1]):
                return Intent.MODIFY_INFO
        
        # Pattern-based recognition
        for intent, patterns in self.intent_patterns.items():
            if any(re.search(pattern, message_lower) for pattern in patterns):
                return intent
        
        # Contextual fallbacks
        if context.current_state in [ConversationState.VEHICLE_DETAILS, 
                                   ConversationState.OWNER_DETAILS,
                                   ConversationState.INCIDENT_DETAILS]:
            return Intent.PROVIDE_INFO
        
        return Intent.UNKNOWN

class WorkflowOrchestrator:
    """Orchestrates complex multi-step workflows"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.workflows = {
            "theft_report": self._theft_report_workflow,
            "information_update": self._information_update_workflow,
            "escalation": self._escalation_workflow
        }
    
    async def execute_workflow(self, workflow_name: str, context: ConversationContext) -> Dict[str, Any]:
        """Execute a specific workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        try:
            return await self.workflows[workflow_name](context)
        except Exception as e:
            logger.error(f"Workflow {workflow_name} failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _theft_report_workflow(self, context: ConversationContext) -> Dict[str, Any]:
        """Complete theft report workflow"""
        steps = [
            ("vehicle_info", self._collect_vehicle_info),
            ("owner_info", self._collect_owner_info),
            ("incident_info", self._collect_incident_info),
            ("insurance_info", self._collect_insurance_info),
            ("verification", self._verify_information),
            ("ticket_creation", self._create_ticket)
        ]
        
        results = {}
        for step_name, step_func in steps:
            try:
                result = await step_func(context)
                results[step_name] = result
                if result.get("status") == "error":
                    break
            except Exception as e:
                results[step_name] = {"status": "error", "error": str(e)}
                break
        
        return results
    
    async def _collect_vehicle_info(self, context: ConversationContext) -> Dict[str, Any]:
        """Collect vehicle information"""
        required_fields = ["make", "model", "year", "color", "vin", "license_plate"]
        collected_fields = {k: v for k, v in context.collected_data.items() if k in required_fields}
        
        missing_fields = [f for f in required_fields if f not in collected_fields]
        
        if missing_fields:
            return {
                "status": "incomplete", 
                "missing_fields": missing_fields,
                "collected": collected_fields
            }
        
        return {"status": "complete", "data": collected_fields}
    
    async def _collect_owner_info(self, context: ConversationContext) -> Dict[str, Any]:
        """Collect owner information"""
        required_fields = ["owner_name", "phone", "email", "address"]
        collected_fields = {k: v for k, v in context.collected_data.items() if k in required_fields}
        
        missing_fields = [f for f in required_fields if f not in collected_fields]
        
        if missing_fields:
            return {
                "status": "incomplete",
                "missing_fields": missing_fields, 
                "collected": collected_fields
            }
        
        return {"status": "complete", "data": collected_fields}
    
    async def _collect_incident_info(self, context: ConversationContext) -> Dict[str, Any]:
        """Collect incident information"""
        required_fields = ["incident_date", "incident_time", "location", "circumstances"]
        collected_fields = {k: v for k, v in context.collected_data.items() if k in required_fields}
        
        missing_fields = [f for f in required_fields if f not in collected_fields]
        
        if missing_fields:
            return {
                "status": "incomplete",
                "missing_fields": missing_fields,
                "collected": collected_fields
            }
        
        return {"status": "complete", "data": collected_fields}
    
    async def _collect_insurance_info(self, context: ConversationContext) -> Dict[str, Any]:
        """Collect insurance information"""
        required_fields = ["insurance_company", "policy_number"]
        collected_fields = {k: v for k, v in context.collected_data.items() if k in required_fields}
        
        missing_fields = [f for f in required_fields if f not in collected_fields]
        
        if missing_fields:
            return {
                "status": "incomplete",
                "missing_fields": missing_fields,
                "collected": collected_fields
            }
        
        return {"status": "complete", "data": collected_fields}
    
    async def _verify_information(self, context: ConversationContext) -> Dict[str, Any]:
        """Verify collected information"""
        # Implementation for verification logic
        return {"status": "pending_confirmation"}
    
    async def _create_ticket(self, context: ConversationContext) -> Dict[str, Any]:
        """Create support ticket"""
        try:
            # Create data models
            vehicle_info = VehicleInfo(**{k: v for k, v in context.collected_data.items() 
                                        if k in ["make", "model", "year", "color", "vin", "license_plate"]})
            
            owner_info = OwnerInfo(**{k: v for k, v in context.collected_data.items()
                                    if k in ["name", "phone", "email", "address"]})
            
            # Create ticket through database manager
            ticket = await self.db_manager.create_ticket(
                context.user_id,
                vehicle_info,
                owner_info,
                None,  # incident_info - would need to be constructed
                None   # insurance_info - would need to be constructed
            )
            
            return {"status": "success", "ticket_id": ticket.ticket_id}
            
        except Exception as e:
            logger.error(f"Ticket creation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _information_update_workflow(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle information updates"""
        return {"status": "success", "message": "Information updated"}
    
    async def _escalation_workflow(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle escalation procedures"""
        context.priority = Priority.HIGH
        context.escalation_reason = "User requested escalation"
        
        return {
            "status": "escalated",
            "priority": context.priority.value,
            "reason": context.escalation_reason
        }

class ContextManager:
    """Advanced context management with conversation memory"""
    
    def __init__(self):
        self.contexts: Dict[str, ConversationContext] = {}
        self.context_timeout = timedelta(hours=2)
    
    def get_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context"""
        context = self.contexts.get(conversation_id)
        
        if context and self._is_context_expired(context):
            self.cleanup_context(conversation_id)
            return None
        
        return context
    
    def create_context(self, user_id: str, conversation_id: str) -> ConversationContext:
        """Create new conversation context"""
        context = ConversationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            current_state=ConversationState.INITIAL
        )
        
        self.contexts[conversation_id] = context
        return context
    
    def update_context(self, conversation_id: str, **updates) -> bool:
        """Update conversation context"""
        context = self.get_context(conversation_id)
        if not context:
            return False
        
        for key, value in updates.items():
            if hasattr(context, key):
                setattr(context, key, value)
        
        context.last_activity = datetime.now(timezone.utc)
        return True
    
    def add_to_memory(self, conversation_id: str, role: str, content: str):
        """Add message to conversation memory"""
        context = self.get_context(conversation_id)
        if context:
            context.conversation_memory.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    def _is_context_expired(self, context: ConversationContext) -> bool:
        """Check if context has expired"""
        return datetime.now(timezone.utc) - context.last_activity > self.context_timeout
    
    def cleanup_context(self, conversation_id: str):
        """Cleanup expired context"""
        if conversation_id in self.contexts:
            del self.contexts[conversation_id]
    
    def cleanup_expired_contexts(self):
        """Cleanup all expired contexts"""
        expired_ids = [
            conv_id for conv_id, context in self.contexts.items()
            if self._is_context_expired(context)
        ]
        
        for conv_id in expired_ids:
            self.cleanup_context(conv_id) 