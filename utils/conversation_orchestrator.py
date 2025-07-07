"""
Main Conversation Orchestrator for SVL Chatbot
Integrates all conversation components and provides the primary interface
for the SVL chatbot system with comprehensive error handling and monitoring.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import traceback
from contextlib import asynccontextmanager

from utils.conversation_engine import ConversationEngine
from utils.conversation_flow import ConversationState, Intent, Priority, ConversationContext
from utils.notification_system import NotificationSystem
from utils.conversation_manager import ConversationManager
from utils.database_manager import DatabaseManager
from utils.logger import get_logger

logger = get_logger("conversation_orchestrator")

class PerformanceMonitor:
    """Monitors system performance and health"""
    
    def __init__(self):
        self.metrics = {
            "total_conversations": 0,
            "active_conversations": 0,
            "completed_successfully": 0,
            "escalated_cases": 0,
            "error_count": 0,
            "average_completion_time": 0.0,
            "system_uptime": datetime.now(timezone.utc),
            "last_error": None,
            "response_times": [],
            "external_system_status": {}
        }
        self.health_checks = {}
    
    def record_conversation_start(self, conversation_id: str):
        """Record conversation start"""
        self.metrics["total_conversations"] += 1
        self.metrics["active_conversations"] += 1
        self.health_checks[conversation_id] = {
            "start_time": datetime.now(timezone.utc),
            "status": "active"
        }
    
    def record_conversation_end(self, conversation_id: str, success: bool):
        """Record conversation completion"""
        if conversation_id in self.health_checks:
            start_time = self.health_checks[conversation_id]["start_time"]
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            self.metrics["active_conversations"] = max(0, self.metrics["active_conversations"] - 1)
            
            if success:
                self.metrics["completed_successfully"] += 1
                self._update_average_completion_time(duration)
            
            self.health_checks[conversation_id]["status"] = "completed" if success else "failed"
            self.health_checks[conversation_id]["duration"] = duration
    
    def record_escalation(self, conversation_id: str, reason: str):
        """Record escalation"""
        self.metrics["escalated_cases"] += 1
        if conversation_id in self.health_checks:
            self.health_checks[conversation_id]["escalated"] = True
            self.health_checks[conversation_id]["escalation_reason"] = reason
    
    def record_error(self, error: str, conversation_id: Optional[str] = None):
        """Record error"""
        self.metrics["error_count"] += 1
        self.metrics["last_error"] = {
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conversation_id": conversation_id
        }
    
    def record_response_time(self, response_time: float):
        """Record response time"""
        self.metrics["response_times"].append(response_time)
        # Keep only last 100 response times
        if len(self.metrics["response_times"]) > 100:
            self.metrics["response_times"] = self.metrics["response_times"][-100:]
    
    def _update_average_completion_time(self, duration: float):
        """Update average completion time"""
        current_avg = self.metrics["average_completion_time"]
        completed_count = self.metrics["completed_successfully"]
        
        if completed_count == 1:
            self.metrics["average_completion_time"] = duration
        else:
            self.metrics["average_completion_time"] = (
                (current_avg * (completed_count - 1) + duration) / completed_count
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        uptime = datetime.now(timezone.utc) - self.metrics["system_uptime"]
        
        # Calculate success rate
        total_processed = self.metrics["completed_successfully"] + self.metrics["error_count"]
        success_rate = (self.metrics["completed_successfully"] / total_processed * 100) if total_processed > 0 else 100
        
        # Calculate average response time
        avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"]) if self.metrics["response_times"] else 0
        
        # Determine health status
        health_status = "healthy"
        if success_rate < 95:
            health_status = "degraded"
        if success_rate < 90 or avg_response_time > 10:
            health_status = "unhealthy"
        
        return {
            "status": health_status,
            "uptime_seconds": uptime.total_seconds(),
            "success_rate": round(success_rate, 2),
            "average_response_time": round(avg_response_time, 3),
            "active_conversations": self.metrics["active_conversations"],
            "total_conversations": self.metrics["total_conversations"],
            "completed_successfully": self.metrics["completed_successfully"],
            "escalated_cases": self.metrics["escalated_cases"],
            "error_count": self.metrics["error_count"],
            "last_error": self.metrics["last_error"]
        }

class ConversationOrchestrator:
    """Main orchestrator for the SVL conversation system"""
    
    def __init__(self, db_manager: DatabaseManager, conversation_manager: ConversationManager):
        self.db_manager = db_manager
        self.conversation_manager = conversation_manager
        self.conversation_engine = ConversationEngine(db_manager, conversation_manager)
        self.notification_system = NotificationSystem(db_manager)
        self.performance_monitor = PerformanceMonitor()
        
        # System configuration
        self.config = {
            "max_conversation_duration": timedelta(hours=4),
            "max_retry_attempts": 3,
            "escalation_threshold": timedelta(minutes=30),
            "auto_escalation_enabled": True,
            "notification_enabled": True,
            "performance_monitoring_enabled": True
        }
        
        # Background tasks
        self.background_tasks = []
        self.shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize the orchestrator and start background tasks"""
        logger.info("Initializing Conversation Orchestrator...")
        
        # Start background tasks
        if self.config["performance_monitoring_enabled"]:
            self.background_tasks.append(
                asyncio.create_task(self._performance_monitoring_task())
            )
        
        if self.config["notification_enabled"]:
            self.background_tasks.append(
                asyncio.create_task(self._notification_processing_task())
            )
        
        # Cleanup task
        self.background_tasks.append(
            asyncio.create_task(self._cleanup_task())
        )
        
        logger.info("Conversation Orchestrator initialized successfully")
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down Conversation Orchestrator...")
        
        # Signal shutdown to background tasks
        self.shutdown_event.set()
        
        # Wait for background tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Conversation Orchestrator shutdown complete")
    
    @asynccontextmanager
    async def conversation_session(self, user_id: str, conversation_id: str):
        """Context manager for conversation sessions"""
        self.performance_monitor.record_conversation_start(conversation_id)
        start_time = datetime.now(timezone.utc)
        success = False
        
        try:
            yield
            success = True
        except Exception as e:
            logger.error(f"Conversation session error: {e}")
            self.performance_monitor.record_error(str(e), conversation_id)
            raise
        finally:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.performance_monitor.record_response_time(duration)
            self.performance_monitor.record_conversation_end(conversation_id, success)
    
    async def process_user_message(self, user_id: str, conversation_id: str, message: str, 
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main entry point for processing user messages"""
        async with self.conversation_session(user_id, conversation_id):
            try:
                # Process message through conversation engine
                result = await self.conversation_engine.process_message(
                    user_id, conversation_id, message
                )
                
                # Handle state-specific post-processing
                await self._handle_post_processing(user_id, conversation_id, result, metadata)
                
                # Check for auto-escalation conditions
                await self._check_auto_escalation(conversation_id, result)
                
                return {
                    "status": "success",
                    "response": result["response"],
                    "conversation_state": result["state"],
                    "intent": result["intent"],
                    "processing_time": result["processing_time"],
                    "metadata": {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "context_data": result.get("context_data", {})
                    }
                }
            
            except Exception as e:
                logger.error(f"Message processing failed: {e}")
                logger.error(traceback.format_exc())
                
                # Generate error response
                error_response = await self._handle_processing_error(
                    user_id, conversation_id, str(e)
                )
                
                return {
                    "status": "error",
                    "response": error_response,
                    "error": str(e),
                    "metadata": {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error_type": type(e).__name__
                    }
                }
    
    async def _handle_post_processing(self, user_id: str, conversation_id: str, 
                                    result: Dict[str, Any], metadata: Optional[Dict[str, Any]]):
        """Handle post-processing based on conversation state"""
        state = result.get("state")
        context_data = result.get("context_data", {})
        
        # Handle ticket creation completion
        if state == ConversationState.CONFIRMATION.value and context_data.get("status") == "ticket_created":
            await self._handle_ticket_creation_complete(user_id, conversation_id, context_data)
        
        # Handle escalation
        elif state == ConversationState.ESCALATION.value:
            await self._handle_escalation_complete(user_id, conversation_id, context_data)
        
        # Handle completion
        elif state == ConversationState.COMPLETED.value:
            await self._handle_conversation_complete(user_id, conversation_id)
    
    async def _handle_ticket_creation_complete(self, user_id: str, conversation_id: str, 
                                             context_data: Dict[str, Any]):
        """Handle completed ticket creation"""
        if not self.config["notification_enabled"]:
            return
        
        try:
            ticket_id = context_data.get("ticket_id")
            if not ticket_id:
                logger.warning("Ticket creation complete but no ticket ID found")
                return
            
            # Get conversation context to extract ticket data
            context = self.conversation_engine.context_manager.get_context(conversation_id)
            if not context:
                logger.warning("No conversation context found for ticket notifications")
                return
            
            # Prepare ticket data for notifications
            ticket_data = {
                "ticket_id": ticket_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                **context.collected_data
            }
            
            # Send all ticket-related notifications
            notification_result = await self.notification_system.create_ticket_notifications(ticket_data)
            
            logger.info(f"Ticket notifications processed: {notification_result}")
        
        except Exception as e:
            logger.error(f"Ticket notification handling failed: {e}")
    
    async def _handle_escalation_complete(self, user_id: str, conversation_id: str, 
                                        context_data: Dict[str, Any]):
        """Handle completed escalation"""
        escalation_reason = context_data.get("escalation_reason", "User requested escalation")
        self.performance_monitor.record_escalation(conversation_id, escalation_reason)
        
        # Log escalation for follow-up
        logger.info(f"Escalation completed for conversation {conversation_id}: {escalation_reason}")
    
    async def _handle_conversation_complete(self, user_id: str, conversation_id: str):
        """Handle conversation completion"""
        # Clean up context
        self.conversation_engine.context_manager.cleanup_context(conversation_id)
        
        logger.info(f"Conversation {conversation_id} completed successfully")
    
    async def _check_auto_escalation(self, conversation_id: str, result: Dict[str, Any]):
        """Check and handle auto-escalation conditions"""
        if not self.config["auto_escalation_enabled"]:
            return
        
        context = self.conversation_engine.context_manager.get_context(conversation_id)
        if not context:
            return
        
        # Check for escalation conditions
        should_escalate = False
        escalation_reason = None
        
        # Too many errors
        if context.retry_count >= self.config["max_retry_attempts"]:
            should_escalate = True
            escalation_reason = f"Too many retry attempts ({context.retry_count})"
        
        # Conversation taking too long
        elif datetime.now(timezone.utc) - context.last_activity > self.config["escalation_threshold"]:
            should_escalate = True
            escalation_reason = "Conversation duration exceeded threshold"
        
        # Stuck in error recovery
        elif (context.current_state == ConversationState.ERROR_RECOVERY and 
              len(context.errors) >= 3):
            should_escalate = True
            escalation_reason = "Multiple errors in recovery state"
        
        if should_escalate:
            logger.info(f"Auto-escalating conversation {conversation_id}: {escalation_reason}")
            
            # Update context
            context.current_state = ConversationState.ESCALATION
            context.escalation_reason = escalation_reason
            context.priority = Priority.HIGH
            
            # Record escalation
            self.performance_monitor.record_escalation(conversation_id, escalation_reason)
    
    async def _handle_processing_error(self, user_id: str, conversation_id: str, error: str) -> str:
        """Handle message processing errors"""
        self.performance_monitor.record_error(error, conversation_id)
        
        # Get context to determine appropriate error response
        context = self.conversation_engine.context_manager.get_context(conversation_id)
        
        if context and context.retry_count >= self.config["max_retry_attempts"]:
            # Too many retries - escalate
            context.current_state = ConversationState.ESCALATION
            context.escalation_reason = "Technical difficulties"
            return ("I apologize for the persistent technical issues. I'm connecting you "
                   "with a human agent who can assist you immediately.")
        
        # Standard error response
        return ("I apologize for the technical difficulty. Let's try again. "
               "How can I help you with your stolen vehicle report?")
    
    async def _performance_monitoring_task(self):
        """Background task for performance monitoring"""
        while not self.shutdown_event.is_set():
            try:
                # Update external system status
                if hasattr(self.notification_system, 'external_systems'):
                    system_status = self.notification_system.external_systems.get_system_status()
                    self.performance_monitor.metrics["external_system_status"] = system_status
                
                # Clean up old health check data
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                old_checks = [
                    conv_id for conv_id, check in self.performance_monitor.health_checks.items()
                    if check["start_time"] < cutoff_time
                ]
                
                for conv_id in old_checks:
                    del self.performance_monitor.health_checks[conv_id]
                
                # Wait for next monitoring cycle
                await asyncio.sleep(300)  # 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring task error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _notification_processing_task(self):
        """Background task for processing scheduled notifications"""
        while not self.shutdown_event.is_set():
            try:
                await self.notification_system.process_scheduled_notifications()
                await asyncio.sleep(60)  # Check every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Notification processing task error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_task(self):
        """Background task for general cleanup"""
        while not self.shutdown_event.is_set():
            try:
                # Cleanup expired conversation contexts
                self.conversation_engine.context_manager.cleanup_expired_contexts()
                
                # Wait for next cleanup cycle
                await asyncio.sleep(3600)  # 1 hour
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(300)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        health_status = self.performance_monitor.get_health_status()
        
        # Add conversation engine metrics
        engine_metrics = self.conversation_engine.get_performance_metrics()
        
        # Add external systems status
        external_status = {}
        if hasattr(self.notification_system, 'external_systems'):
            external_status = self.notification_system.external_systems.get_system_status()
        
        return {
            "overall_health": health_status,
            "conversation_engine": engine_metrics,
            "external_systems": external_status,
            "configuration": {
                "auto_escalation_enabled": self.config["auto_escalation_enabled"],
                "notification_enabled": self.config["notification_enabled"],
                "performance_monitoring_enabled": self.config["performance_monitoring_enabled"]
            },
            "background_tasks": {
                "active_tasks": len([t for t in self.background_tasks if not t.done()]),
                "total_tasks": len(self.background_tasks)
            }
        }
    
    def get_conversation_details(self, conversation_id: str) -> Dict[str, Any]:
        """Get detailed conversation information"""
        # Get conversation status from engine
        conversation_status = self.conversation_engine.get_conversation_status(conversation_id)
        
        # Get health check data
        health_check = self.performance_monitor.health_checks.get(conversation_id, {})
        
        return {
            "conversation_status": conversation_status,
            "health_check": health_check,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def simulate_vehicle_recovery(self, ticket_id: str, recovery_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate vehicle recovery for testing/demo purposes"""
        if not recovery_details:
            recovery_details = {
                "recovery_location": "Downtown Metro Area",
                "vehicle_condition": "Good condition, minor scratches",
                "recovery_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        
        # Mock ticket data (in real system, this would come from database)
        ticket_data = {
            "ticket_id": ticket_id,
            "owner_name": "John Doe",
            "owner_email": "john.doe@email.com",
            "year": "2020",
            "make": "Toyota",
            "model": "Camry",
            "license_plate": "ABC123"
        }
        
        # Send recovery success notification
        result = await self.notification_system.send_recovery_success_notification(
            ticket_data, recovery_details
        )
        
        return result 