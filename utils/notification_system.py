"""
Notification System for SVL Chatbot
Handles update notifications, external integrations, and communication workflows.
Includes simulated external services for POC demonstration.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, field
import uuid
import random

from utils.logger import get_logger
from utils.database_manager import DatabaseManager

logger = get_logger("notification_system")

class NotificationType(Enum):
    """Types of notifications"""
    STATUS_UPDATE = "status_update"
    SYSTEM_ALERT = "system_alert"
    ESCALATION = "escalation"
    EXTERNAL_UPDATE = "external_update"
    RECOVERY_UPDATE = "recovery_update"
    COMPLETION = "completion"
    ERROR_ALERT = "error_alert"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    EMERGENCY = "emergency"

class DeliveryChannel(Enum):
    """Available delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    INTERNAL = "internal"

@dataclass
class Notification:
    """Notification data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: NotificationType = NotificationType.STATUS_UPDATE
    priority: NotificationPriority = NotificationPriority.MEDIUM
    recipient: str = ""
    subject: str = ""
    message: str = ""
    channels: List[DeliveryChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivery_status: Dict[DeliveryChannel, str] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

class ExternalSystemsManager:
    """Manages integration with external systems (simulated for POC)"""
    
    def __init__(self):
        self.systems = {
            "police_database": PoliceSystemIntegration(),
            "insurance_system": InsuranceSystemIntegration(),
            "recovery_services": RecoveryServicesIntegration(),
            "notification_gateway": NotificationGatewayIntegration()
        }
        self.system_status = {name: "operational" for name in self.systems.keys()}
    
    async def submit_police_report(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit report to police database"""
        try:
            result = await self.systems["police_database"].submit_report(ticket_data)
            return result
        except Exception as e:
            logger.error(f"Police system integration failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def notify_insurance(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Notify insurance company"""
        try:
            result = await self.systems["insurance_system"].notify_claim(ticket_data)
            return result
        except Exception as e:
            logger.error(f"Insurance system integration failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def activate_recovery_services(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Activate vehicle recovery services"""
        try:
            result = await self.systems["recovery_services"].activate_recovery(ticket_data)
            return result
        except Exception as e:
            logger.error(f"Recovery services integration failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def send_notification(self, notification: Notification) -> Dict[str, Any]:
        """Send notification through external gateway"""
        try:
            result = await self.systems["notification_gateway"].send_notification(notification)
            return result
        except Exception as e:
            logger.error(f"Notification gateway failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_system_status(self) -> Dict[str, str]:
        """Get status of all external systems"""
        return self.system_status.copy()

class PoliceSystemIntegration:
    """Simulated police database integration"""
    
    async def submit_report(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit stolen vehicle report to police database"""
        # Simulate API call delay
        await asyncio.sleep(random.uniform(1, 3))
        
        # Simulate success/failure
        if random.random() > 0.05:  # 95% success rate
            case_number = f"PD{datetime.now().strftime('%Y%m%d')}{random.randint(1000, 9999)}"
            return {
                "status": "success",
                "case_number": case_number,
                "agency": "Metro Police Department",
                "assigned_officer": f"Officer {random.choice(['Johnson', 'Smith', 'Williams', 'Brown'])}",
                "priority_level": "Standard",
                "estimated_response": "24-48 hours"
            }
        else:
            return {
                "status": "error",
                "error": "Police database temporarily unavailable",
                "retry_after": 1800  # 30 minutes
            }

class InsuranceSystemIntegration:
    """Simulated insurance system integration"""
    
    async def notify_claim(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Notify insurance company of theft claim"""
        await asyncio.sleep(random.uniform(0.5, 2))
        
        if random.random() > 0.03:  # 97% success rate
            claim_number = f"CLM{datetime.now().strftime('%Y%m%d')}{random.randint(100000, 999999)}"
            return {
                "status": "success",
                "claim_number": claim_number,
                "insurance_company": ticket_data.get("insurance_company", "Unknown"),
                "adjuster_assigned": f"Adjuster {random.choice(['Adams', 'Davis', 'Wilson', 'Moore'])}",
                "next_steps": "Adjuster will contact within 24 hours",
                "coverage_confirmed": True
            }
        else:
            return {
                "status": "error",
                "error": "Insurance system connectivity issue",
                "retry_after": 900  # 15 minutes
            }

class RecoveryServicesIntegration:
    """Simulated vehicle recovery services integration"""
    
    async def activate_recovery(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Activate vehicle recovery services"""
        await asyncio.sleep(random.uniform(1, 2.5))
        
        if random.random() > 0.02:  # 98% success rate
            recovery_id = f"REC{datetime.now().strftime('%Y%m%d')}{random.randint(10000, 99999)}"
            return {
                "status": "activated",
                "recovery_id": recovery_id,
                "service_provider": "SVL Recovery Services",
                "team_assigned": f"Team {random.choice(['Alpha', 'Bravo', 'Charlie', 'Delta'])}",
                "coverage_area": "Metropolitan Area",
                "estimated_deployment": "Within 2 hours",
                "tracking_enabled": True
            }
        else:
            return {
                "status": "error",
                "error": "Recovery services at capacity",
                "retry_after": 3600  # 1 hour
            }

class NotificationGatewayIntegration:
    """Simulated notification gateway for email/SMS"""
    
    async def send_notification(self, notification: Notification) -> Dict[str, Any]:
        """Send notification through external gateway"""
        await asyncio.sleep(random.uniform(0.2, 1))
        
        results = {}
        for channel in notification.channels:
            if random.random() > 0.01:  # 99% success rate per channel
                results[channel.value] = {
                    "status": "delivered",
                    "delivery_id": f"{channel.value.upper()}{random.randint(100000, 999999)}",
                    "delivered_at": datetime.now(timezone.utc).isoformat()
                }
            else:
                results[channel.value] = {
                    "status": "failed",
                    "error": f"{channel.value.capitalize()} service temporarily unavailable",
                    "retry_after": 300  # 5 minutes
                }
        
        return {"status": "processed", "channel_results": results}

class NotificationSystem:
    """Main notification system orchestrator"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.external_systems = ExternalSystemsManager()
        self.notification_queue: List[Notification] = []
        self.notification_templates = self._load_notification_templates()
        self.delivery_handlers = {
            DeliveryChannel.EMAIL: self._send_email,
            DeliveryChannel.SMS: self._send_sms,
            DeliveryChannel.PUSH: self._send_push,
            DeliveryChannel.WEBHOOK: self._send_webhook,
            DeliveryChannel.INTERNAL: self._send_internal
        }
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def _load_notification_templates(self) -> Dict[str, Dict[str, str]]:
        """Load notification templates"""
        return {
            "ticket_created": {
                "subject": "SVL Report Created - Ticket #{ticket_id}",
                "email_body": """
Dear {owner_name},

Your stolen vehicle report has been successfully created.

**Ticket Details:**
- Ticket ID: {ticket_id}
- Vehicle: {year} {make} {model}
- License Plate: {license_plate}
- Report Date: {report_date}

**Next Steps:**
1. Police report will be filed within 24 hours
2. Insurance company has been notified
3. Recovery services have been activated
4. You will receive updates as the investigation progresses

**Important Numbers:**
- SVL Support: 1-800-SVL-HELP
- Emergency Line: 1-800-SVL-911

Thank you for choosing SVL Services.

Best regards,
SVL Support Team
                """,
                "sms_body": "SVL Report #{ticket_id} created for {year} {make} {model}. Police & insurance notified. Updates coming. Call 1-800-SVL-HELP for questions."
            },
            "status_update": {
                "subject": "Update on Your SVL Case - Ticket #{ticket_id}",
                "email_body": """
Dear {owner_name},

We have an update on your stolen vehicle case.

**Update Details:**
{update_message}

**Current Status:** {current_status}
**Next Steps:** {next_steps}

If you have any questions, please contact us at 1-800-SVL-HELP.

Best regards,
SVL Support Team
                """,
                "sms_body": "SVL Update #{ticket_id}: {update_message}. Status: {current_status}. Call 1-800-SVL-HELP for details."
            },
            "recovery_success": {
                "subject": "Great News! Your Vehicle Has Been Recovered - #{ticket_id}",
                "email_body": """
Dear {owner_name},

We have excellent news! Your stolen vehicle has been recovered.

**Recovery Details:**
- Vehicle: {year} {make} {model}
- License Plate: {license_plate}
- Recovery Date: {recovery_date}
- Recovery Location: {recovery_location}
- Vehicle Condition: {vehicle_condition}

**Next Steps:**
1. Police will contact you to arrange vehicle pickup
2. Insurance adjuster will assess any damages
3. Our team will coordinate the handover process

You should expect a call from the police within the next few hours.

Congratulations on the successful recovery!

Best regards,
SVL Recovery Team
                """,
                "sms_body": "GREAT NEWS! Your {year} {make} {model} has been recovered! Police will call you soon. Ticket #{ticket_id}."
            }
        }
    
    async def create_ticket_notifications(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create and send all notifications for a new ticket"""
        results = {}
        
        try:
            # 1. Send confirmation to user
            user_notification = self._create_notification(
                type=NotificationType.STATUS_UPDATE,
                priority=NotificationPriority.HIGH,
                recipient=ticket_data.get("owner_email", ""),
                template="ticket_created",
                template_data=ticket_data,
                channels=[DeliveryChannel.EMAIL, DeliveryChannel.SMS]
            )
            
            user_result = await self._send_notification(user_notification)
            results["user_notification"] = user_result
            
            # 2. Submit to police database
            police_result = await self.external_systems.submit_police_report(ticket_data)
            results["police_submission"] = police_result
            
            if police_result["status"] == "success":
                # Send police confirmation notification
                police_notification = self._create_notification(
                    type=NotificationType.STATUS_UPDATE,
                    priority=NotificationPriority.MEDIUM,
                    recipient=ticket_data.get("owner_email", ""),
                    template="status_update",
                    template_data={
                        **ticket_data,
                        "update_message": f"Police report filed successfully. Case Number: {police_result['case_number']}",
                        "current_status": "Police Investigation Active",
                        "next_steps": "Police will investigate and contact you if needed"
                    },
                    channels=[DeliveryChannel.EMAIL]
                )
                await self._send_notification(police_notification)
            
            # 3. Notify insurance
            insurance_result = await self.external_systems.notify_insurance(ticket_data)
            results["insurance_notification"] = insurance_result
            
            if insurance_result["status"] == "success":
                # Send insurance confirmation notification
                insurance_notification = self._create_notification(
                    type=NotificationType.STATUS_UPDATE,
                    priority=NotificationPriority.MEDIUM,
                    recipient=ticket_data.get("owner_email", ""),
                    template="status_update",
                    template_data={
                        **ticket_data,
                        "update_message": f"Insurance claim initiated. Claim Number: {insurance_result['claim_number']}",
                        "current_status": "Insurance Claim Processing",
                        "next_steps": insurance_result["next_steps"]
                    },
                    channels=[DeliveryChannel.EMAIL]
                )
                await self._send_notification(insurance_notification)
            
            # 4. Activate recovery services
            recovery_result = await self.external_systems.activate_recovery_services(ticket_data)
            results["recovery_activation"] = recovery_result
            
            if recovery_result["status"] == "activated":
                # Send recovery activation notification
                recovery_notification = self._create_notification(
                    type=NotificationType.STATUS_UPDATE,
                    priority=NotificationPriority.HIGH,
                    recipient=ticket_data.get("owner_email", ""),
                    template="status_update",
                    template_data={
                        **ticket_data,
                        "update_message": f"Recovery services activated. Recovery ID: {recovery_result['recovery_id']}",
                        "current_status": "Active Recovery Search",
                        "next_steps": f"Recovery team deployed - {recovery_result['estimated_deployment']}"
                    },
                    channels=[DeliveryChannel.EMAIL, DeliveryChannel.SMS]
                )
                await self._send_notification(recovery_notification)
            
            # 5. Schedule follow-up notifications
            await self._schedule_follow_up_notifications(ticket_data)
            
            return {
                "status": "success",
                "notifications_sent": len([r for r in results.values() if r.get("status") == "success"]),
                "external_integrations": results
            }
        
        except Exception as e:
            logger.error(f"Ticket notification creation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def send_recovery_success_notification(self, ticket_data: Dict[str, Any], recovery_details: Dict[str, Any]) -> Dict[str, Any]:
        """Send vehicle recovery success notification"""
        try:
            notification_data = {
                **ticket_data,
                **recovery_details,
                "recovery_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            
            notification = self._create_notification(
                type=NotificationType.COMPLETION,
                priority=NotificationPriority.URGENT,
                recipient=ticket_data.get("owner_email", ""),
                template="recovery_success",
                template_data=notification_data,
                channels=[DeliveryChannel.EMAIL, DeliveryChannel.SMS, DeliveryChannel.PUSH]
            )
            
            result = await self._send_notification(notification)
            
            # Trigger celebration workflow
            await self._trigger_celebration_workflow(ticket_data, recovery_details)
            
            return result
        
        except Exception as e:
            logger.error(f"Recovery success notification failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _create_notification(self, type: NotificationType, priority: NotificationPriority, 
                           recipient: str, template: str, template_data: Dict[str, Any],
                           channels: List[DeliveryChannel]) -> Notification:
        """Create a notification from template"""
        template_content = self.notification_templates.get(template, {})
        
        subject = template_content.get("subject", "SVL Notification").format(**template_data)
        
        # Choose appropriate message based on primary channel
        if DeliveryChannel.EMAIL in channels:
            message = template_content.get("email_body", "").format(**template_data)
        elif DeliveryChannel.SMS in channels:
            message = template_content.get("sms_body", "").format(**template_data)
        else:
            message = template_content.get("email_body", "").format(**template_data)
        
        return Notification(
            type=type,
            priority=priority,
            recipient=recipient,
            subject=subject,
            message=message,
            channels=channels,
            metadata=template_data
        )
    
    async def _send_notification(self, notification: Notification) -> Dict[str, Any]:
        """Send notification through configured channels"""
        try:
            # Add to queue
            self.notification_queue.append(notification)
            
            # Send through external gateway for email/SMS
            if any(channel in [DeliveryChannel.EMAIL, DeliveryChannel.SMS] for channel in notification.channels):
                external_result = await self.external_systems.send_notification(notification)
                notification.delivery_status.update(external_result.get("channel_results", {}))
            
            # Handle other channels
            for channel in notification.channels:
                if channel not in [DeliveryChannel.EMAIL, DeliveryChannel.SMS]:
                    handler_result = await self.delivery_handlers[channel](notification)
                    notification.delivery_status[channel] = handler_result
            
            notification.sent_at = datetime.now(timezone.utc)
            
            # Notify subscribers
            await self._notify_subscribers("notification_sent", notification)
            
            return {
                "status": "success",
                "notification_id": notification.id,
                "delivery_status": notification.delivery_status
            }
        
        except Exception as e:
            logger.error(f"Notification sending failed: {e}")
            notification.retry_count += 1
            return {"status": "error", "error": str(e)}
    
    async def _schedule_follow_up_notifications(self, ticket_data: Dict[str, Any]):
        """Schedule automated follow-up notifications"""
        # 24-hour follow-up
        follow_up_24h = self._create_notification(
            type=NotificationType.STATUS_UPDATE,
            priority=NotificationPriority.MEDIUM,
            recipient=ticket_data.get("owner_email", ""),
            template="status_update",
            template_data={
                **ticket_data,
                "update_message": "Your case is being actively investigated. Our team is working diligently on your vehicle recovery.",
                "current_status": "Investigation in Progress",
                "next_steps": "We will continue monitoring and update you with any developments"
            },
            channels=[DeliveryChannel.EMAIL]
        )
        follow_up_24h.scheduled_at = datetime.now(timezone.utc) + timedelta(hours=24)
        self.notification_queue.append(follow_up_24h)
        
        # 7-day follow-up
        follow_up_7d = self._create_notification(
            type=NotificationType.STATUS_UPDATE,
            priority=NotificationPriority.MEDIUM,
            recipient=ticket_data.get("owner_email", ""),
            template="status_update",
            template_data={
                **ticket_data,
                "update_message": "Weekly update: Investigation continues. Our recovery specialists are actively searching for your vehicle.",
                "current_status": "Extended Investigation",
                "next_steps": "We recommend contacting your insurance adjuster for claim status updates"
            },
            channels=[DeliveryChannel.EMAIL]
        )
        follow_up_7d.scheduled_at = datetime.now(timezone.utc) + timedelta(days=7)
        self.notification_queue.append(follow_up_7d)
    
    async def _trigger_celebration_workflow(self, ticket_data: Dict[str, Any], recovery_details: Dict[str, Any]):
        """Trigger celebration workflow for successful recovery"""
        # Send internal notifications to team
        team_notification = Notification(
            type=NotificationType.COMPLETION,
            priority=NotificationPriority.HIGH,
            recipient="team@svlservices.com",
            subject=f"🎉 Vehicle Recovery Success - Ticket #{ticket_data['ticket_id']}",
            message=f"Great work team! Vehicle {ticket_data['year']} {ticket_data['make']} {ticket_data['model']} has been successfully recovered.",
            channels=[DeliveryChannel.INTERNAL],
            metadata={"celebration": True, "recovery_details": recovery_details}
        )
        
        await self._send_notification(team_notification)
    
    async def _send_email(self, notification: Notification) -> Dict[str, str]:
        """Handle email delivery (simulated)"""
        await asyncio.sleep(0.5)  # Simulate email sending delay
        return {"status": "delivered", "delivery_id": f"EMAIL{random.randint(100000, 999999)}"}
    
    async def _send_sms(self, notification: Notification) -> Dict[str, str]:
        """Handle SMS delivery (simulated)"""
        await asyncio.sleep(0.3)  # Simulate SMS sending delay
        return {"status": "delivered", "delivery_id": f"SMS{random.randint(100000, 999999)}"}
    
    async def _send_push(self, notification: Notification) -> Dict[str, str]:
        """Handle push notification delivery (simulated)"""
        await asyncio.sleep(0.1)  # Simulate push notification delay
        return {"status": "delivered", "delivery_id": f"PUSH{random.randint(100000, 999999)}"}
    
    async def _send_webhook(self, notification: Notification) -> Dict[str, str]:
        """Handle webhook delivery (simulated)"""
        await asyncio.sleep(0.2)  # Simulate webhook delay
        return {"status": "delivered", "delivery_id": f"WEBHOOK{random.randint(100000, 999999)}"}
    
    async def _send_internal(self, notification: Notification) -> Dict[str, str]:
        """Handle internal system notification"""
        logger.info(f"Internal notification: {notification.subject}")
        return {"status": "delivered", "delivery_id": f"INTERNAL{random.randint(100000, 999999)}"}
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to notification events"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    async def _notify_subscribers(self, event_type: str, data: Any):
        """Notify event subscribers"""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Subscriber notification failed: {e}")
    
    def get_notification_status(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get notification delivery status"""
        for notification in self.notification_queue:
            if notification.id == notification_id:
                return {
                    "id": notification.id,
                    "type": notification.type.value,
                    "priority": notification.priority.value,
                    "sent_at": notification.sent_at.isoformat() if notification.sent_at else None,
                    "delivery_status": notification.delivery_status,
                    "retry_count": notification.retry_count
                }
        return None
    
    async def process_scheduled_notifications(self):
        """Process scheduled notifications (run as background task)"""
        now = datetime.now(timezone.utc)
        
        scheduled_notifications = [
            n for n in self.notification_queue 
            if n.scheduled_at and n.scheduled_at <= now and not n.sent_at
        ]
        
        for notification in scheduled_notifications:
            await self._send_notification(notification) 