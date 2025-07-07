"""
Real-time Security Monitoring and Incident Response for SVL Chatbot
Comprehensive security monitoring, threat detection, and automated incident response
"""

import os
import time
import asyncio
import threading
import json
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import queue
from collections import defaultdict, deque
import statistics

from utils.logger import get_logger
from utils.security_core import SecurityLevel, SecurityContext
from utils.audit_logger import get_audit_logger, EventCategory, LogLevel

logger = get_logger("security_monitoring")

class ThreatLevel(Enum):
    """Security threat levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    """Security incident status"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"

class ResponseAction(Enum):
    """Automated response actions"""
    LOG_ONLY = "log_only"
    RATE_LIMIT = "rate_limit"
    TEMPORARY_BLOCK = "temporary_block"
    PERMANENT_BLOCK = "permanent_block"
    ESCALATE = "escalate"
    NOTIFY_ADMIN = "notify_admin"
    SESSION_TERMINATE = "session_terminate"
    IP_BLACKLIST = "ip_blacklist"

@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    user_id: str
    session_id: str
    ip_address: str
    description: str
    details: Dict[str, Any]
    source_component: str
    
    # Risk scoring
    risk_score: float = 0.0
    confidence: float = 0.0
    
    # Context
    user_agent: Optional[str] = None
    request_path: Optional[str] = None
    payload_size: Optional[int] = None
    
    # Correlation
    related_events: List[str] = field(default_factory=list)
    pattern_match: Optional[str] = None

@dataclass
class SecurityIncident:
    """Security incident data structure"""
    incident_id: str
    title: str
    description: str
    threat_level: ThreatLevel
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    
    # Events and timeline
    events: List[SecurityEvent] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    
    # Response
    automated_responses: List[str] = field(default_factory=list)
    manual_actions: List[str] = field(default_factory=list)
    
    # Attribution
    affected_users: Set[str] = field(default_factory=set)
    affected_sessions: Set[str] = field(default_factory=set)
    source_ips: Set[str] = field(default_factory=set)
    
    # Resolution
    resolution_notes: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None

class ThreatDetector:
    """Real-time threat detection engine"""
    
    def __init__(self):
        self.detection_rules = self._load_detection_rules()
        self.pattern_cache = {}
        self.baseline_metrics = {}
        self.anomaly_thresholds = {
            "request_rate": {"warning": 100, "critical": 500},
            "error_rate": {"warning": 0.1, "critical": 0.3},
            "failed_auth_rate": {"warning": 5, "critical": 20},
            "pii_detection_rate": {"warning": 0.05, "critical": 0.2},
            "blocked_content_rate": {"warning": 0.02, "critical": 0.1}
        }
        
        # Sliding window for metrics
        self.metrics_window = deque(maxlen=100)
        self.user_behavior_profiles = defaultdict(lambda: {
            "normal_request_rate": 1.0,
            "normal_session_duration": 30.0,
            "typical_input_length": 50,
            "common_patterns": set()
        })
    
    def _load_detection_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load threat detection rules"""
        return {
            "brute_force_attack": {
                "pattern": "multiple_failed_auth",
                "threshold": 5,
                "window_minutes": 5,
                "threat_level": ThreatLevel.HIGH,
                "response": ResponseAction.TEMPORARY_BLOCK
            },
            "sql_injection_attempt": {
                "pattern": "sql_injection_detected",
                "threshold": 1,
                "window_minutes": 1,
                "threat_level": ThreatLevel.HIGH,
                "response": ResponseAction.PERMANENT_BLOCK
            },
            "xss_attempt": {
                "pattern": "xss_detected",
                "threshold": 1,
                "window_minutes": 1,
                "threat_level": ThreatLevel.HIGH,
                "response": ResponseAction.PERMANENT_BLOCK
            },
            "rate_limit_violation": {
                "pattern": "rate_limit_exceeded",
                "threshold": 3,
                "window_minutes": 10,
                "threat_level": ThreatLevel.MEDIUM,
                "response": ResponseAction.RATE_LIMIT
            },
            "pii_exfiltration_attempt": {
                "pattern": "excessive_pii_requests",
                "threshold": 10,
                "window_minutes": 60,
                "threat_level": ThreatLevel.HIGH,
                "response": ResponseAction.ESCALATE
            },
            "session_hijacking": {
                "pattern": "suspicious_session_activity",
                "threshold": 1,
                "window_minutes": 1,
                "threat_level": ThreatLevel.CRITICAL,
                "response": ResponseAction.SESSION_TERMINATE
            },
            "data_exfiltration": {
                "pattern": "unusual_data_access",
                "threshold": 20,
                "window_minutes": 30,
                "threat_level": ThreatLevel.HIGH,
                "response": ResponseAction.ESCALATE
            },
            "anomalous_behavior": {
                "pattern": "behavioral_anomaly",
                "threshold": 1,
                "window_minutes": 1,
                "threat_level": ThreatLevel.MEDIUM,
                "response": ResponseAction.LOG_ONLY
            }
        }
    
    def analyze_event(self, event: SecurityEvent) -> Optional[str]:
        """Analyze security event for threats"""
        threats_detected = []
        
        # Pattern-based detection
        for rule_name, rule in self.detection_rules.items():
            if self._matches_pattern(event, rule["pattern"]):
                threats_detected.append(rule_name)
                event.pattern_match = rule_name
        
        # Anomaly detection
        anomaly_type = self._detect_anomalies(event)
        if anomaly_type:
            threats_detected.append(anomaly_type)
        
        # Behavioral analysis
        behavior_threat = self._analyze_user_behavior(event)
        if behavior_threat:
            threats_detected.append(behavior_threat)
        
        return threats_detected[0] if threats_detected else None
    
    def _matches_pattern(self, event: SecurityEvent, pattern: str) -> bool:
        """Check if event matches threat pattern"""
        pattern_map = {
            "multiple_failed_auth": event.event_type == "authentication_failure",
            "sql_injection_detected": event.event_type == "input_validation_failure" and "sql" in event.description.lower(),
            "xss_detected": event.event_type == "input_validation_failure" and "xss" in event.description.lower(),
            "rate_limit_exceeded": event.event_type == "rate_limit_violation",
            "excessive_pii_requests": event.event_type == "pii_access" and event.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH],
            "suspicious_session_activity": event.event_type == "session_security_violation",
            "unusual_data_access": event.event_type == "data_access" and event.risk_score > 0.7,
            "behavioral_anomaly": event.risk_score > 0.8
        }
        
        return pattern_map.get(pattern, False)
    
    def _detect_anomalies(self, event: SecurityEvent) -> Optional[str]:
        """Detect statistical anomalies"""
        current_time = datetime.now(timezone.utc)
        
        # Check request rate anomalies
        recent_events = [e for e in self.metrics_window 
                        if (current_time - e["timestamp"]).total_seconds() < 300]  # 5 minutes
        
        request_rate = len(recent_events) / 5.0  # requests per minute
        
        if request_rate > self.anomaly_thresholds["request_rate"]["critical"]:
            return "critical_request_rate_anomaly"
        elif request_rate > self.anomaly_thresholds["request_rate"]["warning"]:
            return "high_request_rate_anomaly"
        
        return None
    
    def _analyze_user_behavior(self, event: SecurityEvent) -> Optional[str]:
        """Analyze user behavior for anomalies"""
        user_profile = self.user_behavior_profiles[event.user_id]
        
        # Check for unusual input patterns
        if hasattr(event, 'input_length'):
            if event.input_length > user_profile["typical_input_length"] * 5:
                return "unusual_input_length"
        
        # Check for rapid successive requests
        if hasattr(event, 'request_interval'):
            if event.request_interval < 0.1:  # Less than 100ms between requests
                return "automated_behavior_detected"
        
        return None

class IncidentManager:
    """Security incident management system"""
    
    def __init__(self):
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.incident_history: List[SecurityIncident] = []
        self.auto_response_enabled = True
        self.escalation_rules = self._load_escalation_rules()
        self._incident_counter = 0
    
    def _load_escalation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load incident escalation rules"""
        return {
            ThreatLevel.CRITICAL.value: {
                "auto_escalate": True,
                "escalate_after_minutes": 0,
                "notify_channels": ["email", "sms", "slack"],
                "required_response_time": 5
            },
            ThreatLevel.HIGH.value: {
                "auto_escalate": True,
                "escalate_after_minutes": 15,
                "notify_channels": ["email", "slack"],
                "required_response_time": 30
            },
            ThreatLevel.MEDIUM.value: {
                "auto_escalate": False,
                "escalate_after_minutes": 60,
                "notify_channels": ["email"],
                "required_response_time": 120
            },
            ThreatLevel.LOW.value: {
                "auto_escalate": False,
                "escalate_after_minutes": 240,
                "notify_channels": ["email"],
                "required_response_time": 480
            }
        }
    
    def create_incident(self, event: SecurityEvent, threat_type: str) -> SecurityIncident:
        """Create new security incident"""
        self._incident_counter += 1
        incident_id = f"SEC-{datetime.now().strftime('%Y%m%d')}-{self._incident_counter:04d}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"{threat_type.replace('_', ' ').title()} - {event.event_type}",
            description=f"Security incident triggered by {threat_type}: {event.description}",
            threat_level=event.threat_level,
            status=IncidentStatus.OPEN,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            events=[event],
            affected_users={event.user_id},
            affected_sessions={event.session_id},
            source_ips={event.ip_address}
        )
        
        # Add to timeline
        incident.timeline.append({
            "timestamp": incident.created_at,
            "action": "incident_created",
            "details": f"Incident created from {threat_type} detection",
            "automated": True
        })
        
        self.active_incidents[incident_id] = incident
        logger.warning(f"Security incident created: {incident_id} - {incident.title}")
        
        return incident
    
    def update_incident(self, incident_id: str, event: SecurityEvent):
        """Update existing incident with new event"""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident.events.append(event)
            incident.updated_at = datetime.now(timezone.utc)
            incident.affected_users.add(event.user_id)
            incident.affected_sessions.add(event.session_id)
            incident.source_ips.add(event.ip_address)
            
            # Update threat level if escalated
            if event.threat_level.value > incident.threat_level.value:
                incident.threat_level = event.threat_level
            
            incident.timeline.append({
                "timestamp": event.timestamp,
                "action": "event_added",
                "details": f"Related event added: {event.event_type}",
                "automated": True
            })
    
    def resolve_incident(self, incident_id: str, resolution_notes: str, resolved_by: str = "system"):
        """Resolve security incident"""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident.status = IncidentStatus.RESOLVED
            incident.resolution_notes = resolution_notes
            incident.resolved_by = resolved_by
            incident.resolved_at = datetime.now(timezone.utc)
            
            incident.timeline.append({
                "timestamp": incident.resolved_at,
                "action": "incident_resolved",
                "details": resolution_notes,
                "automated": resolved_by == "system"
            })
            
            # Move to history
            self.incident_history.append(incident)
            del self.active_incidents[incident_id]
            
            logger.info(f"Security incident resolved: {incident_id}")

class SecurityMonitor:
    """Main security monitoring system"""
    
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.incident_manager = IncidentManager()
        self.audit_logger = get_audit_logger()
        
        # Event processing
        self.event_queue = queue.Queue()
        self.processing_thread = None
        self.monitoring_active = False
        
        # Metrics and statistics
        self.metrics = {
            "events_processed": 0,
            "threats_detected": 0,
            "incidents_created": 0,
            "automated_responses": 0,
            "false_positives": 0,
            "monitoring_uptime": 0.0
        }
        
        # Response handlers
        self.response_handlers = {
            ResponseAction.LOG_ONLY: self._handle_log_only,
            ResponseAction.RATE_LIMIT: self._handle_rate_limit,
            ResponseAction.TEMPORARY_BLOCK: self._handle_temporary_block,
            ResponseAction.PERMANENT_BLOCK: self._handle_permanent_block,
            ResponseAction.ESCALATE: self._handle_escalate,
            ResponseAction.NOTIFY_ADMIN: self._handle_notify_admin,
            ResponseAction.SESSION_TERMINATE: self._handle_session_terminate,
            ResponseAction.IP_BLACKLIST: self._handle_ip_blacklist
        }
        
        # Blocked entities
        self.blocked_users = set()
        self.blocked_ips = set()
        self.rate_limited_users = {}
        self.temporary_blocks = {}
        
        self.start_time = datetime.now(timezone.utc)
    
    def start_monitoring(self):
        """Start real-time security monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.processing_thread = threading.Thread(target=self._process_events_loop, daemon=True)
            self.processing_thread.start()
            logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Security monitoring stopped")
    
    def submit_event(self, event: SecurityEvent):
        """Submit security event for analysis"""
        self.event_queue.put(event)
    
    def _process_events_loop(self):
        """Main event processing loop"""
        while self.monitoring_active:
            try:
                # Process events from queue
                try:
                    event = self.event_queue.get(timeout=1)
                    self._process_event(event)
                    self.metrics["events_processed"] += 1
                except queue.Empty:
                    continue
                
                # Periodic maintenance
                self._perform_maintenance()
                
            except Exception as e:
                logger.error(f"Error in security monitoring loop: {e}")
                time.sleep(1)
    
    def _process_event(self, event: SecurityEvent):
        """Process individual security event"""
        # Analyze for threats
        threat_type = self.threat_detector.analyze_event(event)
        
        if threat_type:
            self.metrics["threats_detected"] += 1
            logger.warning(f"Threat detected: {threat_type} - {event.description}")
            
            # Create or update incident
            incident = self._handle_threat(event, threat_type)
            
            # Execute automated response
            if self.incident_manager.auto_response_enabled:
                self._execute_automated_response(event, threat_type, incident)
        
        # Log event for audit
        self.audit_logger.log_security_violation(
            event_type=event.event_type,
            user_id=event.user_id,
            violation_details={
                "description": event.description,
                "threat_level": event.threat_level.value,
                "session_id": event.session_id,
                "source_component": event.source_component,
                "risk_score": event.risk_score,
                "confidence": event.confidence,
                "details": event.details
            },
            ip_address=event.ip_address,
            session_id=event.session_id
        )
    
    def _handle_threat(self, event: SecurityEvent, threat_type: str) -> SecurityIncident:
        """Handle detected threat"""
        # Check for existing related incidents
        related_incident = self._find_related_incident(event, threat_type)
        
        if related_incident:
            self.incident_manager.update_incident(related_incident.incident_id, event)
            return related_incident
        else:
            incident = self.incident_manager.create_incident(event, threat_type)
            self.metrics["incidents_created"] += 1
            return incident
    
    def _find_related_incident(self, event: SecurityEvent, threat_type: str) -> Optional[SecurityIncident]:
        """Find related active incident"""
        for incident in self.incident_manager.active_incidents.values():
            # Check if same user, IP, or threat type within time window
            if (event.user_id in incident.affected_users or 
                event.ip_address in incident.source_ips or
                threat_type in incident.title.lower()):
                
                time_diff = (event.timestamp - incident.created_at).total_seconds()
                if time_diff < 3600:  # Within 1 hour
                    return incident
        
        return None
    
    def _execute_automated_response(self, event: SecurityEvent, threat_type: str, incident: SecurityIncident):
        """Execute automated response to threat"""
        detection_rule = self.threat_detector.detection_rules.get(threat_type)
        if not detection_rule:
            return
        
        response_action = detection_rule["response"]
        
        # Execute response
        if response_action in self.response_handlers:
            success = self.response_handlers[response_action](event, threat_type, incident)
            
            if success:
                self.metrics["automated_responses"] += 1
                incident.automated_responses.append(f"{response_action.value} executed at {datetime.now(timezone.utc)}")
                
                incident.timeline.append({
                    "timestamp": datetime.now(timezone.utc),
                    "action": "automated_response",
                    "details": f"Executed {response_action.value} in response to {threat_type}",
                    "automated": True
                })
    
    def _handle_log_only(self, event: SecurityEvent, threat_type: str, incident: SecurityIncident) -> bool:
        """Handle log-only response"""
        logger.info(f"Security event logged: {threat_type} - {event.description}")
        return True
    
    def _handle_rate_limit(self, event: SecurityEvent, threat_type: str, incident: SecurityIncident) -> bool:
        """Handle rate limiting response"""
        self.rate_limited_users[event.user_id] = {
            "timestamp": datetime.now(timezone.utc),
            "duration_minutes": 30,
            "reason": threat_type
        }
        logger.warning(f"User rate limited: {event.user_id} for {threat_type}")
        return True
    
    def _handle_temporary_block(self, event: SecurityEvent, threat_type: str, incident: SecurityIncident) -> bool:
        """Handle temporary block response"""
        block_duration = timedelta(hours=1)
        self.temporary_blocks[event.user_id] = {
            "timestamp": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + block_duration,
            "reason": threat_type
        }
        logger.warning(f"User temporarily blocked: {event.user_id} for {threat_type}")
        return True
    
    def _handle_permanent_block(self, event: SecurityEvent, threat_type: str, incident: SecurityIncident) -> bool:
        """Handle permanent block response"""
        self.blocked_users.add(event.user_id)
        self.blocked_ips.add(event.ip_address)
        logger.critical(f"User permanently blocked: {event.user_id} ({event.ip_address}) for {threat_type}")
        return True
    
    def _handle_escalate(self, event: SecurityEvent, threat_type: str, incident: SecurityIncident) -> bool:
        """Handle escalation response"""
        incident.status = IncidentStatus.INVESTIGATING
        # In a real implementation, this would notify security team
        logger.critical(f"Security incident escalated: {incident.incident_id} - {threat_type}")
        return True
    
    def _handle_notify_admin(self, event: SecurityEvent, threat_type: str, incident: SecurityIncident) -> bool:
        """Handle admin notification response"""
        # In a real implementation, this would send notifications
        logger.warning(f"Admin notification sent for incident: {incident.incident_id}")
        return True
    
    def _handle_session_terminate(self, event: SecurityEvent, threat_type: str, incident: SecurityIncident) -> bool:
        """Handle session termination response"""
        # In a real implementation, this would terminate the user's session
        logger.critical(f"Session terminated for user: {event.user_id} due to {threat_type}")
        return True
    
    def _handle_ip_blacklist(self, event: SecurityEvent, threat_type: str, incident: SecurityIncident) -> bool:
        """Handle IP blacklisting response"""
        self.blocked_ips.add(event.ip_address)
        logger.critical(f"IP address blacklisted: {event.ip_address} for {threat_type}")
        return True
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        current_time = datetime.now(timezone.utc)
        
        # Clean up expired temporary blocks
        expired_blocks = [user_id for user_id, block_info in self.temporary_blocks.items()
                         if current_time > block_info["expires_at"]]
        
        for user_id in expired_blocks:
            del self.temporary_blocks[user_id]
            logger.info(f"Temporary block expired for user: {user_id}")
        
        # Clean up expired rate limits
        expired_rate_limits = [user_id for user_id, limit_info in self.rate_limited_users.items()
                              if (current_time - limit_info["timestamp"]).total_seconds() > limit_info["duration_minutes"] * 60]
        
        for user_id in expired_rate_limits:
            del self.rate_limited_users[user_id]
        
        # Update uptime metric
        self.metrics["monitoring_uptime"] = (current_time - self.start_time).total_seconds()
    
    def is_user_blocked(self, user_id: str) -> bool:
        """Check if user is blocked"""
        if user_id in self.blocked_users:
            return True
        
        if user_id in self.temporary_blocks:
            block_info = self.temporary_blocks[user_id]
            if datetime.now(timezone.utc) < block_info["expires_at"]:
                return True
            else:
                # Clean up expired block
                del self.temporary_blocks[user_id]
        
        return False
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        return ip_address in self.blocked_ips
    
    def is_user_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited"""
        if user_id in self.rate_limited_users:
            limit_info = self.rate_limited_users[user_id]
            elapsed = (datetime.now(timezone.utc) - limit_info["timestamp"]).total_seconds()
            return elapsed < limit_info["duration_minutes"] * 60
        return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        current_time = datetime.now(timezone.utc)
        uptime_hours = (current_time - self.start_time).total_seconds() / 3600
        
        return {
            "monitoring_active": self.monitoring_active,
            "uptime_hours": round(uptime_hours, 2),
            "metrics": self.metrics.copy(),
            "active_incidents": len(self.incident_manager.active_incidents),
            "total_incidents": len(self.incident_manager.incident_history) + len(self.incident_manager.active_incidents),
            "blocked_users": len(self.blocked_users),
            "blocked_ips": len(self.blocked_ips),
            "rate_limited_users": len(self.rate_limited_users),
            "temporary_blocks": len(self.temporary_blocks),
            "recent_threats": [
                {
                    "incident_id": incident.incident_id,
                    "title": incident.title,
                    "threat_level": incident.threat_level.value,
                    "status": incident.status.value,
                    "created_at": incident.created_at.isoformat()
                }
                for incident in list(self.incident_manager.active_incidents.values())[-5:]
            ]
        }

# Global security monitor instance
global_security_monitor = None

def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance"""
    global global_security_monitor
    if global_security_monitor is None:
        global_security_monitor = SecurityMonitor()
        global_security_monitor.start_monitoring()
    return global_security_monitor

def create_security_event(event_type: str, description: str, user_id: str, session_id: str,
                         ip_address: str, threat_level: ThreatLevel = ThreatLevel.INFO,
                         details: Optional[Dict[str, Any]] = None,
                         source_component: str = "unknown") -> SecurityEvent:
    """Create a security event"""
    import uuid
    
    return SecurityEvent(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        threat_level=threat_level,
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        description=description,
        details=details or {},
        source_component=source_component
    )

def submit_security_event(event_type: str, description: str, user_id: str, session_id: str,
                         ip_address: str, threat_level: ThreatLevel = ThreatLevel.INFO,
                         details: Optional[Dict[str, Any]] = None,
                         source_component: str = "unknown"):
    """Submit security event to monitoring system"""
    event = create_security_event(
        event_type, description, user_id, session_id, ip_address,
        threat_level, details, source_component
    )
    
    monitor = get_security_monitor()
    monitor.submit_event(event) 