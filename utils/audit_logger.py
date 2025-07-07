"""
Comprehensive Audit Logging System for SVL Chatbot
Advanced logging for compliance, security monitoring, and audit trails
"""

import json
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import queue
import os
from pathlib import Path

from utils.logger import get_logger
from utils.security_core import SecurityUtils, DataEncryption

logger = get_logger("audit_logger")

class LogLevel(Enum):
    """Audit log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"

class EventCategory(Enum):
    """Event categories for audit logging"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    PII_ACCESS = "pii_access"
    SECURITY_VIOLATION = "security_violation"
    SESSION_MANAGEMENT = "session_management"
    SYSTEM_ADMINISTRATION = "system_administration"
    USER_ACTIVITY = "user_activity"
    API_ACCESS = "api_access"
    COMPLIANCE = "compliance"
    ERROR_EVENT = "error_event"

class ComplianceFramework(Enum):
    """Compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"

@dataclass
class AuditEvent:
    """Comprehensive audit event structure"""
    event_id: str
    timestamp: datetime
    event_type: str
    category: EventCategory
    level: LogLevel
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: List[ComplianceFramework] = field(default_factory=list)
    risk_score: float = 0.0
    data_classification: Optional[str] = None
    retention_period: Optional[timedelta] = None
    encrypted_fields: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.event_id:
            self.event_id = self._generate_event_id()
        
        if not self.retention_period:
            self.retention_period = self._get_default_retention()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp_str = self.timestamp.isoformat()
        unique_data = f"{timestamp_str}{self.event_type}{self.user_id or ''}"
        return f"audit_{hashlib.sha256(unique_data.encode()).hexdigest()[:16]}"
    
    def _get_default_retention(self) -> timedelta:
        """Get default retention period based on category"""
        retention_map = {
            EventCategory.AUTHENTICATION: timedelta(days=90),
            EventCategory.AUTHORIZATION: timedelta(days=90),
            EventCategory.DATA_ACCESS: timedelta(days=365),
            EventCategory.PII_ACCESS: timedelta(days=2555),  # 7 years
            EventCategory.SECURITY_VIOLATION: timedelta(days=2555),
            EventCategory.COMPLIANCE: timedelta(days=2555),
            EventCategory.SESSION_MANAGEMENT: timedelta(days=30),
            EventCategory.USER_ACTIVITY: timedelta(days=90),
        }
        return retention_map.get(self.category, timedelta(days=365))

class AuditLogFormatter:
    """Formats audit logs for different outputs"""
    
    @staticmethod
    def to_json(event: AuditEvent, encrypt_sensitive: bool = False) -> str:
        """Convert audit event to JSON format"""
        event_dict = asdict(event)
        
        # Convert datetime to ISO format
        event_dict['timestamp'] = event.timestamp.isoformat()
        event_dict['retention_period'] = str(event.retention_period)
        
        # Convert enums to strings
        event_dict['category'] = event.category.value
        event_dict['level'] = event.level.value
        event_dict['compliance_tags'] = [tag.value for tag in event.compliance_tags]
        
        # Encrypt sensitive fields if requested
        if encrypt_sensitive and event.encrypted_fields:
            encryption = DataEncryption()
            for field in event.encrypted_fields:
                if field in event_dict and event_dict[field]:
                    event_dict[field] = encryption.encrypt(str(event_dict[field]))
        
        return json.dumps(event_dict, default=str, ensure_ascii=False)
    
    @staticmethod
    def to_syslog(event: AuditEvent) -> str:
        """Convert audit event to syslog format (RFC 5424)"""
        # Priority calculation: facility * 8 + severity
        facility = 16  # local0
        severity_map = {
            LogLevel.DEBUG: 7,
            LogLevel.INFO: 6,
            LogLevel.WARNING: 4,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 2,
            LogLevel.SECURITY: 2
        }
        severity = severity_map.get(event.level, 6)
        priority = facility * 8 + severity
        
        # Format: <priority>version timestamp hostname app-name procid msgid structured-data msg
        timestamp = event.timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        hostname = os.getenv('HOSTNAME', 'svl-chatbot')
        app_name = 'svl-audit'
        proc_id = str(os.getpid())
        msg_id = event.event_type
        
        # Structured data
        structured_data = f'[audit eventId="{event.event_id}" category="{event.category.value}" userId="{event.user_id or "-"}" sessionId="{event.session_id or "-"}"]'
        
        message = f"{event.action or event.event_type}: {event.message}"
        
        return f"<{priority}>1 {timestamp} {hostname} {app_name} {proc_id} {msg_id} {structured_data} {message}"
    
    @staticmethod
    def to_csv(event: AuditEvent) -> str:
        """Convert audit event to CSV format"""
        fields = [
            event.event_id,
            event.timestamp.isoformat(),
            event.event_type,
            event.category.value,
            event.level.value,
            event.user_id or "",
            event.session_id or "",
            event.ip_address or "",
            event.resource or "",
            event.action or "",
            event.result or "",
            event.message.replace('"', '""'),  # Escape quotes
            str(event.risk_score),
            "|".join(tag.value for tag in event.compliance_tags)
        ]
        
        return ",".join(f'"{field}"' for field in fields)

class AuditLogStore:
    """Secure audit log storage"""
    
    def __init__(self, base_path: str = "audit_logs", encryption_key: Optional[bytes] = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.encryption = DataEncryption(encryption_key) if encryption_key else None
        self._lock = threading.Lock()
        
        # File rotation settings
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.max_files = 100
        
        # Current file handles
        self._current_files = {}
    
    def store_event(self, event: AuditEvent, format_type: str = "json") -> bool:
        """Store audit event to file"""
        try:
            with self._lock:
                file_path = self._get_file_path(event, format_type)
                
                # Format event
                if format_type == "json":
                    content = AuditLogFormatter.to_json(event, encrypt_sensitive=bool(self.encryption))
                elif format_type == "syslog":
                    content = AuditLogFormatter.to_syslog(event)
                elif format_type == "csv":
                    content = AuditLogFormatter.to_csv(event)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
                
                # Encrypt content if encryption is enabled
                if self.encryption:
                    content = self.encryption.encrypt(content)
                
                # Write to file
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(content + '\n')
                
                # Check for rotation
                self._check_rotation(file_path, format_type)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            return False
    
    def _get_file_path(self, event: AuditEvent, format_type: str) -> Path:
        """Get file path for audit event"""
        date_str = event.timestamp.strftime('%Y-%m-%d')
        category_dir = self.base_path / event.category.value
        category_dir.mkdir(exist_ok=True)
        
        filename = f"audit_{date_str}.{format_type}"
        return category_dir / filename
    
    def _check_rotation(self, file_path: Path, format_type: str):
        """Check if file rotation is needed"""
        if file_path.stat().st_size > self.max_file_size:
            self._rotate_file(file_path, format_type)
    
    def _rotate_file(self, file_path: Path, format_type: str):
        """Rotate audit log file"""
        base_name = file_path.stem
        extension = file_path.suffix
        directory = file_path.parent
        
        # Find next rotation number
        rotation_num = 1
        while True:
            rotated_path = directory / f"{base_name}.{rotation_num}{extension}"
            if not rotated_path.exists():
                break
            rotation_num += 1
        
        # Rename current file
        file_path.rename(rotated_path)
        
        # Clean old rotations
        self._clean_old_rotations(directory, base_name, extension)
    
    def _clean_old_rotations(self, directory: Path, base_name: str, extension: str):
        """Clean old rotation files"""
        pattern = f"{base_name}.*{extension}"
        files = list(directory.glob(pattern))
        
        if len(files) > self.max_files:
            # Sort by modification time and remove oldest
            files.sort(key=lambda x: x.stat().st_mtime)
            for old_file in files[:-self.max_files]:
                old_file.unlink()

class ComplianceTracker:
    """Tracks compliance-related activities"""
    
    def __init__(self):
        self.compliance_rules = {
            ComplianceFramework.GDPR: {
                'data_access_logging': True,
                'consent_tracking': True,
                'right_to_erasure': True,
                'data_portability': True,
                'retention_limits': True
            },
            ComplianceFramework.CCPA: {
                'data_access_logging': True,
                'right_to_know': True,
                'right_to_delete': True,
                'right_to_opt_out': True
            },
            ComplianceFramework.PCI_DSS: {
                'payment_data_protection': True,
                'access_logging': True,
                'network_monitoring': True
            }
        }
    
    def check_compliance_requirements(self, event: AuditEvent) -> List[ComplianceFramework]:
        """Check which compliance frameworks apply to an event"""
        applicable_frameworks = []
        
        # GDPR requirements
        if self._requires_gdpr_logging(event):
            applicable_frameworks.append(ComplianceFramework.GDPR)
        
        # CCPA requirements
        if self._requires_ccpa_logging(event):
            applicable_frameworks.append(ComplianceFramework.CCPA)
        
        # PCI DSS requirements
        if self._requires_pci_logging(event):
            applicable_frameworks.append(ComplianceFramework.PCI_DSS)
        
        return applicable_frameworks
    
    def _requires_gdpr_logging(self, event: AuditEvent) -> bool:
        """Check if event requires GDPR logging"""
        gdpr_categories = [
            EventCategory.PII_ACCESS,
            EventCategory.DATA_ACCESS,
            EventCategory.DATA_MODIFICATION,
            EventCategory.USER_ACTIVITY
        ]
        return event.category in gdpr_categories
    
    def _requires_ccpa_logging(self, event: AuditEvent) -> bool:
        """Check if event requires CCPA logging"""
        ccpa_categories = [
            EventCategory.PII_ACCESS,
            EventCategory.DATA_ACCESS,
            EventCategory.DATA_MODIFICATION
        ]
        return event.category in ccpa_categories
    
    def _requires_pci_logging(self, event: AuditEvent) -> bool:
        """Check if event requires PCI DSS logging"""
        # For SVL chatbot, this would apply if payment data is processed
        payment_keywords = ['payment', 'credit', 'card', 'billing']
        return any(keyword in event.message.lower() for keyword in payment_keywords)

class AuditLogger:
    """Main audit logging system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.store = AuditLogStore(
            base_path=self.config.get('log_path', 'audit_logs'),
            encryption_key=self.config.get('encryption_key')
        )
        self.compliance_tracker = ComplianceTracker()
        
        # Async logging queue
        self.log_queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        
        # Start background logging thread
        self._start_logging_thread()
        
        # Risk scoring weights
        self.risk_weights = {
            EventCategory.SECURITY_VIOLATION: 1.0,
            EventCategory.PII_ACCESS: 0.8,
            EventCategory.AUTHENTICATION: 0.6,
            EventCategory.DATA_MODIFICATION: 0.7,
            EventCategory.AUTHORIZATION: 0.5,
        }
    
    def log_event(self, event_type: str, category: EventCategory, level: LogLevel = LogLevel.INFO,
                  user_id: Optional[str] = None, session_id: Optional[str] = None,
                  ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                  resource: Optional[str] = None, action: Optional[str] = None,
                  result: Optional[str] = None, message: str = "",
                  details: Optional[Dict[str, Any]] = None,
                  data_classification: Optional[str] = None,
                  encrypted_fields: Optional[List[str]] = None) -> str:
        """Log an audit event"""
        
        # Create audit event
        event = AuditEvent(
            event_id="",  # Will be generated
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            category=category,
            level=level,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            message=message,
            details=details or {},
            data_classification=data_classification,
            encrypted_fields=encrypted_fields or []
        )
        
        # Check compliance requirements
        event.compliance_tags = self.compliance_tracker.check_compliance_requirements(event)
        
        # Calculate risk score
        event.risk_score = self._calculate_risk_score(event)
        
        # Queue for async processing
        try:
            self.log_queue.put_nowait(event)
            return event.event_id
        except queue.Full:
            logger.error("Audit log queue is full, dropping event")
            return ""
    
    def log_authentication(self, user_id: str, action: str, result: str,
                          ip_address: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> str:
        """Log authentication event"""
        return self.log_event(
            event_type="authentication",
            category=EventCategory.AUTHENTICATION,
            level=LogLevel.SECURITY if result == "failed" else LogLevel.INFO,
            user_id=user_id,
            ip_address=ip_address,
            action=action,
            result=result,
            message=f"Authentication {action}: {result}",
            details=details
        )
    
    def log_data_access(self, user_id: str, resource: str, action: str,
                       session_id: Optional[str] = None, ip_address: Optional[str] = None,
                       data_classification: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None) -> str:
        """Log data access event"""
        return self.log_event(
            event_type="data_access",
            category=EventCategory.DATA_ACCESS,
            level=LogLevel.INFO,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            resource=resource,
            action=action,
            message=f"Data access: {action} on {resource}",
            data_classification=data_classification,
            details=details
        )
    
    def log_pii_access(self, user_id: str, pii_type: str, action: str, approved: bool,
                      session_id: Optional[str] = None, ip_address: Optional[str] = None,
                      details: Optional[Dict[str, Any]] = None) -> str:
        """Log PII access event"""
        return self.log_event(
            event_type="pii_access",
            category=EventCategory.PII_ACCESS,
            level=LogLevel.SECURITY,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            resource=pii_type,
            action=action,
            result="approved" if approved else "denied",
            message=f"PII access: {action} on {pii_type} ({'approved' if approved else 'denied'})",
            data_classification="sensitive",
            details=details,
            encrypted_fields=["details"]
        )
    
    def log_security_violation(self, event_type: str, user_id: Optional[str], 
                             violation_details: Dict[str, Any],
                             ip_address: Optional[str] = None,
                             session_id: Optional[str] = None) -> str:
        """Log security violation"""
        return self.log_event(
            event_type=event_type,
            category=EventCategory.SECURITY_VIOLATION,
            level=LogLevel.CRITICAL,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            message=f"Security violation: {event_type}",
            details=violation_details,
            data_classification="restricted"
        )
    
    def _calculate_risk_score(self, event: AuditEvent) -> float:
        """Calculate risk score for event"""
        base_score = self.risk_weights.get(event.category, 0.3)
        
        # Adjust based on level
        level_multiplier = {
            LogLevel.DEBUG: 0.1,
            LogLevel.INFO: 0.3,
            LogLevel.WARNING: 0.6,
            LogLevel.ERROR: 0.8,
            LogLevel.CRITICAL: 1.0,
            LogLevel.SECURITY: 1.0
        }
        
        score = base_score * level_multiplier.get(event.level, 0.5)
        
        # Adjust based on result
        if event.result in ['failed', 'denied', 'error']:
            score *= 1.5
        
        # Adjust based on details
        if event.details:
            if 'suspicious' in str(event.details).lower():
                score *= 1.3
            if 'attack' in str(event.details).lower():
                score *= 1.5
        
        return min(score, 1.0)
    
    def _start_logging_thread(self):
        """Start background logging thread"""
        def logging_worker():
            while not self.stop_event.is_set():
                try:
                    # Get event from queue with timeout
                    event = self.log_queue.get(timeout=1.0)
                    
                    # Store event in configured formats
                    for format_type in self.config.get('formats', ['json']):
                        self.store.store_event(event, format_type)
                    
                    self.log_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Audit logging error: {e}")
        
        logging_thread = threading.Thread(target=logging_worker, daemon=True)
        logging_thread.start()
        logger.info("Audit logging thread started")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default audit logging configuration"""
        return {
            'log_path': 'audit_logs',
            'formats': ['json', 'syslog'],
            'encryption_enabled': True,
            'max_queue_size': 1000,
            'retention_days': 2555,  # 7 years default
            'compression_enabled': True
        }
    
    def shutdown(self):
        """Shutdown audit logger"""
        self.stop_event.set()
        
        # Process remaining events
        while not self.log_queue.empty():
            try:
                event = self.log_queue.get_nowait()
                for format_type in self.config.get('formats', ['json']):
                    self.store.store_event(event, format_type)
            except queue.Empty:
                break
        
        logger.info("Audit logger shutdown complete")

# Global audit logger instance
global_audit_logger = None

def get_audit_logger(config: Optional[Dict[str, Any]] = None) -> AuditLogger:
    """Get global audit logger instance"""
    global global_audit_logger
    if global_audit_logger is None:
        global_audit_logger = AuditLogger(config)
    return global_audit_logger

# Convenience functions
def log_authentication(user_id: str, action: str, result: str, **kwargs) -> str:
    """Log authentication event"""
    return get_audit_logger().log_authentication(user_id, action, result, **kwargs)

def log_data_access(user_id: str, resource: str, action: str, **kwargs) -> str:
    """Log data access event"""
    return get_audit_logger().log_data_access(user_id, resource, action, **kwargs)

def log_pii_access(user_id: str, pii_type: str, action: str, approved: bool, **kwargs) -> str:
    """Log PII access event"""
    return get_audit_logger().log_pii_access(user_id, pii_type, action, approved, **kwargs)

def log_security_violation(event_type: str, user_id: Optional[str], 
                          violation_details: Dict[str, Any], **kwargs) -> str:
    """Log security violation"""
    return get_audit_logger().log_security_violation(event_type, user_id, violation_details, **kwargs) 