"""
Secure Error Handling for SVL Chatbot
Prevents information leakage while providing useful error information
"""

import sys
import traceback
import re
import uuid
from typing import Dict, List, Optional, Any, Type, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import logging

from utils.logger import get_logger
from utils.audit_logger import get_audit_logger, EventCategory, LogLevel
from utils.security_core import PIIType, detect_and_mask_pii

logger = get_logger("secure_error_handler")

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    SYSTEM = "system"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    SECURITY = "security"
    BUSINESS_LOGIC = "business_logic"
    UNKNOWN = "unknown"

@dataclass
class SecureError:
    """Secure error structure"""
    error_id: str
    error_code: str
    category: ErrorCategory
    severity: ErrorSeverity
    user_message: str
    technical_message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    remediation_steps: List[str] = field(default_factory=list)
    is_retryable: bool = False

@dataclass
class ErrorPattern:
    """Error pattern for classification"""
    pattern: str
    category: ErrorCategory
    severity: ErrorSeverity
    user_message: str
    remediation_steps: List[str] = field(default_factory=list)
    is_retryable: bool = False

class SensitiveDataScrubber:
    """Scrubs sensitive data from error messages and stack traces"""
    
    def __init__(self):
        # Patterns for sensitive data
        self.sensitive_patterns = {
            'password': [
                r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
                r'pwd["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
                r'passwd["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?'
            ],
            'api_key': [
                r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
                r'secret[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
                r'access[_-]?token["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?'
            ],
            'connection_string': [
                r'(postgres|mysql|mongodb)://[^"\'\s]+',
                r'Server\s*=\s*[^;]+;[^"\']*',
                r'Data\s+Source\s*=\s*[^;]+;[^"\']*'
            ],
            'file_path': [
                r'(/[^/\s]*){2,}',  # Unix paths
                r'[A-Z]:\\[^"\'\s]+',  # Windows paths
            ],
            'ip_address': [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ]
        }
        
        # System information patterns
        self.system_patterns = [
            r'File "([^"]+)"',  # Python file paths
            r'at\s+([^\s]+\.[^\s]+)\([^)]*\)',  # Java stack traces
            r'line\s+\d+,\s+in\s+([^\s]+)',  # Python function names
            r'(\w+Error:\s+)',  # Error type prefixes
        ]
    
    def scrub_error_message(self, message: str) -> str:
        """Scrub sensitive data from error message"""
        scrubbed = message
        
        # Remove PII using existing detector
        pii_result = detect_and_mask_pii(message)
        if pii_result.pii_found:
            scrubbed = pii_result.masked_text
        
        # Scrub other sensitive patterns
        for category, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                scrubbed = re.sub(pattern, f'[REDACTED_{category.upper()}]', scrubbed, flags=re.IGNORECASE)
        
        return scrubbed
    
    def scrub_stack_trace(self, stack_trace: str) -> str:
        """Scrub sensitive data from stack trace"""
        if not stack_trace:
            return stack_trace
        
        scrubbed = stack_trace
        
        # Remove file paths but keep relative information
        scrubbed = re.sub(
            r'File "([^"]+)"',
            lambda m: f'File "{self._sanitize_file_path(m.group(1))}"',
            scrubbed
        )
        
        # Remove sensitive data patterns
        scrubbed = self.scrub_error_message(scrubbed)
        
        return scrubbed
    
    def _sanitize_file_path(self, file_path: str) -> str:
        """Sanitize file path to remove sensitive directory information"""
        # Keep only the filename and one parent directory
        parts = file_path.replace('\\', '/').split('/')
        if len(parts) > 2:
            return f".../{parts[-2]}/{parts[-1]}"
        return file_path

class ErrorClassifier:
    """Classifies errors into categories and severity levels"""
    
    def __init__(self):
        self.error_patterns = [
            # Validation errors
            ErrorPattern(
                pattern=r'(validation|invalid|missing|required|format).*error',
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW,
                user_message="Please check your input and try again.",
                remediation_steps=["Verify all required fields are filled", "Check data format"],
                is_retryable=True
            ),
            
            # Authentication errors
            ErrorPattern(
                pattern=r'(auth|login|credential|unauthorized).*error',
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.MEDIUM,
                user_message="Authentication failed. Please check your credentials.",
                remediation_steps=["Verify your credentials", "Try logging in again"],
                is_retryable=True
            ),
            
            # Authorization errors
            ErrorPattern(
                pattern=r'(permission|forbidden|access.*denied|not.*authorized)',
                category=ErrorCategory.AUTHORIZATION,
                severity=ErrorSeverity.MEDIUM,
                user_message="You don't have permission to perform this action.",
                remediation_steps=["Contact an administrator for access"]
            ),
            
            # Rate limit errors
            ErrorPattern(
                pattern=r'(rate.*limit|too.*many.*requests|throttl)',
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.MEDIUM,
                user_message="Too many requests. Please try again later.",
                remediation_steps=["Wait a few minutes before trying again"],
                is_retryable=True
            ),
            
            # Database errors
            ErrorPattern(
                pattern=r'(database|connection|sql|query).*error',
                category=ErrorCategory.DATABASE,
                severity=ErrorSeverity.HIGH,
                user_message="A database error occurred. Please try again later.",
                remediation_steps=["Try again in a few minutes", "Contact support if the problem persists"]
            ),
            
            # External API errors
            ErrorPattern(
                pattern=r'(api|service|endpoint|network|timeout).*error',
                category=ErrorCategory.EXTERNAL_API,
                severity=ErrorSeverity.MEDIUM,
                user_message="An external service is currently unavailable. Please try again later.",
                remediation_steps=["Try again in a few minutes"],
                is_retryable=True
            ),
            
            # Security errors
            ErrorPattern(
                pattern=r'(security|attack|malicious|suspicious|blocked)',
                category=ErrorCategory.SECURITY,
                severity=ErrorSeverity.CRITICAL,
                user_message="A security issue was detected. This incident has been logged.",
                remediation_steps=["Contact support if you believe this is an error"]
            ),
            
            # System errors
            ErrorPattern(
                pattern=r'(system|internal|server|runtime).*error',
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.HIGH,
                user_message="An internal system error occurred. Please try again later.",
                remediation_steps=["Try again in a few minutes", "Contact support if the problem persists"]
            )
        ]
    
    def classify_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorPattern:
        """Classify error into category and severity"""
        error_text = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check patterns
        for pattern in self.error_patterns:
            if re.search(pattern.pattern, error_text, re.IGNORECASE) or \
               re.search(pattern.pattern, error_type, re.IGNORECASE):
                return pattern
        
        # Default classification
        return ErrorPattern(
            pattern="unknown",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            user_message="An unexpected error occurred. Please try again later.",
            remediation_steps=["Try again", "Contact support if the problem persists"]
        )

class SecureErrorHandler:
    """Main secure error handling system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.scrubber = SensitiveDataScrubber()
        self.classifier = ErrorClassifier()
        
        # Error message templates
        self.user_message_templates = {
            ErrorCategory.VALIDATION: "Please check your input and try again.",
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please verify your credentials.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to perform this action.",
            ErrorCategory.RATE_LIMIT: "Too many requests. Please try again later.",
            ErrorCategory.SYSTEM: "A system error occurred. Please try again later.",
            ErrorCategory.DATABASE: "A database error occurred. Please try again later.",
            ErrorCategory.EXTERNAL_API: "An external service is temporarily unavailable.",
            ErrorCategory.SECURITY: "A security issue was detected. This incident has been logged.",
            ErrorCategory.BUSINESS_LOGIC: "The requested operation cannot be completed.",
            ErrorCategory.UNKNOWN: "An unexpected error occurred. Please try again later."
        }
    
    def handle_error(self, error: Exception, 
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    request_id: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> SecureError:
        """Handle error securely"""
        try:
            # Generate error ID
            error_id = str(uuid.uuid4())
            
            # Classify error
            pattern = self.classifier.classify_error(error, context)
            
            # Get stack trace if configured
            stack_trace = None
            if self.config.get('include_stack_trace', False):
                stack_trace = self.scrubber.scrub_stack_trace(traceback.format_exc())
            
            # Create secure error
            secure_error = SecureError(
                error_id=error_id,
                error_code=f"{pattern.category.value}_{pattern.severity.value}",
                category=pattern.category,
                severity=pattern.severity,
                user_message=pattern.user_message,
                technical_message=self.scrubber.scrub_error_message(str(error)),
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                stack_trace=stack_trace,
                context=self._scrub_context(context or {}),
                remediation_steps=pattern.remediation_steps,
                is_retryable=pattern.is_retryable
            )
            
            # Log error
            self._log_error(secure_error, error)
            
            # Send alerts for critical errors
            if pattern.severity == ErrorSeverity.CRITICAL:
                self._send_alert(secure_error)
            
            return secure_error
            
        except Exception as handler_error:
            # Fallback error handling
            logger.error(f"Error handler failed: {handler_error}")
            return self._create_fallback_error(error, error_id if 'error_id' in locals() else str(uuid.uuid4()))
    
    def handle_security_error(self, error: Exception, 
                             security_context: Dict[str, Any],
                             user_id: Optional[str] = None,
                             session_id: Optional[str] = None) -> SecureError:
        """Handle security-related errors with special logging"""
        secure_error = self.handle_error(error, user_id, session_id, security_context)
        
        # Additional security logging
        get_audit_logger().log_security_violation(
            event_type="security_error",
            user_id=user_id,
            violation_details={
                "error_id": secure_error.error_id,
                "error_type": type(error).__name__,
                "context": security_context,
                "severity": secure_error.severity.value
            },
            ip_address=security_context.get('ip_address'),
            session_id=session_id
        )
        
        return secure_error
    
    def create_user_error_response(self, secure_error: SecureError) -> Dict[str, Any]:
        """Create error response safe for users"""
        response = {
            "error": True,
            "error_id": secure_error.error_id,
            "message": secure_error.user_message,
            "timestamp": secure_error.timestamp.isoformat(),
            "retryable": secure_error.is_retryable
        }
        
        # Add remediation steps if configured
        if self.config.get('include_remediation_steps', True):
            response["remediation"] = secure_error.remediation_steps
        
        # Add error code for API responses
        if self.config.get('include_error_codes', True):
            response["error_code"] = secure_error.error_code
        
        return response
    
    def create_technical_error_response(self, secure_error: SecureError) -> Dict[str, Any]:
        """Create detailed error response for technical users"""
        response = self.create_user_error_response(secure_error)
        
        # Add technical details
        response.update({
            "technical_message": secure_error.technical_message,
            "category": secure_error.category.value,
            "severity": secure_error.severity.value,
            "context": secure_error.context
        })
        
        # Add stack trace if available and configured
        if secure_error.stack_trace and self.config.get('include_stack_trace_in_response', False):
            response["stack_trace"] = secure_error.stack_trace
        
        return response
    
    def _scrub_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Scrub sensitive data from context"""
        scrubbed_context = {}
        
        for key, value in context.items():
            if isinstance(value, str):
                scrubbed_context[key] = self.scrubber.scrub_error_message(value)
            elif isinstance(value, dict):
                scrubbed_context[key] = self._scrub_context(value)
            elif isinstance(value, list):
                scrubbed_context[key] = [
                    self.scrubber.scrub_error_message(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                scrubbed_context[key] = value
        
        return scrubbed_context
    
    def _log_error(self, secure_error: SecureError, original_error: Exception):
        """Log error to audit system"""
        # Determine log level based on severity
        log_level_map = {
            ErrorSeverity.LOW: LogLevel.INFO,
            ErrorSeverity.MEDIUM: LogLevel.WARNING,
            ErrorSeverity.HIGH: LogLevel.ERROR,
            ErrorSeverity.CRITICAL: LogLevel.CRITICAL
        }
        
        log_level = log_level_map.get(secure_error.severity, LogLevel.ERROR)
        
        # Log to audit system
        get_audit_logger().log_event(
            event_type="error_occurred",
            category=EventCategory.ERROR_EVENT,
            level=log_level,
            user_id=secure_error.user_id,
            session_id=secure_error.session_id,
            action="handle_error",
            resource=secure_error.category.value,
            message=f"Error handled: {secure_error.error_code}",
            details={
                "error_id": secure_error.error_id,
                "error_type": type(original_error).__name__,
                "technical_message": secure_error.technical_message,
                "context": secure_error.context,
                "stack_trace": secure_error.stack_trace,
                "is_retryable": secure_error.is_retryable
            }
        )
        
        # Log to application logger
        logger.error(
            f"Error {secure_error.error_id}: {secure_error.technical_message}",
            extra={
                "error_id": secure_error.error_id,
                "user_id": secure_error.user_id,
                "session_id": secure_error.session_id,
                "category": secure_error.category.value,
                "severity": secure_error.severity.value
            }
        )
    
    def _send_alert(self, secure_error: SecureError):
        """Send alert for critical errors"""
        # In a real implementation, this would send alerts via email, Slack, etc.
        logger.critical(
            f"CRITICAL ERROR ALERT - Error ID: {secure_error.error_id}",
            extra={
                "error_id": secure_error.error_id,
                "user_id": secure_error.user_id,
                "category": secure_error.category.value,
                "message": secure_error.technical_message
            }
        )
    
    def _create_fallback_error(self, original_error: Exception, error_id: str) -> SecureError:
        """Create fallback error when error handling fails"""
        return SecureError(
            error_id=error_id,
            error_code="system_critical",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            user_message="A system error occurred. Please contact support.",
            technical_message="Error handler failure",
            remediation_steps=["Contact support immediately"]
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'include_stack_trace': False,
            'include_stack_trace_in_response': False,
            'include_remediation_steps': True,
            'include_error_codes': True,
            'log_all_errors': True,
            'alert_on_critical': True
        }

class ErrorHandlerMiddleware:
    """Middleware for automatic error handling"""
    
    def __init__(self, error_handler: SecureErrorHandler):
        self.error_handler = error_handler
    
    def __call__(self, func):
        """Decorator for automatic error handling"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Extract context from arguments
                user_id = kwargs.get('user_id')
                session_id = kwargs.get('session_id')
                request_id = kwargs.get('request_id')
                
                # Handle error
                secure_error = self.error_handler.handle_error(
                    e, user_id, session_id, request_id
                )
                
                # Return user-safe error response
                raise SecureErrorException(secure_error)
        
        return wrapper

class SecureErrorException(Exception):
    """Exception that contains secure error information"""
    
    def __init__(self, secure_error: SecureError):
        self.secure_error = secure_error
        super().__init__(secure_error.user_message)

# Global error handler instance
global_error_handler = None

def get_error_handler(config: Optional[Dict[str, Any]] = None) -> SecureErrorHandler:
    """Get global error handler"""
    global global_error_handler
    if global_error_handler is None:
        global_error_handler = SecureErrorHandler(config)
    return global_error_handler

# Convenience functions
def handle_error(error: Exception, **kwargs) -> SecureError:
    """Handle error securely"""
    return get_error_handler().handle_error(error, **kwargs)

def handle_security_error(error: Exception, security_context: Dict[str, Any], **kwargs) -> SecureError:
    """Handle security error"""
    return get_error_handler().handle_security_error(error, security_context, **kwargs)

def secure_error_response(secure_error: SecureError, include_technical: bool = False) -> Dict[str, Any]:
    """Create secure error response"""
    if include_technical:
        return get_error_handler().create_technical_error_response(secure_error)
    else:
        return get_error_handler().create_user_error_response(secure_error)

# Decorator for automatic error handling
def with_error_handling(func):
    """Decorator for automatic secure error handling"""
    return ErrorHandlerMiddleware(get_error_handler())(func) 