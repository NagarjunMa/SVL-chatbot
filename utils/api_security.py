"""
API Security and Authentication for SVL Chatbot
Comprehensive API security, authentication, authorization, and key management
"""

import os
import jwt
import hmac
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from functools import wraps
import base64

from utils.logger import get_logger
from utils.security_core import SecurityUtils, DataEncryption, SecurityLevel
from utils.session_security import SessionSecurityLevel
from utils.audit_logger import get_audit_logger, EventCategory, LogLevel

logger = get_logger("api_security")

class AuthMethod(Enum):
    """Authentication methods"""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    SESSION_TOKEN = "session_token"
    HMAC_SIGNATURE = "hmac_signature"
    OAUTH2 = "oauth2"

class Permission(Enum):
    """API permissions"""
    READ_USER_DATA = "read_user_data"
    WRITE_USER_DATA = "write_user_data"
    DELETE_USER_DATA = "delete_user_data"
    READ_VEHICLE_DATA = "read_vehicle_data"
    WRITE_VEHICLE_DATA = "write_vehicle_data"
    CREATE_TICKETS = "create_tickets"
    READ_TICKETS = "read_tickets"
    UPDATE_TICKETS = "update_tickets"
    DELETE_TICKETS = "delete_tickets"
    ADMIN_ACCESS = "admin_access"
    SYSTEM_MONITOR = "system_monitor"
    AUDIT_LOG_ACCESS = "audit_log_access"

class APIKeyType(Enum):
    """Types of API keys"""
    PUBLIC = "public"           # Read-only access
    PRIVATE = "private"         # Full access
    RESTRICTED = "restricted"   # Limited scope
    INTERNAL = "internal"       # Internal services
    WEBHOOK = "webhook"         # Webhook endpoints

@dataclass
class APIKey:
    """API key structure"""
    key_id: str
    key_hash: str
    key_type: APIKeyType
    user_id: Optional[str] = None
    name: str = ""
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit: Optional[int] = None
    allowed_ips: List[str] = field(default_factory=list)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class JWTClaims:
    """JWT token claims"""
    user_id: str
    session_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1))
    issuer: str = "svl-chatbot"
    audience: str = "svl-api"
    scope: List[str] = field(default_factory=list)

@dataclass
class AuthenticationResult:
    """Result of authentication"""
    is_authenticated: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    auth_method: Optional[AuthMethod] = None
    api_key_id: Optional[str] = None
    error_message: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    rate_limit: Optional[int] = None

class SecureKeyManager:
    """Secure API key management"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.api_keys: Dict[str, APIKey] = {}
        self.key_hashes: Dict[str, str] = {}  # hash -> key_id mapping
        self.encryption = DataEncryption(encryption_key)
        self._lock = threading.Lock()
        
        # JWT configuration
        self.jwt_secret = os.getenv('JWT_SECRET_KEY') or SecurityUtils.generate_secure_token(64)
        self.jwt_algorithm = 'HS256'
        self.jwt_issuer = 'svl-chatbot'
        
        # Default permissions by key type
        self.default_permissions = {
            APIKeyType.PUBLIC: {
                Permission.READ_VEHICLE_DATA,
                Permission.READ_TICKETS
            },
            APIKeyType.PRIVATE: {
                Permission.READ_USER_DATA,
                Permission.WRITE_USER_DATA,
                Permission.READ_VEHICLE_DATA,
                Permission.WRITE_VEHICLE_DATA,
                Permission.CREATE_TICKETS,
                Permission.READ_TICKETS,
                Permission.UPDATE_TICKETS
            },
            APIKeyType.RESTRICTED: {
                Permission.READ_VEHICLE_DATA,
                Permission.CREATE_TICKETS
            },
            APIKeyType.INTERNAL: {
                Permission.READ_USER_DATA,
                Permission.WRITE_USER_DATA,
                Permission.READ_VEHICLE_DATA,
                Permission.WRITE_VEHICLE_DATA,
                Permission.CREATE_TICKETS,
                Permission.READ_TICKETS,
                Permission.UPDATE_TICKETS,
                Permission.SYSTEM_MONITOR
            },
            APIKeyType.WEBHOOK: {
                Permission.READ_TICKETS,
                Permission.UPDATE_TICKETS
            }
        }
    
    def generate_api_key(self, key_type: APIKeyType, user_id: Optional[str] = None,
                        name: str = "", custom_permissions: Optional[Set[Permission]] = None,
                        expires_in: Optional[timedelta] = None,
                        allowed_ips: Optional[List[str]] = None) -> Tuple[str, str]:
        """Generate new API key"""
        with self._lock:
            # Generate key components
            key_id = f"svl_{key_type.value}_{SecurityUtils.generate_secure_token(16)}"
            api_key = SecurityUtils.generate_secure_token(32)
            full_key = f"{key_id}.{api_key}"
            
            # Hash the key for storage
            key_hash = hashlib.sha256(full_key.encode()).hexdigest()
            
            # Set permissions
            permissions = custom_permissions or self.default_permissions.get(key_type, set())
            
            # Set expiration
            expires_at = None
            if expires_in:
                expires_at = datetime.now(timezone.utc) + expires_in
            elif key_type == APIKeyType.PUBLIC:
                expires_at = datetime.now(timezone.utc) + timedelta(days=90)
            
            # Create API key record
            api_key_record = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                key_type=key_type,
                user_id=user_id,
                name=name,
                permissions=permissions,
                expires_at=expires_at,
                allowed_ips=allowed_ips or [],
                rate_limit=self._get_default_rate_limit(key_type)
            )
            
            # Store key
            self.api_keys[key_id] = api_key_record
            self.key_hashes[key_hash] = key_id
            
            # Log key creation
            get_audit_logger().log_event(
                event_type="api_key_created",
                category=EventCategory.AUTHENTICATION,
                level=LogLevel.INFO,
                user_id=user_id,
                action="create_api_key",
                resource=key_type.value,
                message=f"API key created: {key_type.value}",
                details={
                    "key_id": key_id,
                    "permissions": [p.value for p in permissions],
                    "expires_at": expires_at.isoformat() if expires_at else None
                }
            )
            
            return key_id, full_key
    
    def authenticate_api_key(self, api_key: str, ip_address: Optional[str] = None) -> AuthenticationResult:
        """Authenticate API key"""
        try:
            # Parse key
            if '.' not in api_key:
                return AuthenticationResult(
                    is_authenticated=False,
                    error_message="Invalid API key format"
                )
            
            key_id, key_secret = api_key.split('.', 1)
            full_key = f"{key_id}.{key_secret}"
            key_hash = hashlib.sha256(full_key.encode()).hexdigest()
            
            # Find key record
            if key_hash not in self.key_hashes:
                return AuthenticationResult(
                    is_authenticated=False,
                    error_message="Invalid API key"
                )
            
            stored_key_id = self.key_hashes[key_hash]
            api_key_record = self.api_keys.get(stored_key_id)
            
            if not api_key_record:
                return AuthenticationResult(
                    is_authenticated=False,
                    error_message="API key not found"
                )
            
            # Validate key
            validation_result = self._validate_api_key(api_key_record, ip_address)
            if not validation_result.is_authenticated:
                return validation_result
            
            # Update usage
            with self._lock:
                api_key_record.last_used = datetime.now(timezone.utc)
                api_key_record.usage_count += 1
            
            # Log successful authentication
            get_audit_logger().log_authentication(
                user_id=api_key_record.user_id or "api_key_user",
                action="api_key_auth",
                result="success",
                ip_address=ip_address,
                details={"key_id": key_id, "key_type": api_key_record.key_type.value}
            )
            
            return AuthenticationResult(
                is_authenticated=True,
                user_id=api_key_record.user_id,
                permissions=api_key_record.permissions,
                auth_method=AuthMethod.API_KEY,
                api_key_id=key_id,
                rate_limit=api_key_record.rate_limit,
                security_level=self._get_security_level(api_key_record.key_type)
            )
            
        except Exception as e:
            logger.error(f"API key authentication error: {e}")
            return AuthenticationResult(
                is_authenticated=False,
                error_message="Authentication failed"
            )
    
    def revoke_api_key(self, key_id: str, reason: str = "Revoked by admin") -> bool:
        """Revoke API key"""
        with self._lock:
            api_key_record = self.api_keys.get(key_id)
            if not api_key_record:
                return False
            
            api_key_record.is_active = False
            
            # Remove from hash mapping
            if api_key_record.key_hash in self.key_hashes:
                del self.key_hashes[api_key_record.key_hash]
            
            # Log revocation
            get_audit_logger().log_event(
                event_type="api_key_revoked",
                category=EventCategory.AUTHENTICATION,
                level=LogLevel.WARNING,
                user_id=api_key_record.user_id,
                action="revoke_api_key",
                resource=key_id,
                message=f"API key revoked: {reason}",
                details={"key_id": key_id, "reason": reason}
            )
            
            return True
    
    def generate_jwt_token(self, claims: JWTClaims) -> str:
        """Generate JWT token"""
        payload = {
            'sub': claims.user_id,
            'iat': int(claims.issued_at.timestamp()),
            'exp': int(claims.expires_at.timestamp()),
            'iss': claims.issuer,
            'aud': claims.audience,
            'permissions': claims.permissions,
            'scope': claims.scope
        }
        
        if claims.session_id:
            payload['sid'] = claims.session_id
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Log token generation
        get_audit_logger().log_event(
            event_type="jwt_token_generated",
            category=EventCategory.AUTHENTICATION,
            level=LogLevel.INFO,
            user_id=claims.user_id,
            action="generate_jwt",
            message="JWT token generated",
            details={
                "permissions": claims.permissions,
                "expires_at": claims.expires_at.isoformat()
            }
        )
        
        return token
    
    def authenticate_jwt_token(self, token: str) -> AuthenticationResult:
        """Authenticate JWT token"""
        try:
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                issuer=self.jwt_issuer
            )
            
            # Extract claims
            user_id = payload.get('sub')
            session_id = payload.get('sid')
            permissions = set(Permission(p) for p in payload.get('permissions', []))
            
            # Log successful authentication
            get_audit_logger().log_authentication(
                user_id=user_id,
                action="jwt_auth",
                result="success",
                details={"session_id": session_id}
            )
            
            return AuthenticationResult(
                is_authenticated=True,
                user_id=user_id,
                session_id=session_id,
                permissions=permissions,
                auth_method=AuthMethod.JWT_TOKEN,
                security_level=SecurityLevel.HIGH
            )
            
        except jwt.ExpiredSignatureError:
            return AuthenticationResult(
                is_authenticated=False,
                error_message="Token expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return AuthenticationResult(
                is_authenticated=False,
                error_message="Invalid token"
            )
        except Exception as e:
            logger.error(f"JWT authentication error: {e}")
            return AuthenticationResult(
                is_authenticated=False,
                error_message="Authentication failed"
            )
    
    def _validate_api_key(self, api_key_record: APIKey, ip_address: Optional[str]) -> AuthenticationResult:
        """Validate API key constraints"""
        # Check if key is active
        if not api_key_record.is_active:
            return AuthenticationResult(
                is_authenticated=False,
                error_message="API key is inactive"
            )
        
        # Check expiration
        if api_key_record.expires_at and datetime.now(timezone.utc) > api_key_record.expires_at:
            return AuthenticationResult(
                is_authenticated=False,
                error_message="API key expired"
            )
        
        # Check IP restrictions
        if api_key_record.allowed_ips and ip_address:
            if ip_address not in api_key_record.allowed_ips:
                return AuthenticationResult(
                    is_authenticated=False,
                    error_message="IP address not allowed"
                )
        
        return AuthenticationResult(is_authenticated=True)
    
    def _get_default_rate_limit(self, key_type: APIKeyType) -> int:
        """Get default rate limit for key type"""
        rate_limits = {
            APIKeyType.PUBLIC: 100,      # requests per hour
            APIKeyType.PRIVATE: 1000,
            APIKeyType.RESTRICTED: 50,
            APIKeyType.INTERNAL: 10000,
            APIKeyType.WEBHOOK: 500
        }
        return rate_limits.get(key_type, 100)
    
    def _get_security_level(self, key_type: APIKeyType) -> SecurityLevel:
        """Get security level for key type"""
        security_levels = {
            APIKeyType.PUBLIC: SecurityLevel.LOW,
            APIKeyType.PRIVATE: SecurityLevel.HIGH,
            APIKeyType.RESTRICTED: SecurityLevel.MEDIUM,
            APIKeyType.INTERNAL: SecurityLevel.CRITICAL,
            APIKeyType.WEBHOOK: SecurityLevel.MEDIUM
        }
        return security_levels.get(key_type, SecurityLevel.MEDIUM)

class HMACSignatureValidator:
    """HMAC signature validation for webhooks"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
    
    def generate_signature(self, payload: bytes, timestamp: str) -> str:
        """Generate HMAC signature"""
        message = f"{timestamp}.{payload.decode('utf-8')}"
        signature = hmac.new(
            self.secret_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def validate_signature(self, payload: bytes, signature: str, 
                          timestamp: str, tolerance: int = 300) -> bool:
        """Validate HMAC signature"""
        try:
            # Check timestamp tolerance (default 5 minutes)
            current_time = int(time.time())
            request_time = int(timestamp)
            
            if abs(current_time - request_time) > tolerance:
                return False
            
            # Generate expected signature
            expected_signature = self.generate_signature(payload, timestamp)
            
            # Compare signatures
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"HMAC signature validation error: {e}")
            return False

class APIAuthenticator:
    """Main API authentication system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.key_manager = SecureKeyManager(
            encryption_key=self.config.get('encryption_key')
        )
        self.hmac_validators: Dict[str, HMACSignatureValidator] = {}
        
        # Initialize HMAC validators for webhooks
        webhook_secrets = self.config.get('webhook_secrets', {})
        for webhook_id, secret in webhook_secrets.items():
            self.hmac_validators[webhook_id] = HMACSignatureValidator(secret)
    
    def authenticate_request(self, auth_header: Optional[str] = None,
                           api_key: Optional[str] = None,
                           jwt_token: Optional[str] = None,
                           signature: Optional[str] = None,
                           timestamp: Optional[str] = None,
                           payload: Optional[bytes] = None,
                           ip_address: Optional[str] = None) -> AuthenticationResult:
        """Authenticate API request using various methods"""
        
        # Try API key authentication
        if api_key:
            return self.key_manager.authenticate_api_key(api_key, ip_address)
        
        # Try JWT token authentication
        if jwt_token:
            return self.key_manager.authenticate_jwt_token(jwt_token)
        
        # Parse Authorization header
        if auth_header:
            auth_result = self._parse_auth_header(auth_header, ip_address)
            if auth_result.is_authenticated:
                return auth_result
        
        # Try HMAC signature authentication
        if signature and timestamp and payload:
            return self._authenticate_hmac_signature(signature, timestamp, payload)
        
        return AuthenticationResult(
            is_authenticated=False,
            error_message="No valid authentication method provided"
        )
    
    def authorize_request(self, auth_result: AuthenticationResult, 
                         required_permission: Permission) -> bool:
        """Authorize request based on permissions"""
        if not auth_result.is_authenticated:
            return False
        
        has_permission = required_permission in auth_result.permissions
        
        # Log authorization attempt
        get_audit_logger().log_event(
            event_type="authorization_check",
            category=EventCategory.AUTHORIZATION,
            level=LogLevel.INFO if has_permission else LogLevel.WARNING,
            user_id=auth_result.user_id,
            action="check_permission",
            resource=required_permission.value,
            result="granted" if has_permission else "denied",
            message=f"Authorization check for {required_permission.value}: {'granted' if has_permission else 'denied'}",
            details={
                "auth_method": auth_result.auth_method.value if auth_result.auth_method else None,
                "user_permissions": [p.value for p in auth_result.permissions]
            }
        )
        
        return has_permission
    
    def _parse_auth_header(self, auth_header: str, ip_address: Optional[str]) -> AuthenticationResult:
        """Parse Authorization header"""
        parts = auth_header.split(' ', 1)
        if len(parts) != 2:
            return AuthenticationResult(
                is_authenticated=False,
                error_message="Invalid authorization header format"
            )
        
        auth_type, credentials = parts
        
        if auth_type.lower() == 'bearer':
            # JWT token
            return self.key_manager.authenticate_jwt_token(credentials)
        
        elif auth_type.lower() == 'apikey':
            # API key
            return self.key_manager.authenticate_api_key(credentials, ip_address)
        
        else:
            return AuthenticationResult(
                is_authenticated=False,
                error_message=f"Unsupported authentication type: {auth_type}"
            )
    
    def _authenticate_hmac_signature(self, signature: str, timestamp: str, 
                                   payload: bytes) -> AuthenticationResult:
        """Authenticate HMAC signature"""
        # For webhook authentication, we'd need to identify which webhook this is
        # For now, use a default validator
        if 'default' in self.hmac_validators:
            validator = self.hmac_validators['default']
            
            if validator.validate_signature(payload, signature, timestamp):
                return AuthenticationResult(
                    is_authenticated=True,
                    auth_method=AuthMethod.HMAC_SIGNATURE,
                    permissions={Permission.READ_TICKETS, Permission.UPDATE_TICKETS},
                    security_level=SecurityLevel.HIGH
                )
        
        return AuthenticationResult(
            is_authenticated=False,
            error_message="Invalid HMAC signature"
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'encryption_key': None,
            'webhook_secrets': {
                'default': os.getenv('WEBHOOK_SECRET', 'default_webhook_secret')
            },
            'jwt_expiry_hours': 1,
            'api_key_expiry_days': 90
        }

# Decorators for API security
def require_authentication(auth_method: Optional[AuthMethod] = None):
    """Decorator to require authentication"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract authentication from request context
            # This would be implemented based on your web framework
            auth_result = kwargs.get('auth_result')
            
            if not auth_result or not auth_result.is_authenticated:
                raise ValueError("Authentication required")
            
            if auth_method and auth_result.auth_method != auth_method:
                raise ValueError(f"Authentication method {auth_method.value} required")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth_result = kwargs.get('auth_result')
            
            if not auth_result or not auth_result.is_authenticated:
                raise ValueError("Authentication required")
            
            if permission not in auth_result.permissions:
                raise ValueError(f"Permission {permission.value} required")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def require_security_level(min_level: SecurityLevel):
    """Decorator to require minimum security level"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth_result = kwargs.get('auth_result')
            
            if not auth_result or not auth_result.is_authenticated:
                raise ValueError("Authentication required")
            
            # Compare security levels (assuming they have numeric values)
            level_values = {
                SecurityLevel.LOW: 1,
                SecurityLevel.MEDIUM: 2,
                SecurityLevel.HIGH: 3,
                SecurityLevel.CRITICAL: 4
            }
            
            if level_values[auth_result.security_level] < level_values[min_level]:
                raise ValueError(f"Security level {min_level.value} required")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global API authenticator instance
global_api_authenticator = None

def get_api_authenticator(config: Optional[Dict[str, Any]] = None) -> APIAuthenticator:
    """Get global API authenticator"""
    global global_api_authenticator
    if global_api_authenticator is None:
        global_api_authenticator = APIAuthenticator(config)
    return global_api_authenticator

# Convenience functions
def authenticate_request(**kwargs) -> AuthenticationResult:
    """Authenticate API request"""
    return get_api_authenticator().authenticate_request(**kwargs)

def generate_api_key(key_type: APIKeyType, **kwargs) -> Tuple[str, str]:
    """Generate API key"""
    return get_api_authenticator().key_manager.generate_api_key(key_type, **kwargs)

def generate_jwt_token(claims: JWTClaims) -> str:
    """Generate JWT token"""
    return get_api_authenticator().key_manager.generate_jwt_token(claims) 