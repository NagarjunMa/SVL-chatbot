"""
Core Security Utilities for SVL Chatbot
Comprehensive security framework with input validation, PII detection, and data protection
"""

import re
import hmac
import hashlib
import secrets
import html
import json
import base64
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import bleach
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from utils.logger import get_logger

logger = get_logger("security_core")

class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PIIType(Enum):
    """Types of Personally Identifiable Information"""
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    DRIVERS_LICENSE = "drivers_license"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    NAME = "name"

@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: str
    session_id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    permissions: Set[str] = field(default_factory=set)
    rate_limit_key: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    sanitized_input: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    security_flags: List[str] = field(default_factory=list)

@dataclass
class PIIDetectionResult:
    """Result of PII detection"""
    pii_found: bool
    detected_types: Dict[PIIType, List[str]] = field(default_factory=dict)
    masked_text: str = ""
    confidence_scores: Dict[PIIType, float] = field(default_factory=dict)
    flagged_positions: List[Tuple[int, int, PIIType]] = field(default_factory=list)

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self):
        self.max_input_length = 10000
        self.min_input_length = 1
        
        # Dangerous patterns for injection attacks
        self.sql_injection_patterns = [
            r"('\s*(OR|AND)\s*')|('\s*(=|!=|<>)\s*')",
            r"(UNION\s+SELECT|INSERT\s+INTO|DELETE\s+FROM|DROP\s+TABLE)",
            r"(EXEC\s*\(|EXECUTE\s*\(|SP_|XP_)",
            r"(\|\||&&|;|--|/\*|\*/)",
            r"(SCRIPT\s*>|<\s*SCRIPT)",
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<\s*script[^>]*>.*?</\s*script\s*>",
            r"<\s*iframe[^>]*>.*?</\s*iframe\s*>",
            r"javascript\s*:",
            r"on\w+\s*=",
            r"<\s*img[^>]*src\s*=\s*[\"']?\s*javascript:",
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"[;&|`$(){}[\]<>]",
            r"(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig|ping)",
        ]
        
        # Allowed HTML tags for safe content
        self.allowed_tags = ['b', 'i', 'em', 'strong', 'p', 'br', 'ul', 'ol', 'li']
        self.allowed_attributes = {}
    
    def validate_input(self, user_input: str, context: SecurityContext) -> ValidationResult:
        """Comprehensive input validation"""
        errors = []
        warnings = []
        security_flags = []
        
        # Basic length validation
        if len(user_input) < self.min_input_length:
            errors.append("Input too short")
        
        if len(user_input) > self.max_input_length:
            errors.append(f"Input too long (max {self.max_input_length} characters)")
            user_input = user_input[:self.max_input_length]
            warnings.append("Input truncated to maximum length")
        
        # Check for injection attacks
        security_flags.extend(self._check_injection_attacks(user_input))
        
        # Sanitize input
        sanitized_input = self._sanitize_input(user_input)
        
        # Additional context-based validation
        if context.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            additional_flags = self._advanced_security_checks(sanitized_input)
            security_flags.extend(additional_flags)
        
        is_valid = len(errors) == 0 and len([f for f in security_flags if "CRITICAL" in f]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_input if is_valid else None,
            errors=errors,
            warnings=warnings,
            security_flags=security_flags
        )
    
    def _check_injection_attacks(self, user_input: str) -> List[str]:
        """Check for various injection attack patterns"""
        flags = []
        input_lower = user_input.lower()
        
        # SQL injection detection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, input_lower, re.IGNORECASE):
                flags.append("CRITICAL: Potential SQL injection detected")
                logger.warning(f"SQL injection pattern detected: {pattern}")
                break
        
        # XSS detection
        for pattern in self.xss_patterns:
            if re.search(pattern, input_lower, re.IGNORECASE):
                flags.append("CRITICAL: Potential XSS attack detected")
                logger.warning(f"XSS pattern detected: {pattern}")
                break
        
        # Command injection detection
        for pattern in self.command_injection_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                flags.append("HIGH: Potential command injection detected")
                logger.warning(f"Command injection pattern detected: {pattern}")
                break
        
        return flags
    
    def _sanitize_input(self, user_input: str) -> str:
        """Sanitize user input"""
        # HTML escape
        sanitized = html.escape(user_input)
        
        # Clean HTML with bleach
        sanitized = bleach.clean(
            sanitized,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def _advanced_security_checks(self, user_input: str) -> List[str]:
        """Advanced security checks for high-security contexts"""
        flags = []
        
        # Check for encoded payloads
        try:
            decoded = base64.b64decode(user_input, validate=True)
            if self._check_injection_attacks(decoded.decode('utf-8', errors='ignore')):
                flags.append("HIGH: Encoded malicious payload detected")
        except:
            pass
        
        # Check for unusual character patterns
        if len(re.findall(r'[^\w\s\.,!?;:\-\'"()]', user_input)) > len(user_input) * 0.1:
            flags.append("MEDIUM: Unusual character patterns detected")
        
        # Check for very long words (potential buffer overflow)
        words = user_input.split()
        if any(len(word) > 50 for word in words):
            flags.append("MEDIUM: Unusually long words detected")
        
        return flags

class PIIDetector:
    """Advanced PII detection and masking"""
    
    def __init__(self):
        self.patterns = {
            PIIType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # 123-45-6789
                r'\b\d{3}\s\d{2}\s\d{4}\b',  # 123 45 6789
                r'\b\d{9}\b',  # 123456789
            ],
            PIIType.CREDIT_CARD: [
                r'\b4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',  # Visa
                r'\b5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',  # MasterCard
                r'\b3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}\b',  # Amex
                r'\b6011[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',  # Discover
            ],
            PIIType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            PIIType.PHONE: [
                r'\b\+?1?[\s\-\.]?\(?[0-9]{3}\)?[\s\-\.]?[0-9]{3}[\s\-\.]?[0-9]{4}\b',
                r'\b\d{3}[\s\-\.]\d{3}[\s\-\.]\d{4}\b',
            ],
            PIIType.DRIVERS_LICENSE: [
                r'\b[A-Z]{1,2}\d{6,8}\b',  # State DL patterns (simplified)
            ],
            PIIType.BANK_ACCOUNT: [
                r'\b\d{8,17}\b',  # Generic account number
            ],
            PIIType.IP_ADDRESS: [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',  # IPv4
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',  # IPv6 (simplified)
            ],
            PIIType.DATE_OF_BIRTH: [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
                r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # YYYY-MM-DD
            ],
        }
        
        # Common name patterns (basic detection)
        self.name_indicators = [
            r'\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\bi am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\bcall me\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
    
    def detect_pii(self, text: str) -> PIIDetectionResult:
        """Detect PII in text with confidence scoring"""
        detected_types = {}
        confidence_scores = {}
        flagged_positions = []
        
        # Check each PII type
        for pii_type, patterns in self.patterns.items():
            matches = []
            total_confidence = 0
            
            for pattern in patterns:
                pattern_matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in pattern_matches:
                    matches.append(match.group())
                    flagged_positions.append((match.start(), match.end(), pii_type))
                    
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_confidence(match.group(), pii_type)
                    total_confidence += confidence
            
            if matches:
                detected_types[pii_type] = matches
                confidence_scores[pii_type] = min(total_confidence / len(matches), 1.0)
        
        # Check for names
        name_matches = self._detect_names(text)
        if name_matches:
            detected_types[PIIType.NAME] = name_matches
            confidence_scores[PIIType.NAME] = 0.8
        
        # Generate masked text
        masked_text = self._mask_pii(text, flagged_positions)
        
        return PIIDetectionResult(
            pii_found=len(detected_types) > 0,
            detected_types=detected_types,
            masked_text=masked_text,
            confidence_scores=confidence_scores,
            flagged_positions=flagged_positions
        )
    
    def _calculate_confidence(self, match: str, pii_type: PIIType) -> float:
        """Calculate confidence score for PII detection"""
        if pii_type == PIIType.SSN:
            # Check for valid SSN format and ranges
            digits = re.sub(r'[\s\-]', '', match)
            if len(digits) == 9 and not digits.startswith('000') and not digits[3:5] == '00':
                return 0.95
            return 0.7
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Luhn algorithm check for credit cards
            digits = re.sub(r'[\s\-]', '', match)
            if self._luhn_check(digits):
                return 0.95
            return 0.6
        
        elif pii_type == PIIType.EMAIL:
            # Email format validation
            if '@' in match and '.' in match.split('@')[1]:
                return 0.9
            return 0.5
        
        elif pii_type == PIIType.PHONE:
            # Phone number format validation
            digits = re.sub(r'[\s\-\.\(\)]', '', match)
            if len(digits) == 10 or (len(digits) == 11 and digits[0] == '1'):
                return 0.85
            return 0.6
        
        return 0.7  # Default confidence
    
    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        def luhn_checksum(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10
        
        try:
            return luhn_checksum(int(card_number)) == 0
        except ValueError:
            return False
    
    def _detect_names(self, text: str) -> List[str]:
        """Detect potential names in text"""
        names = []
        
        for pattern in self.name_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1)
                # Basic validation: 2-4 words, proper capitalization
                words = name.split()
                if 1 <= len(words) <= 4 and all(word[0].isupper() for word in words):
                    names.append(name)
        
        return names
    
    def _mask_pii(self, text: str, flagged_positions: List[Tuple[int, int, PIIType]]) -> str:
        """Mask PII in text"""
        # Sort positions in reverse order to maintain indices
        flagged_positions.sort(key=lambda x: x[0], reverse=True)
        
        masked_text = text
        for start, end, pii_type in flagged_positions:
            original = text[start:end]
            mask = self._generate_mask(original, pii_type)
            masked_text = masked_text[:start] + mask + masked_text[end:]
        
        return masked_text
    
    def _generate_mask(self, original: str, pii_type: PIIType) -> str:
        """Generate appropriate mask for PII type"""
        if pii_type == PIIType.SSN:
            return "***-**-****"
        elif pii_type == PIIType.CREDIT_CARD:
            return "**** **** **** ****"
        elif pii_type == PIIType.EMAIL:
            parts = original.split('@')
            if len(parts) == 2:
                return f"***@{parts[1]}"
            return "***@***.***"
        elif pii_type == PIIType.PHONE:
            return "***-***-****"
        elif pii_type == PIIType.NAME:
            words = original.split()
            return " ".join("*" * len(word) for word in words)
        else:
            return "*" * min(len(original), 10)

class DataEncryption:
    """Data encryption utilities"""
    
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            key = Fernet.generate_key()
        self.fernet = Fernet(key)
        self._key = key
    
    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a new encryption key"""
        return Fernet.generate_key()
    
    @classmethod
    def derive_key_from_password(cls, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password"""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        encrypted_data = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Invalid encrypted data")
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data"""
        json_data = json.dumps(data, default=str)
        return self.encrypt(json_data)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data"""
        json_data = self.decrypt(encrypted_data)
        return json.loads(json_data)

class SecurityUtils:
    """General security utilities"""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate secure session ID"""
        return f"svl_session_{secrets.token_urlsafe(24)}"
    
    @staticmethod
    def hash_data(data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash data with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return hash_obj.hex(), salt
    
    @staticmethod
    def verify_hash(data: str, hashed_data: str, salt: str) -> bool:
        """Verify hashed data"""
        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return hmac.compare_digest(hash_obj.hex(), hashed_data)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations"""
        # Remove path traversal attempts
        filename = filename.replace('..', '').replace('/', '').replace('\\', '')
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate IP address format"""
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_private_ip(ip: str) -> bool:
        """Check if IP address is private"""
        import ipaddress
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private
        except ValueError:
            return False

# Initialize global security components
input_validator = InputValidator()
pii_detector = PIIDetector()
security_utils = SecurityUtils()

def validate_and_sanitize(user_input: str, context: SecurityContext) -> ValidationResult:
    """Main entry point for input validation and sanitization"""
    return input_validator.validate_input(user_input, context)

def detect_and_mask_pii(text: str) -> PIIDetectionResult:
    """Main entry point for PII detection and masking"""
    return pii_detector.detect_pii(text)

def create_security_context(user_id: str, session_id: str, **kwargs) -> SecurityContext:
    """Create security context for operations"""
    return SecurityContext(
        user_id=user_id,
        session_id=session_id,
        **kwargs
    ) 