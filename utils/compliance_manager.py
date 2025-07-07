"""
GDPR/CCPA Compliance Manager for SVL Chatbot
Comprehensive privacy compliance, data rights management, and regulatory compliance
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from pathlib import Path

from utils.logger import get_logger
from utils.security_core import PIIType, PIIDetectionResult, detect_and_mask_pii, DataEncryption
from utils.audit_logger import get_audit_logger, EventCategory, LogLevel

logger = get_logger("compliance_manager")

class DataSubjectRight(Enum):
    """Data subject rights under GDPR/CCPA"""
    ACCESS = "access"                    # Right to access personal data
    RECTIFICATION = "rectification"      # Right to rectify inaccurate data
    ERASURE = "erasure"                  # Right to be forgotten
    PORTABILITY = "portability"          # Right to data portability
    RESTRICTION = "restriction"          # Right to restrict processing
    OBJECTION = "objection"              # Right to object to processing
    OPT_OUT = "opt_out"                  # Right to opt out of sale (CCPA)
    KNOW = "know"                        # Right to know (CCPA)
    DELETE = "delete"                    # Right to delete (CCPA)

class LegalBasis(Enum):
    """Legal basis for data processing under GDPR"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

class DataCategory(Enum):
    """Categories of personal data"""
    IDENTITY = "identity"                # Name, ID numbers
    CONTACT = "contact"                  # Email, phone, address
    FINANCIAL = "financial"             # Payment information
    VEHICLE = "vehicle"                  # Vehicle information
    LOCATION = "location"                # Location data
    BEHAVIORAL = "behavioral"           # Usage patterns
    TECHNICAL = "technical"             # IP addresses, cookies
    BIOMETRIC = "biometric"             # Biometric data
    SPECIAL_CATEGORY = "special"        # Sensitive personal data

@dataclass
class ConsentRecord:
    """Record of user consent"""
    user_id: str
    consent_id: str
    purpose: str
    legal_basis: LegalBasis
    given_at: datetime
    withdrawn_at: Optional[datetime] = None
    version: str = "1.0"
    scope: List[str] = field(default_factory=list)
    is_active: bool = True
    processing_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataProcessingRecord:
    """Record of data processing activities"""
    processing_id: str
    user_id: str
    data_categories: List[DataCategory]
    purposes: List[str]
    legal_basis: LegalBasis
    consent_id: Optional[str] = None
    processed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retention_period: Optional[timedelta] = None
    recipients: List[str] = field(default_factory=list)
    transfer_countries: List[str] = field(default_factory=list)
    automated_decision_making: bool = False

@dataclass
class DataSubjectRequest:
    """Data subject rights request"""
    request_id: str
    user_id: str
    request_type: DataSubjectRight
    submitted_at: datetime
    status: str = "pending"  # pending, processing, completed, rejected
    processed_at: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    verification_completed: bool = False
    legal_review_required: bool = False
    notes: List[str] = field(default_factory=list)

@dataclass
class DataRetentionPolicy:
    """Data retention policy"""
    data_category: DataCategory
    retention_period: timedelta
    legal_basis: LegalBasis
    deletion_method: str = "secure_delete"
    exceptions: List[str] = field(default_factory=list)
    review_required: bool = False

class PrivacyNoticeManager:
    """Manages privacy notices and consent"""
    
    def __init__(self):
        self.current_privacy_notice_version = "2.0"
        self.consent_purposes = {
            "vehicle_theft_reporting": {
                "description": "Processing personal data to facilitate stolen vehicle reporting and recovery",
                "legal_basis": LegalBasis.LEGITIMATE_INTERESTS,
                "data_categories": [DataCategory.IDENTITY, DataCategory.CONTACT, DataCategory.VEHICLE],
                "retention_period": timedelta(days=2555),  # 7 years
                "required": True
            },
            "service_improvement": {
                "description": "Analyzing usage patterns to improve our services",
                "legal_basis": LegalBasis.CONSENT,
                "data_categories": [DataCategory.BEHAVIORAL, DataCategory.TECHNICAL],
                "retention_period": timedelta(days=365),
                "required": False
            },
            "marketing_communications": {
                "description": "Sending marketing communications about our services",
                "legal_basis": LegalBasis.CONSENT,
                "data_categories": [DataCategory.CONTACT],
                "retention_period": timedelta(days=1095),  # 3 years
                "required": False
            }
        }
    
    def get_privacy_notice(self, language: str = "en") -> Dict[str, Any]:
        """Get current privacy notice"""
        return {
            "version": self.current_privacy_notice_version,
            "effective_date": "2024-01-01",
            "language": language,
            "controller": {
                "name": "SVL Services Inc.",
                "contact": "privacy@svlservices.com",
                "dpo_contact": "dpo@svlservices.com"
            },
            "purposes": self.consent_purposes,
            "rights": {
                "access": "You have the right to access your personal data",
                "rectification": "You have the right to correct inaccurate data",
                "erasure": "You have the right to request deletion of your data",
                "portability": "You have the right to receive your data in a portable format",
                "restriction": "You have the right to restrict processing",
                "objection": "You have the right to object to processing",
                "complaint": "You have the right to lodge a complaint with supervisory authorities"
            },
            "retention": "Data is retained according to our retention policies",
            "international_transfers": "Data may be transferred internationally with appropriate safeguards",
            "automated_decision_making": "We may use automated decision making for fraud detection"
        }
    
    def record_consent(self, user_id: str, purposes: List[str], 
                      ip_address: Optional[str] = None) -> List[ConsentRecord]:
        """Record user consent for specified purposes"""
        consent_records = []
        
        for purpose in purposes:
            if purpose not in self.consent_purposes:
                continue
            
            purpose_info = self.consent_purposes[purpose]
            consent_record = ConsentRecord(
                user_id=user_id,
                consent_id=f"consent_{user_id}_{purpose}_{int(datetime.now().timestamp())}",
                purpose=purpose,
                legal_basis=purpose_info["legal_basis"],
                given_at=datetime.now(timezone.utc),
                version=self.current_privacy_notice_version,
                scope=purpose_info["data_categories"],
                processing_details={
                    "ip_address": ip_address,
                    "user_agent": "SVL Chatbot",
                    "method": "explicit_consent"
                }
            )
            
            consent_records.append(consent_record)
            
            # Log consent
            get_audit_logger().log_event(
                event_type="consent_given",
                category=EventCategory.COMPLIANCE,
                level=LogLevel.INFO,
                user_id=user_id,
                ip_address=ip_address,
                action="consent_given",
                resource=purpose,
                message=f"User consent given for {purpose}",
                details={"consent_id": consent_record.consent_id}
            )
        
        return consent_records

class DataMaskingService:
    """Advanced data masking for compliance"""
    
    def __init__(self):
        self.masking_rules = {
            PIIType.SSN: self._mask_ssn,
            PIIType.CREDIT_CARD: self._mask_credit_card,
            PIIType.EMAIL: self._mask_email,
            PIIType.PHONE: self._mask_phone,
            PIIType.NAME: self._mask_name,
            PIIType.ADDRESS: self._mask_address,
            PIIType.IP_ADDRESS: self._mask_ip,
        }
    
    def mask_data_for_compliance(self, data: Any, purpose: str = "general") -> Any:
        """Mask data based on compliance requirements"""
        if isinstance(data, str):
            return self._mask_string(data, purpose)
        elif isinstance(data, dict):
            return self._mask_dict(data, purpose)
        elif isinstance(data, list):
            return [self.mask_data_for_compliance(item, purpose) for item in data]
        else:
            return data
    
    def _mask_string(self, text: str, purpose: str) -> str:
        """Mask PII in string"""
        pii_result = detect_and_mask_pii(text)
        
        if pii_result.pii_found:
            # Log PII masking
            get_audit_logger().log_pii_access(
                user_id="system",
                pii_type=str(list(pii_result.detected_types.keys())),
                action="mask_data",
                approved=True,
                details={"purpose": purpose, "original_length": len(text)}
            )
            
            return pii_result.masked_text
        
        return text
    
    def _mask_dict(self, data: Dict[str, Any], purpose: str) -> Dict[str, Any]:
        """Mask PII in dictionary"""
        masked_data = {}
        
        for key, value in data.items():
            # Check if key indicates sensitive data
            if self._is_sensitive_key(key):
                masked_data[key] = self._apply_field_masking(value, key)
            else:
                masked_data[key] = self.mask_data_for_compliance(value, purpose)
        
        return masked_data
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if key indicates sensitive data"""
        sensitive_keywords = [
            'ssn', 'social', 'credit', 'card', 'password', 'secret',
            'token', 'key', 'email', 'phone', 'address', 'name'
        ]
        return any(keyword in key.lower() for keyword in sensitive_keywords)
    
    def _apply_field_masking(self, value: Any, field_name: str) -> str:
        """Apply field-specific masking"""
        if not isinstance(value, str):
            value = str(value)
        
        field_lower = field_name.lower()
        
        if 'email' in field_lower:
            return self._mask_email(value)
        elif 'phone' in field_lower:
            return self._mask_phone(value)
        elif 'ssn' in field_lower or 'social' in field_lower:
            return self._mask_ssn(value)
        elif 'credit' in field_lower or 'card' in field_lower:
            return self._mask_credit_card(value)
        elif 'name' in field_lower:
            return self._mask_name(value)
        elif 'address' in field_lower:
            return self._mask_address(value)
        else:
            return "*" * min(len(value), 10)
    
    def _mask_ssn(self, ssn: str) -> str:
        """Mask SSN"""
        return "***-**-****"
    
    def _mask_credit_card(self, card: str) -> str:
        """Mask credit card"""
        return "**** **** **** ****"
    
    def _mask_email(self, email: str) -> str:
        """Mask email address"""
        if '@' in email:
            local, domain = email.split('@', 1)
            return f"***@{domain}"
        return "***@***.***"
    
    def _mask_phone(self, phone: str) -> str:
        """Mask phone number"""
        return "***-***-****"
    
    def _mask_name(self, name: str) -> str:
        """Mask name"""
        words = name.split()
        return " ".join("*" * len(word) for word in words)
    
    def _mask_address(self, address: str) -> str:
        """Mask address"""
        return "*** *** *** ***"
    
    def _mask_ip(self, ip: str) -> str:
        """Mask IP address"""
        return "***.***.***.*"

class DataSubjectRightsManager:
    """Manages data subject rights requests"""
    
    def __init__(self):
        self.requests: Dict[str, DataSubjectRequest] = {}
        self.masking_service = DataMaskingService()
        self._lock = threading.Lock()
    
    def submit_request(self, user_id: str, request_type: DataSubjectRight,
                      details: Optional[Dict[str, Any]] = None) -> str:
        """Submit a data subject rights request"""
        with self._lock:
            request_id = f"dsr_{user_id}_{request_type.value}_{int(datetime.now().timestamp())}"
            
            request = DataSubjectRequest(
                request_id=request_id,
                user_id=user_id,
                request_type=request_type,
                submitted_at=datetime.now(timezone.utc)
            )
            
            # Set legal review requirement
            if request_type in [DataSubjectRight.ERASURE, DataSubjectRight.RESTRICTION]:
                request.legal_review_required = True
            
            self.requests[request_id] = request
            
            # Log request
            get_audit_logger().log_event(
                event_type="data_subject_request",
                category=EventCategory.COMPLIANCE,
                level=LogLevel.INFO,
                user_id=user_id,
                action="submit_request",
                resource=request_type.value,
                message=f"Data subject rights request submitted: {request_type.value}",
                details={"request_id": request_id, "details": details}
            )
            
            return request_id
    
    def process_access_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Process right of access request"""
        request = self.requests.get(request_id)
        if not request or request.request_type != DataSubjectRight.ACCESS:
            return None
        
        try:
            # Collect user data (simplified for POC)
            user_data = self._collect_user_data(request.user_id)
            
            # Mask sensitive data for third parties
            masked_data = self.masking_service.mask_data_for_compliance(
                user_data, "data_subject_access"
            )
            
            # Update request
            request.status = "completed"
            request.processed_at = datetime.now(timezone.utc)
            request.response_data = masked_data
            
            # Log completion
            get_audit_logger().log_event(
                event_type="access_request_completed",
                category=EventCategory.COMPLIANCE,
                level=LogLevel.INFO,
                user_id=request.user_id,
                action="process_access_request",
                resource="user_data",
                result="completed",
                message="Right of access request completed",
                details={"request_id": request_id, "data_items": len(masked_data)}
            )
            
            return masked_data
            
        except Exception as e:
            logger.error(f"Failed to process access request {request_id}: {e}")
            request.status = "failed"
            return None
    
    def process_erasure_request(self, request_id: str, 
                               legal_review_approved: bool = False) -> bool:
        """Process right to be forgotten request"""
        request = self.requests.get(request_id)
        if not request or request.request_type != DataSubjectRight.ERASURE:
            return False
        
        if request.legal_review_required and not legal_review_approved:
            request.status = "pending_legal_review"
            return False
        
        try:
            # Check for legal obligations to retain data
            retention_exceptions = self._check_retention_exceptions(request.user_id)
            
            if retention_exceptions:
                request.status = "partially_completed"
                request.notes.append(f"Some data retained due to: {', '.join(retention_exceptions)}")
            else:
                # Perform data deletion (simplified for POC)
                deleted_count = self._delete_user_data(request.user_id)
                request.status = "completed"
                request.notes.append(f"Deleted {deleted_count} data items")
            
            request.processed_at = datetime.now(timezone.utc)
            
            # Log erasure
            get_audit_logger().log_event(
                event_type="erasure_request_completed",
                category=EventCategory.COMPLIANCE,
                level=LogLevel.SECURITY,
                user_id=request.user_id,
                action="process_erasure_request",
                resource="user_data",
                result=request.status,
                message=f"Right to erasure request {request.status}",
                details={"request_id": request_id, "exceptions": retention_exceptions}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process erasure request {request_id}: {e}")
            request.status = "failed"
            return False
    
    def process_portability_request(self, request_id: str) -> Optional[str]:
        """Process data portability request"""
        request = self.requests.get(request_id)
        if not request or request.request_type != DataSubjectRight.PORTABILITY:
            return None
        
        try:
            # Collect user data in portable format
            portable_data = self._export_user_data(request.user_id)
            
            # Generate secure download link or file
            export_file = self._create_export_file(request.user_id, portable_data)
            
            request.status = "completed"
            request.processed_at = datetime.now(timezone.utc)
            request.response_data = {"export_file": export_file}
            
            # Log portability
            get_audit_logger().log_event(
                event_type="portability_request_completed",
                category=EventCategory.COMPLIANCE,
                level=LogLevel.INFO,
                user_id=request.user_id,
                action="process_portability_request",
                resource="user_data",
                result="completed",
                message="Data portability request completed",
                details={"request_id": request_id, "export_file": export_file}
            )
            
            return export_file
            
        except Exception as e:
            logger.error(f"Failed to process portability request {request_id}: {e}")
            request.status = "failed"
            return None
    
    def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect all user data for access request"""
        # Simplified implementation for POC
        return {
            "user_profile": {"user_id": user_id, "created_at": "2024-01-01"},
            "conversations": [],
            "vehicle_reports": [],
            "consent_records": [],
            "processing_records": []
        }
    
    def _check_retention_exceptions(self, user_id: str) -> List[str]:
        """Check for legal obligations to retain data"""
        exceptions = []
        
        # Example retention exceptions
        # In practice, this would check against actual retention policies
        if self._has_ongoing_legal_proceedings(user_id):
            exceptions.append("ongoing_legal_proceedings")
        
        if self._has_regulatory_retention_requirement(user_id):
            exceptions.append("regulatory_retention")
        
        return exceptions
    
    def _has_ongoing_legal_proceedings(self, user_id: str) -> bool:
        """Check if user has ongoing legal proceedings"""
        # Simplified check - in practice would check legal case database
        return False
    
    def _has_regulatory_retention_requirement(self, user_id: str) -> bool:
        """Check if data must be retained for regulatory reasons"""
        # Simplified check - in practice would check retention policies
        return False
    
    def _delete_user_data(self, user_id: str) -> int:
        """Delete user data (simplified for POC)"""
        # In practice, this would delete from all systems
        deleted_count = 0
        
        # Delete from different data stores
        # deleted_count += delete_from_database(user_id)
        # deleted_count += delete_from_logs(user_id)
        # deleted_count += delete_from_backups(user_id)
        
        return deleted_count
    
    def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data in portable format"""
        user_data = self._collect_user_data(user_id)
        
        # Format data for portability
        portable_data = {
            "export_info": {
                "user_id": user_id,
                "export_date": datetime.now(timezone.utc).isoformat(),
                "format": "JSON",
                "version": "1.0"
            },
            "data": user_data
        }
        
        return portable_data
    
    def _create_export_file(self, user_id: str, data: Dict[str, Any]) -> str:
        """Create secure export file"""
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"svl_data_export_{user_id}_{timestamp}.json"
        
        # In practice, this would create an encrypted file and secure download link
        return filename

class ComplianceManager:
    """Main compliance management system"""
    
    def __init__(self):
        self.privacy_notice_manager = PrivacyNoticeManager()
        self.data_masking_service = DataMaskingService()
        self.rights_manager = DataSubjectRightsManager()
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.processing_records: Dict[str, List[DataProcessingRecord]] = {}
        self.retention_policies: List[DataRetentionPolicy] = self._get_default_retention_policies()
        self._lock = threading.Lock()
    
    def record_consent(self, user_id: str, purposes: List[str], 
                      ip_address: Optional[str] = None) -> List[ConsentRecord]:
        """Record user consent"""
        with self._lock:
            consent_records = self.privacy_notice_manager.record_consent(
                user_id, purposes, ip_address
            )
            
            if user_id not in self.consent_records:
                self.consent_records[user_id] = []
            
            self.consent_records[user_id].extend(consent_records)
            return consent_records
    
    def withdraw_consent(self, user_id: str, consent_id: str) -> bool:
        """Withdraw user consent"""
        with self._lock:
            user_consents = self.consent_records.get(user_id, [])
            
            for consent in user_consents:
                if consent.consent_id == consent_id and consent.is_active:
                    consent.withdrawn_at = datetime.now(timezone.utc)
                    consent.is_active = False
                    
                    # Log consent withdrawal
                    get_audit_logger().log_event(
                        event_type="consent_withdrawn",
                        category=EventCategory.COMPLIANCE,
                        level=LogLevel.INFO,
                        user_id=user_id,
                        action="withdraw_consent",
                        resource=consent.purpose,
                        message=f"User consent withdrawn for {consent.purpose}",
                        details={"consent_id": consent_id}
                    )
                    
                    return True
            
            return False
    
    def check_processing_lawfulness(self, user_id: str, purpose: str, 
                                   data_categories: List[DataCategory]) -> bool:
        """Check if data processing is lawful"""
        # Check for valid consent or legal basis
        user_consents = self.consent_records.get(user_id, [])
        
        for consent in user_consents:
            if (consent.purpose == purpose and consent.is_active and
                all(cat.value in consent.scope for cat in data_categories)):
                return True
        
        # Check for other legal bases
        purpose_config = self.privacy_notice_manager.consent_purposes.get(purpose)
        if purpose_config and purpose_config["legal_basis"] != LegalBasis.CONSENT:
            return True
        
        return False
    
    def record_processing_activity(self, user_id: str, data_categories: List[DataCategory],
                                 purposes: List[str], legal_basis: LegalBasis,
                                 consent_id: Optional[str] = None) -> str:
        """Record data processing activity"""
        with self._lock:
            processing_id = f"proc_{user_id}_{int(datetime.now().timestamp())}"
            
            record = DataProcessingRecord(
                processing_id=processing_id,
                user_id=user_id,
                data_categories=data_categories,
                purposes=purposes,
                legal_basis=legal_basis,
                consent_id=consent_id
            )
            
            if user_id not in self.processing_records:
                self.processing_records[user_id] = []
            
            self.processing_records[user_id].append(record)
            
            # Log processing activity
            get_audit_logger().log_data_access(
                user_id=user_id,
                resource=",".join(cat.value for cat in data_categories),
                action="process_data",
                details={
                    "processing_id": processing_id,
                    "purposes": purposes,
                    "legal_basis": legal_basis.value
                }
            )
            
            return processing_id
    
    def handle_data_subject_request(self, user_id: str, request_type: DataSubjectRight,
                                  details: Optional[Dict[str, Any]] = None) -> str:
        """Handle data subject rights request"""
        return self.rights_manager.submit_request(user_id, request_type, details)
    
    def get_user_data_for_access_request(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data for access request"""
        # Find the most recent access request
        user_requests = [
            req for req in self.rights_manager.requests.values()
            if req.user_id == user_id and req.request_type == DataSubjectRight.ACCESS
        ]
        
        if not user_requests:
            return None
        
        latest_request = max(user_requests, key=lambda r: r.submitted_at)
        return self.rights_manager.process_access_request(latest_request.request_id)
    
    def apply_data_retention_policies(self) -> Dict[str, int]:
        """Apply data retention policies"""
        results = {"deleted_records": 0, "archived_records": 0, "errors": 0}
        
        for policy in self.retention_policies:
            try:
                # Check for data that exceeds retention period
                cutoff_date = datetime.now(timezone.utc) - policy.retention_period
                
                # In practice, this would query actual data stores
                # For POC, we'll just log the policy application
                get_audit_logger().log_event(
                    event_type="retention_policy_applied",
                    category=EventCategory.COMPLIANCE,
                    level=LogLevel.INFO,
                    action="apply_retention_policy",
                    resource=policy.data_category.value,
                    message=f"Applied retention policy for {policy.data_category.value}",
                    details={
                        "retention_period_days": policy.retention_period.days,
                        "cutoff_date": cutoff_date.isoformat(),
                        "deletion_method": policy.deletion_method
                    }
                )
                
            except Exception as e:
                logger.error(f"Error applying retention policy for {policy.data_category}: {e}")
                results["errors"] += 1
        
        return results
    
    def generate_compliance_report(self, start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report"""
        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "consent_statistics": self._get_consent_statistics(start_date, end_date),
            "processing_activities": self._get_processing_statistics(start_date, end_date),
            "data_subject_requests": self._get_dsr_statistics(start_date, end_date),
            "privacy_incidents": self._get_privacy_incidents(start_date, end_date),
            "retention_policy_compliance": self._check_retention_compliance()
        }
        
        return report
    
    def _get_consent_statistics(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """Get consent statistics for period"""
        stats = {"consents_given": 0, "consents_withdrawn": 0}
        
        for user_consents in self.consent_records.values():
            for consent in user_consents:
                if start_date <= consent.given_at <= end_date:
                    stats["consents_given"] += 1
                
                if (consent.withdrawn_at and 
                    start_date <= consent.withdrawn_at <= end_date):
                    stats["consents_withdrawn"] += 1
        
        return stats
    
    def _get_processing_statistics(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """Get processing statistics for period"""
        stats = {"total_processing_activities": 0, "by_legal_basis": {}}
        
        for user_records in self.processing_records.values():
            for record in user_records:
                if start_date <= record.processed_at <= end_date:
                    stats["total_processing_activities"] += 1
                    
                    basis = record.legal_basis.value
                    stats["by_legal_basis"][basis] = stats["by_legal_basis"].get(basis, 0) + 1
        
        return stats
    
    def _get_dsr_statistics(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """Get data subject request statistics"""
        stats = {"total_requests": 0, "by_type": {}, "by_status": {}}
        
        for request in self.rights_manager.requests.values():
            if start_date <= request.submitted_at <= end_date:
                stats["total_requests"] += 1
                
                req_type = request.request_type.value
                stats["by_type"][req_type] = stats["by_type"].get(req_type, 0) + 1
                
                status = request.status
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
        
        return stats
    
    def _get_privacy_incidents(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get privacy incidents for period"""
        # In practice, this would query the audit log for privacy incidents
        return {
            "total_incidents": 0,
            "by_severity": {},
            "breach_notifications": 0
        }
    
    def _check_retention_compliance(self) -> Dict[str, Any]:
        """Check retention policy compliance"""
        return {
            "policies_applied": len(self.retention_policies),
            "compliance_percentage": 100.0,  # Simplified for POC
            "overdue_deletions": 0
        }
    
    def _get_default_retention_policies(self) -> List[DataRetentionPolicy]:
        """Get default data retention policies"""
        return [
            DataRetentionPolicy(
                data_category=DataCategory.IDENTITY,
                retention_period=timedelta(days=2555),  # 7 years
                legal_basis=LegalBasis.LEGAL_OBLIGATION
            ),
            DataRetentionPolicy(
                data_category=DataCategory.CONTACT,
                retention_period=timedelta(days=1095),  # 3 years
                legal_basis=LegalBasis.LEGITIMATE_INTERESTS
            ),
            DataRetentionPolicy(
                data_category=DataCategory.BEHAVIORAL,
                retention_period=timedelta(days=365),   # 1 year
                legal_basis=LegalBasis.CONSENT
            ),
            DataRetentionPolicy(
                data_category=DataCategory.TECHNICAL,
                retention_period=timedelta(days=90),    # 3 months
                legal_basis=LegalBasis.LEGITIMATE_INTERESTS
            )
        ]

# Global compliance manager instance
global_compliance_manager = ComplianceManager()

def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager"""
    return global_compliance_manager

# Convenience functions
def record_consent(user_id: str, purposes: List[str], ip_address: Optional[str] = None) -> List[ConsentRecord]:
    """Record user consent"""
    return get_compliance_manager().record_consent(user_id, purposes, ip_address)

def submit_data_subject_request(user_id: str, request_type: DataSubjectRight) -> str:
    """Submit data subject rights request"""
    return get_compliance_manager().handle_data_subject_request(user_id, request_type)

def mask_data_for_compliance(data: Any, purpose: str = "general") -> Any:
    """Mask data for compliance"""
    return get_compliance_manager().data_masking_service.mask_data_for_compliance(data, purpose) 