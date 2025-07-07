import re
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from data.models import Conversation, Ticket, Message, PII_PATTERNS
from utils.logger import get_logger

logger = get_logger("data_utils")

class DataValidator:
    """Data validation and sanitization utilities"""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Remove potentially dangerous characters and normalize text"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def validate_phone_number(phone: str) -> bool:
        """Validate phone number format"""
        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)
        return len(digits) == 10
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_vin(vin: str) -> bool:
        """Validate VIN format"""
        pattern = r'^[A-HJ-NPR-Z0-9]{17}$'
        return bool(re.match(pattern, vin.upper()))

class PIIHandler:
    """PII detection and masking utilities"""
    
    @staticmethod
    def detect_pii(text: str) -> Dict[str, List[str]]:
        """Detect PII in text"""
        pii_found = {}
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pii_found[pii_type] = matches
        return pii_found
    
    @staticmethod
    def mask_pii(text: str, mask_char: str = '*') -> str:
        """Mask PII in text"""
        masked_text = text
        
        # Mask SSN
        masked_text = re.sub(r'\b(\d{3})-(\d{2})-(\d{4})\b', 
                            rf'\1-{mask_char * 2}-{mask_char * 4}', masked_text)
        
        # Mask credit cards
        masked_text = re.sub(r'\b(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b',
                            rf'\1-{mask_char * 4}-{mask_char * 4}-{mask_char * 4}', masked_text)
        
        # Mask phone numbers
        masked_text = re.sub(r'\b(\d{3})[-.]?(\d{3})[-.]?(\d{4})\b',
                            rf'\1-{mask_char * 3}-{mask_char * 4}', masked_text)
        
        # Mask email addresses
        masked_text = re.sub(r'\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
                            rf'{mask_char * 3}@{mask_char * 3}.{mask_char * 2}', masked_text)
        
        return masked_text
    
    @staticmethod
    def is_sensitive_field(field_name: str) -> bool:
        """Check if a field contains sensitive information"""
        sensitive_fields = ['phone', 'email', 'address', 'ssn', 'credit_card', 'vin']
        return any(sensitive in field_name.lower() for sensitive in sensitive_fields)

class DataExporter:
    """Data export and reporting utilities"""
    
    @staticmethod
    def export_conversations_to_csv(conversations: List[Conversation], 
                                  output_path: str,
                                  include_pii: bool = False) -> bool:
        """Export conversations to CSV format"""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['conversation_id', 'user_id', 'created_at', 'message_count', 'pii_detected']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for conv in conversations:
                    pii_summary = conv.get_pii_summary()
                    pii_detected = ', '.join(pii_summary.keys()) if pii_summary else 'None'
                    
                    writer.writerow({
                        'conversation_id': conv.conversation_id,
                        'user_id': conv.user_id,
                        'created_at': conv.created_at.isoformat(),
                        'message_count': len(conv.messages),
                        'pii_detected': pii_detected
                    })
            
            logger.info(f"Exported {len(conversations)} conversations to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export conversations to CSV: {e}")
            return False
    
    @staticmethod
    def export_tickets_to_json(tickets: List[Ticket], 
                              output_path: str,
                              include_sensitive: bool = False) -> bool:
        """Export tickets to JSON format"""
        try:
            export_data = []
            
            for ticket in tickets:
                ticket_data = ticket.dict()
                
                if not include_sensitive:
                    # Remove sensitive information
                    if 'owner_info' in ticket_data:
                        owner_info = ticket_data['owner_info']
                        owner_info['phone'] = '[REDACTED]'
                        owner_info['email'] = '[REDACTED]'
                        owner_info['address'] = '[REDACTED]'
                
                export_data.append(ticket_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(tickets)} tickets to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export tickets to JSON: {e}")
            return False
    
    @staticmethod
    def generate_report(conversations: List[Conversation], 
                       tickets: List[Ticket],
                       output_path: str) -> bool:
        """Generate a comprehensive report"""
        try:
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'summary': {
                    'total_conversations': len(conversations),
                    'total_tickets': len(tickets),
                    'active_tickets': len([t for t in tickets if t.status == 'open']),
                    'resolved_tickets': len([t for t in tickets if t.status == 'resolved'])
                },
                'conversations_by_date': {},
                'tickets_by_status': {},
                'pii_summary': {}
            }
            
            # Group conversations by date
            for conv in conversations:
                date_key = conv.created_at.date().isoformat()
                report['conversations_by_date'][date_key] = report['conversations_by_date'].get(date_key, 0) + 1
            
            # Group tickets by status
            for ticket in tickets:
                status = ticket.status
                report['tickets_by_status'][status] = report['tickets_by_status'].get(status, 0) + 1
            
            # PII summary
            total_pii = {}
            for conv in conversations:
                pii_summary = conv.get_pii_summary()
                for pii_type, count in pii_summary.items():
                    total_pii[pii_type] = total_pii.get(pii_type, 0) + count
            report['pii_summary'] = total_pii
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Generated report: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return False

class ComplianceLogger:
    """Compliance and audit logging utilities"""
    
    @staticmethod
    def log_data_access(user_id: str, record_type: str, record_id: str, 
                       operation: str, success: bool, details: Dict[str, Any] = None):
        """Log data access for compliance"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'record_type': record_type,
            'record_id': record_id,
            'operation': operation,
            'success': success,
            'details': details or {}
        }
        
        logger.info(f"COMPLIANCE: {json.dumps(log_entry)}")
    
    @staticmethod
    def log_pii_access(user_id: str, pii_type: str, record_id: str, 
                      access_reason: str, authorized: bool):
        """Log PII access specifically"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'pii_type': pii_type,
            'record_id': record_id,
            'access_reason': access_reason,
            'authorized': authorized
        }
        
        logger.warning(f"PII_ACCESS: {json.dumps(log_entry)}")
    
    @staticmethod
    def log_data_export(user_id: str, export_type: str, record_count: int, 
                       include_sensitive: bool, destination: str):
        """Log data export activities"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'export_type': export_type,
            'record_count': record_count,
            'include_sensitive': include_sensitive,
            'destination': destination
        }
        
        logger.info(f"DATA_EXPORT: {json.dumps(log_entry)}")

class DataCleaner:
    """Data cleaning and maintenance utilities"""
    
    @staticmethod
    def clean_old_conversations(conversations: List[Conversation], 
                               days_old: int = 90) -> List[Conversation]:
        """Filter out conversations older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        return [conv for conv in conversations if conv.updated_at > cutoff_date]
    
    @staticmethod
    def anonymize_conversation(conversation: Conversation) -> Conversation:
        """Anonymize a conversation by removing PII"""
        anonymized_conv = conversation.copy()
        
        for message in anonymized_conv.messages:
            message.content = PIIHandler.mask_pii(message.content)
        
        return anonymized_conv
    
    @staticmethod
    def validate_data_integrity(conversations: List[Conversation], 
                               tickets: List[Ticket]) -> Dict[str, List[str]]:
        """Validate data integrity and return issues found"""
        issues = {
            'conversations': [],
            'tickets': []
        }
        
        # Check conversations
        for conv in conversations:
            if not conv.conversation_id:
                issues['conversations'].append(f"Missing conversation ID")
            if not conv.user_id:
                issues['conversations'].append(f"Missing user ID in conversation {conv.conversation_id}")
            if len(conv.messages) > 1000:
                issues['conversations'].append(f"Conversation {conv.conversation_id} has too many messages")
        
        # Check tickets
        for ticket in tickets:
            if not ticket.ticket_id:
                issues['tickets'].append("Missing ticket ID")
            if not DataValidator.validate_vin(ticket.vehicle_info.vin):
                issues['tickets'].append(f"Invalid VIN in ticket {ticket.ticket_id}")
            if not DataValidator.validate_email(ticket.owner_info.email):
                issues['tickets'].append(f"Invalid email in ticket {ticket.ticket_id}")
        
        return issues 