#!/usr/bin/env python3
"""
PII Masking Utility for SVL Chatbot Logging
Masks personally identifiable information in logs while preserving analytical value
"""

import re
from typing import Dict, Any, Union

class PIIMasker:
    """Comprehensive PII detection and masking for logs"""
    
    def __init__(self):
        # Define PII patterns
        self.pii_patterns = {
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            'driver_license': r'\b[A-Z]{1,2}[0-9]{6,8}\b',
            'vin': r'\b[A-HJ-NPR-Z0-9]{17}\b',
            'license_plate': r'\b[A-Z0-9]{2,8}\b(?=\s*(?:plate|license|tag))|(?:plate|license|tag)\s*[A-Z0-9]{2,8}\b',
            'address': r'\b\d+\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct)\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
            'birth_date': r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b|\b(?:19|20)\d{2}[/-](?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])\b',
            'bank_account': r'\b\d{8,17}\b',
            'passport': r'\b[A-Z]{1,2}\d{7,9}\b',
            'insurance_policy': r'\b[A-Z]{2,4}\d{6,12}\b'
        }
        
        # Common name patterns (basic detection)
        self.name_patterns = [
            r'\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\bi am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\bcalled\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
    
    def mask_pii(self, text: str, preserve_analytics: bool = True) -> Dict[str, Any]:
        """
        Mask PII in text while preserving analytical value
        
        Args:
            text: Input text to mask
            preserve_analytics: Whether to preserve partial info for analytics
            
        Returns:
            Dict with masked_text, detected_pii_types, and analytics_data
        """
        if not text or not isinstance(text, str):
            return {
                'masked_text': text,
                'detected_pii_types': [],
                'analytics_data': {},
                'pii_detected': False
            }
        
        masked_text = text
        detected_types = []
        analytics_data = {}
        
        # Mask each PII type
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, masked_text, re.IGNORECASE)
            match_count = 0
            
            for match in matches:
                match_count += 1
                original = match.group(0)
                
                if preserve_analytics:
                    # Create analytics-friendly mask that preserves some structure
                    masked_value = self._create_analytical_mask(original, pii_type)
                else:
                    # Complete masking
                    masked_value = f"[{pii_type.upper()}_REDACTED]"
                
                masked_text = masked_text.replace(original, masked_value)
                
                if pii_type not in detected_types:
                    detected_types.append(pii_type)
            
            if match_count > 0:
                analytics_data[pii_type] = {
                    'count': match_count,
                    'type': pii_type
                }
        
        # Mask potential names
        for pattern in self.name_patterns:
            matches = re.finditer(pattern, masked_text, re.IGNORECASE)
            for match in matches:
                name = match.group(1)
                if len(name.split()) >= 2:  # Likely full name
                    if preserve_analytics:
                        masked_name = f"[NAME_{len(name.split())}_WORDS]"
                    else:
                        masked_name = "[NAME_REDACTED]"
                    
                    masked_text = masked_text.replace(match.group(0), 
                                                    match.group(0).replace(name, masked_name))
                    
                    if 'name' not in detected_types:
                        detected_types.append('name')
        
        return {
            'masked_text': masked_text,
            'detected_pii_types': detected_types,
            'analytics_data': analytics_data,
            'pii_detected': len(detected_types) > 0,
            'original_length': len(text),
            'masked_length': len(masked_text)
        }
    
    def _create_analytical_mask(self, original: str, pii_type: str) -> str:
        """Create analytics-friendly masks that preserve some structure"""
        
        if pii_type == 'ssn':
            return "***-**-****"
        elif pii_type == 'phone':
            return "***-***-****"
        elif pii_type == 'email':
            parts = original.split('@')
            if len(parts) == 2:
                return f"***@{parts[1]}"
            return "[EMAIL_REDACTED]"
        elif pii_type == 'credit_card':
            return "****-****-****-****"
        elif pii_type == 'vin':
            return f"{original[:3]}***********{original[-3:]}"
        elif pii_type == 'license_plate':
            return "***" + original[-2:] if len(original) > 2 else "***"
        elif pii_type == 'address':
            return f"[ADDRESS_{len(original.split())}_WORDS]"
        elif pii_type == 'zip_code':
            return original[:2] + "***"
        elif pii_type == 'birth_date':
            return "**/**/****"
        else:
            return f"[{pii_type.upper()}_REDACTED]"
    
    def mask_conversation_data(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask PII in conversation data structure"""
        masked_data = conversation_data.copy()
        
        # Mask user message
        if 'user_message' in masked_data:
            mask_result = self.mask_pii(masked_data['user_message'])
            masked_data['user_message'] = mask_result['masked_text']
            masked_data['user_message_pii_detected'] = mask_result['pii_detected']
            masked_data['user_message_pii_types'] = mask_result['detected_pii_types']
        
        # Mask bot response (shouldn't contain PII, but safety check)
        if 'bot_response' in masked_data:
            mask_result = self.mask_pii(masked_data['bot_response'])
            masked_data['bot_response'] = mask_result['masked_text']
            if mask_result['pii_detected']:
                masked_data['bot_response_pii_warning'] = True
        
        # Mask any other text fields
        text_fields = ['user_input', 'message', 'content', 'query']
        for field in text_fields:
            if field in masked_data and isinstance(masked_data[field], str):
                mask_result = self.mask_pii(masked_data[field])
                masked_data[field] = mask_result['masked_text']
                if mask_result['pii_detected']:
                    masked_data[f'{field}_pii_detected'] = True
                    masked_data[f'{field}_pii_types'] = mask_result['detected_pii_types']
        
        return masked_data

# Global PII masker instance
pii_masker = PIIMasker() 