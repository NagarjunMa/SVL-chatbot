"""
Comprehensive Observability and Tracing System for SVL Chatbot
Tracks every aspect of the application flow from user request to final response
"""

import json
import time
import uuid
import boto3
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
from functools import wraps
import logging
from botocore.exceptions import ClientError
from utils.pii_masker import pii_masker

# AWS Clients
cloudwatch_logs = boto3.client('logs', region_name='us-east-1')
cloudwatch_metrics = boto3.client('cloudwatch', region_name='us-east-1')
xray = boto3.client('xray', region_name='us-east-1')

class TraceContext:
    """Context manager for distributed tracing"""
    
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())
        self.parent_span_id = None
        self.start_time = None
        self.end_time = None
        self.metadata = {}
        self.spans = []
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the current trace"""
        self.metadata[key] = value
    
    def create_child_span(self, operation_name: str):
        """Create a child span for sub-operations"""
        child_span = {
            'span_id': str(uuid.uuid4()),
            'parent_span_id': self.span_id,
            'operation_name': operation_name,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'metadata': {}
        }
        self.spans.append(child_span)
        return child_span

class SVLObservability:
    """Main observability class for SVL Chatbot"""
    
    def __init__(self):
        """Initialize observability with CloudWatch clients"""
        try:
            self.cloudwatch_logs = boto3.client('logs', region_name='us-east-1')
            self.cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
            self.active_traces = {}
            self.current_trace = None
            self.service_name = "svl-chatbot"
            
            # Production CloudWatch log groups
            self.log_groups = {
                'api_requests': '/aws/svl/api-requests',
                'bedrock_operations': '/aws/svl/bedrock-operations',
                'knowledge_base': '/aws/svl/knowledge-base',
                'vector_search': '/aws/svl/vector-search',
                'ingestion_pipeline': '/aws/svl/ingestion-pipeline',
                'performance': '/aws/svl/performance',
                'errors': '/aws/svl/errors',
                'security': '/aws/svl/security',
                'conversation': '/aws/svl/conversation',
                'guardrails': '/aws/svl/guardrails'
            }
            
            # Initialize counters for status tracking
            self._logs_sent_count = 0
            self._errors_logged_count = 0
            
        except Exception as e:
            print(f"Warning: CloudWatch initialization failed: {e}")
            self.cloudwatch_logs = None
            self.cloudwatch = None
    
    def _ensure_log_groups(self):
        """Ensure all required log groups exist"""
        log_groups = [
            f"/aws/lambda/{self.service_name}",
            f"/aws/lambda/{self.service_name}/api-requests",
            f"/aws/lambda/{self.service_name}/bedrock-operations",
            f"/aws/lambda/{self.service_name}/knowledge-base",
            f"/aws/lambda/{self.service_name}/vector-search",
            f"/aws/lambda/{self.service_name}/ingestion-pipeline",
            f"/aws/lambda/{self.service_name}/performance",
            f"/aws/lambda/{self.service_name}/errors"
        ]
        
        for log_group in log_groups:
            try:
                self.cloudwatch_logs.create_log_group(logGroupName=log_group)
                # Set retention policy (30 days)
                self.cloudwatch_logs.put_retention_policy(
                    logGroupName=log_group,
                    retentionInDays=30
                )
            except self.cloudwatch_logs.exceptions.ResourceAlreadyExistsException:
                pass  # Log group already exists
            except Exception as e:
                print(f"Error creating log group {log_group}: {e}")
    
    def start_trace(self, operation_name: str, user_id: str = None, session_id: str = None) -> TraceContext:
        """Start a new distributed trace"""
        trace = TraceContext()
        trace.operation_name = operation_name
        trace.start_time = datetime.now(timezone.utc)
        trace.add_metadata('service', self.service_name)
        trace.add_metadata('operation', operation_name)
        
        if user_id:
            trace.add_metadata('user_id', user_id)
        if session_id:
            trace.add_metadata('session_id', session_id)
        
        self.current_trace = trace
        
        # Log trace start
        self._log_structured({
            'event_type': 'trace_start',
            'trace_id': trace.trace_id,
            'operation_name': operation_name,
            'timestamp': trace.start_time.isoformat(),
            'metadata': trace.metadata
        }, log_group='/aws/lambda/svl-chatbot/api-requests')
        
        return trace
    
    def end_trace(self, trace: TraceContext, status: str = 'success', error: str = None):
        """End a distributed trace"""
        trace.end_time = datetime.now(timezone.utc)
        duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000
        
        # Log trace end
        self._log_structured({
            'event_type': 'trace_end',
            'trace_id': trace.trace_id,
            'operation_name': trace.operation_name,
            'duration_ms': duration_ms,
            'status': status,
            'error': error,
            'timestamp': trace.end_time.isoformat(),
            'spans_count': len(trace.spans),
            'metadata': trace.metadata
        }, log_group='/aws/lambda/svl-chatbot/api-requests')
        
        # Send performance metrics
        self._send_metric('RequestDuration', duration_ms, 'Milliseconds', 
                         dimensions=[('Operation', trace.operation_name)])
        self._send_metric('RequestCount', 1, 'Count', 
                         dimensions=[('Operation', trace.operation_name), ('Status', status)])
    
    @contextmanager
    def trace_operation(self, operation_name: str, **metadata):
        """Context manager for tracing operations"""
        if self.current_trace:
            span = self.current_trace.create_child_span(operation_name)
            span['metadata'].update(metadata)
            start_time = datetime.now(timezone.utc)
            
            try:
                yield span
                span['status'] = 'success'
            except Exception as e:
                span['status'] = 'error'
                span['error'] = str(e)
                span['error_type'] = type(e).__name__
                raise
            finally:
                end_time = datetime.now(timezone.utc)
                span['end_time'] = end_time.isoformat()
                span['duration_ms'] = (end_time - start_time).total_seconds() * 1000
                
                # Log span
                self._log_structured({
                    'event_type': 'span',
                    'trace_id': self.current_trace.trace_id,
                    'span_id': span['span_id'],
                    'parent_span_id': span['parent_span_id'],
                    'operation_name': operation_name,
                    'duration_ms': span['duration_ms'],
                    'status': span['status'],
                    'metadata': span['metadata'],
                    'timestamp': start_time.isoformat()
                }, log_group='/aws/lambda/svl-chatbot/performance')
        else:
            # No active trace - just yield a dummy span for compatibility
            dummy_span = {
                'span_id': 'no-trace',
                'operation_name': operation_name,
                'metadata': metadata,
                'status': 'success'
            }
            try:
                yield dummy_span
            except Exception as e:
                dummy_span['status'] = 'error'
                dummy_span['error'] = str(e)
                raise
    
    def log_conversation(self, conversation_data: Dict[str, Any]):
        """Log ALL conversation activities to CloudWatch with PII masking"""
        try:
            # Apply PII masking to protect sensitive information
            masked_data = pii_masker.mask_conversation_data(conversation_data)
            
            conversation_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'conversation_id': masked_data.get('conversation_id'),
                'user_id': masked_data.get('user_id', 'anonymous'),
                'session_id': masked_data.get('session_id'),
                
                # Masked content (PII-safe)
                'user_message': masked_data.get('user_message', ''),
                'bot_response': masked_data.get('bot_response', ''),
                
                # Performance metrics
                'response_time_ms': masked_data.get('response_time_ms'),
                'message_length': len(conversation_data.get('user_message', '')),
                'response_length': len(conversation_data.get('bot_response', '')),
                
                # System behavior
                'guardrails_triggered': masked_data.get('guardrails_triggered', False),
                'knowledge_base_used': masked_data.get('knowledge_base_used', False),
                'bedrock_model_used': masked_data.get('bedrock_model_used', 'unknown'),
                
                # PII detection metrics (for security monitoring)
                'pii_detected_in_user_message': masked_data.get('user_message_pii_detected', False),
                'pii_types_detected': masked_data.get('user_message_pii_types', []),
                'bot_response_pii_warning': masked_data.get('bot_response_pii_warning', False),
                
                # Analytics metadata (PII-safe)
                'query_type': self._classify_query_type(conversation_data.get('user_message', '')),
                'conversation_phase': masked_data.get('conversation_phase', 'active'),
                'metadata': masked_data.get('metadata', {}),
                
                # Compliance and audit
                'log_level': 'INFO',
                'data_classification': 'PII_MASKED',
                'retention_policy': '30_days'
            }
            
            self._send_to_cloudwatch(self.log_groups['conversation'], conversation_log)
            self._increment_counter('logs_sent')
            
        except Exception as e:
            print(f"Failed to log conversation to CloudWatch: {e}")
            self.log_error(e, {'context': 'conversation_logging'})

    def _classify_query_type(self, user_message: str) -> str:
        """Classify the type of user query for analytics (PII-safe)"""
        if not user_message:
            return 'empty'
        
        message_lower = user_message.lower()
        
        # Query classification without exposing PII
        if any(word in message_lower for word in ['price', 'cost', 'fee', 'pricing', 'money', 'expensive', 'cheap']):
            return 'pricing_inquiry'
        elif any(word in message_lower for word in ['contact', 'phone', 'email', 'address', 'reach']):
            return 'contact_request'
        elif any(word in message_lower for word in ['report', 'stolen', 'theft', 'missing', 'recover']):
            return 'vehicle_report'
        elif any(word in message_lower for word in ['help', 'how', 'what', 'explain', 'process']):
            return 'information_request'
        elif any(word in message_lower for word in ['thank', 'thanks', 'goodbye', 'bye']):
            return 'social_interaction'
        elif len(message_lower) < 10:
            return 'short_query'
        else:
            return 'general_inquiry'

    def log_api_request(self, request_data: Dict[str, Any]):
        """Log API requests and responses"""
        try:
            api_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'method': request_data.get('method'),
                'endpoint': request_data.get('endpoint'),
                'user_id': request_data.get('user_id'),
                'session_id': request_data.get('session_id'),
                'response_time_ms': request_data.get('response_time_ms'),
                'status_code': request_data.get('status_code'),
                'request_size': request_data.get('request_size'),
                'response_size': request_data.get('response_size'),
                'ip_address': request_data.get('ip_address'),
                'user_agent': request_data.get('user_agent')
            }
            
            self._send_to_cloudwatch(self.log_groups['api_requests'], api_log)
            
        except Exception as e:
            print(f"Failed to log API request to CloudWatch: {e}")

    def log_guardrails_activity(self, guardrails_data: Dict[str, Any]):
        """Log guardrails and content filtering activities"""
        try:
            guardrails_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'conversation_id': guardrails_data.get('conversation_id'),
                'user_message': guardrails_data.get('user_message', ''),
                'filter_triggered': guardrails_data.get('filter_triggered', False),
                'filter_type': guardrails_data.get('filter_type'),
                'confidence_score': guardrails_data.get('confidence_score'),
                'action_taken': guardrails_data.get('action_taken'),
                'blocked_content': guardrails_data.get('blocked_content', False)
            }
            
            self._send_to_cloudwatch(self.log_groups['guardrails'], guardrails_log)
            
        except Exception as e:
            print(f"Failed to log guardrails activity to CloudWatch: {e}")
    
    def log_bedrock_operation(self, operation_type: str, model_id: str, input_data: Dict[str, Any], 
                            output_data: Dict[str, Any] = None, metrics: Dict[str, Any] = None):
        """Log Bedrock operations (embeddings, LLM calls, etc.)"""
        log_data = {
            'event_type': 'bedrock_operation',
            'trace_id': self.current_trace.trace_id if self.current_trace else None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation_type': operation_type,  # 'embedding', 'llm_invoke', 'knowledge_base_query'
            'model_id': model_id,
            'input_data': self._sanitize_input_data(input_data),
            'metrics': metrics or {}
        }
        
        if output_data:
            log_data['output_data'] = self._sanitize_output_data(output_data)
        
        self._log_structured(log_data, log_group='/aws/lambda/svl-chatbot/bedrock-operations')
    
    def log_knowledge_base_operation(self, operation_type: str, knowledge_base_id: str, 
                                   query: str, results: List[Dict], metrics: Dict[str, Any]):
        """Log Knowledge Base operations"""
        self._log_structured({
            'event_type': 'knowledge_base_operation',
            'trace_id': self.current_trace.trace_id if self.current_trace else None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation_type': operation_type,  # 'query', 'retrieve_and_generate'
            'knowledge_base_id': knowledge_base_id,
            'query': query[:500],  # Truncate long queries
            'results_count': len(results),
            'results': [self._sanitize_kb_result(result) for result in results[:3]],  # Log top 3 results
            'metrics': metrics
        }, log_group='/aws/lambda/svl-chatbot/knowledge-base')
    
    def log_vector_search(self, search_type: str, query_vector: List[float], 
                         search_params: Dict, results: List[Dict], opensearch_metrics: Dict):
        """Log vector search operations in OpenSearch"""
        self._log_structured({
            'event_type': 'vector_search',
            'trace_id': self.current_trace.trace_id if self.current_trace else None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'search_type': search_type,  # 'similarity_search', 'hybrid_search'
            'vector_dimension': len(query_vector),
            'query_vector_norm': sum(x*x for x in query_vector)**0.5,
            'search_params': search_params,
            'results_count': len(results),
            'results': [self._sanitize_vector_result(result) for result in results[:5]],
            'opensearch_metrics': opensearch_metrics
        }, log_group='/aws/lambda/svl-chatbot/vector-search')
    
    def log_ingestion_pipeline(self, stage: str, document_info: Dict, metrics: Dict[str, Any]):
        """Log document ingestion pipeline stages"""
        self._log_structured({
            'event_type': 'ingestion_pipeline',
            'trace_id': self.current_trace.trace_id if self.current_trace else None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'stage': stage,  # 's3_retrieval', 'text_extraction', 'chunking', 'embedding_generation', 'vector_storage'
            'document_info': document_info,
            'metrics': metrics
        }, log_group='/aws/lambda/svl-chatbot/ingestion-pipeline')
    
    def log_guardrails_activation(self, guardrail_id: str, input_text: str, 
                                action: str, reason: str, confidence: float):
        """Log Bedrock Guardrails activations"""
        self._log_structured({
            'event_type': 'guardrails_activation',
            'trace_id': self.current_trace.trace_id if self.current_trace else None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'guardrail_id': guardrail_id,
            'input_text': input_text[:200],  # Truncate for privacy
            'action': action,  # 'blocked', 'allowed', 'filtered'
            'reason': reason,
            'confidence': confidence
        }, log_group='/aws/lambda/svl-chatbot/bedrock-operations')
        
        # Send guardrails metrics
        self._send_metric('GuardrailsActivation', 1, 'Count', 
                         dimensions=[('Action', action), ('GuardrailId', guardrail_id)])
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log detailed error information"""
        try:
            self._increment_counter('errors_logged')
            
            error_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc(),
                'context': context or {},
                'severity': 'error'
            }
            
            self._send_to_cloudwatch('/aws/svl/errors', error_data)
            
        except Exception as e:
            print(f"Failed to log error to CloudWatch: {e}")
            print(f"Original error: {error}")
            print(f"Context: {context}")
    
    def _log_structured(self, data: Dict[str, Any], log_group: str):
        """Send structured log to CloudWatch"""
        try:
            log_stream = f"{datetime.now().strftime('%Y/%m/%d')}/[{self.service_name}]"
            
            # Create log stream if it doesn't exist
            try:
                self.cloudwatch_logs.create_log_stream(
                    logGroupName=log_group,
                    logStreamName=log_stream
                )
            except self.cloudwatch_logs.exceptions.ResourceAlreadyExistsException:
                pass
            
            # Send log event
            self.cloudwatch_logs.put_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                logEvents=[{
                    'timestamp': int(time.time() * 1000),
                    'message': json.dumps(data, default=str, ensure_ascii=False)
                }]
            )
        except Exception as e:
            # Fallback to local logging if CloudWatch fails
            print(f"CloudWatch logging failed: {e}")
            print(f"Log data: {json.dumps(data, default=str, indent=2)}")
    
    def _send_metric(self, metric_name: str, value: float, unit: str, 
                    dimensions: List[tuple] = None):
        """Send custom metrics to CloudWatch"""
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.now(timezone.utc)
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': name, 'Value': value} for name, value in dimensions
                ]
            
            cloudwatch_metrics.put_metric_data(
                Namespace=f'SVL/{self.service_name}',
                MetricData=[metric_data]
            )
        except Exception as e:
            print(f"CloudWatch metrics failed: {e}")
    
    def _sanitize_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data for logging"""
        sanitized = data.copy()
        
        # Remove sensitive headers
        if 'headers' in sanitized:
            sensitive_headers = ['authorization', 'cookie', 'x-api-key']
            for header in sensitive_headers:
                if header in sanitized['headers']:
                    sanitized['headers'][header] = '[REDACTED]'
        
        # Limit body size
        if 'body' in sanitized and sanitized['body']:
            if len(sanitized['body']) > 1000:
                sanitized['body'] = sanitized['body'][:1000] + '...[TRUNCATED]'
        
        return sanitized
    
    def _sanitize_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data for Bedrock operations"""
        sanitized = {}
        
        for key, value in data.items():
            if key in ['prompt', 'text', 'query']:
                # Truncate long text for logging
                sanitized[key] = value[:500] + ('...' if len(value) > 500 else '')
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_output_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize output data from Bedrock operations"""
        sanitized = {}
        
        for key, value in data.items():
            if key in ['text', 'response', 'completion']:
                # Truncate long responses
                sanitized[key] = value[:1000] + ('...' if len(value) > 1000 else '')
            elif key == 'embedding':
                # Just log embedding metadata
                sanitized[key] = {
                    'dimension': len(value) if isinstance(value, list) else 'unknown',
                    'norm': sum(x*x for x in value)**0.5 if isinstance(value, list) else None
                }
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_kb_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize knowledge base results"""
        return {
            'score': result.get('score'),
            'source': result.get('source'),
            'text_preview': result.get('text', '')[:200] + ('...' if len(result.get('text', '')) > 200 else ''),
            'metadata': result.get('metadata', {})
        }
    
    def _sanitize_vector_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize vector search results"""
        return {
            'score': result.get('_score'),
            'id': result.get('_id'),
            'source': result.get('_source', {}).get('AMAZON_BEDROCK_METADATA', {}).get('source'),
            'text_preview': str(result.get('_source', {}).get('AMAZON_BEDROCK_TEXT_CHUNK', ''))[:200]
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current observability system status"""
        try:
            return {
                "active": True,
                "active_traces": len(self.active_traces),
                "logs_sent": getattr(self, '_logs_sent_count', 0),
                "errors_logged": getattr(self, '_errors_logged_count', 0),
                "cloudwatch_groups": [
                    "/aws/svl/api-requests",
                    "/aws/svl/bedrock-operations", 
                    "/aws/svl/knowledge-base",
                    "/aws/svl/vector-search",
                    "/aws/svl/ingestion-pipeline",
                    "/aws/svl/performance",
                    "/aws/svl/errors",
                    "/aws/svl/general"
                ],
                "region": "us-east-1",
                "dashboard_url": "https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=SVL-Observability-Dashboard"
            }
        except Exception as e:
            return {
                "active": False,
                "error": str(e),
                "message": "Observability system not properly initialized"
            }
    
    def _increment_counter(self, counter_name: str):
        """Increment internal counters for status tracking"""
        current_count = getattr(self, f'_{counter_name}_count', 0)
        setattr(self, f'_{counter_name}_count', current_count + 1)

    def _send_to_cloudwatch(self, log_group_name: str, log_data: Dict[str, Any]):
        """Send log data to CloudWatch"""
        try:
            if not self.cloudwatch_logs:
                print(f"CloudWatch not initialized, logging locally: {log_data}")
                return
            
            # Create log stream name with timestamp
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H')
            stream_name = f"svl-chatbot-{timestamp}"
            
            # Ensure log stream exists
            try:
                self.cloudwatch_logs.create_log_stream(
                    logGroupName=log_group_name,
                    logStreamName=stream_name
                )
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                    print(f"Error creating log stream: {e}")
                    return
            
            # Prepare log event
            log_event = {
                'timestamp': int(time.time() * 1000),
                'message': json.dumps(log_data, default=str, ensure_ascii=False)
            }
            
            # Send to CloudWatch
            self.cloudwatch_logs.put_log_events(
                logGroupName=log_group_name,
                logStreamName=stream_name,
                logEvents=[log_event]
            )
            
            # Increment counter
            self._increment_counter('logs_sent')
            
        except Exception as e:
            print(f"Failed to send to CloudWatch {log_group_name}: {e}")
            print(f"Log data: {log_data}")

# Global observability instance
observability = SVLObservability()

# Decorator for automatic function tracing
def trace_function(operation_name: str = None):
    """Decorator to automatically trace function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with observability.trace_operation(op_name) as span:
                span['metadata']['function'] = func.__name__
                span['metadata']['module'] = func.__module__
                span['metadata']['args_count'] = len(args)
                span['metadata']['kwargs_keys'] = list(kwargs.keys())
                
                try:
                    result = func(*args, **kwargs)
                    span['metadata']['result_type'] = type(result).__name__
                    return result
                except Exception as e:
                    span['metadata']['error'] = str(e)
                    observability.log_error(e, {
                        'function': func.__name__,
                        'module': func.__module__,
                        'operation': op_name
                    })
                    raise
        
        return wrapper
    return decorator 