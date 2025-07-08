"""
Create CloudWatch Dashboard for SVL Chatbot Observability
Comprehensive monitoring dashboard for all application components
"""

import boto3
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_svl_monitoring_dashboard():
    """Create comprehensive CloudWatch dashboard for SVL chatbot monitoring"""
    
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    
    # Dashboard configuration
    dashboard_body = {
        "widgets": [
            # API Request Metrics Row
            {
                "type": "metric",
                "x": 0, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        [ "SVL/svl-chatbot", "RequestCount", "Operation", "conversation_request", "Status", "success" ],
                        [ "...", "error" ],
                        [ "...", "Operation", "api_chat", "Status", "success" ],
                        [ "...", "error" ]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "API Request Counts",
                    "period": 300,
                    "stat": "Sum"
                }
            },
            {
                "type": "metric",
                "x": 12, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        [ "SVL/svl-chatbot", "RequestDuration", "Operation", "conversation_request" ],
                        [ "...", "api_chat" ],
                        [ "...", "bedrock_kb_query" ],
                        [ "...", "bedrock_response_generation" ]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Response Times (ms)",
                    "period": 300,
                    "stat": "Average"
                }
            },
            
            # Bedrock Operations Row
            {
                "type": "metric", 
                "x": 0, "y": 6, "width": 8, "height": 6,
                "properties": {
                    "metrics": [
                        [ "SVL/svl-chatbot", "RequestCount", "Operation", "bedrock_llm_invoke", "Status", "success" ],
                        [ "...", "error" ],
                        [ "...", "Operation", "bedrock_embedding", "Status", "success" ],
                        [ "...", "error" ]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Bedrock Operations",
                    "period": 300,
                    "stat": "Sum"
                }
            },
            {
                "type": "metric",
                "x": 8, "y": 6, "width": 8, "height": 6,
                "properties": {
                    "metrics": [
                        [ "SVL/svl-chatbot", "GuardrailsActivation", "Action", "blocked" ],
                        [ "...", "allowed" ],
                        [ "...", "filtered" ]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Guardrails Activations",
                    "period": 300,
                    "stat": "Sum"
                }
            },
            {
                "type": "metric",
                "x": 16, "y": 6, "width": 8, "height": 6,
                "properties": {
                    "metrics": [
                        [ "SVL/svl-chatbot", "ErrorCount", "ErrorType", "ConnectionError" ],
                        [ "...", "ValueError" ],
                        [ "...", "TimeoutError" ],
                        [ "...", "Exception" ]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Error Counts by Type",
                    "period": 300,
                    "stat": "Sum"
                }
            },
            
            # Knowledge Base and Vector Search Row
            {
                "type": "metric",
                "x": 0, "y": 12, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        [ "AWS/Bedrock", "RequestCount", "ModelId", "amazon.nova-pro-v1:0" ],
                        [ "AWS/Bedrock", "ResponseTime", "ModelId", "amazon.nova-pro-v1:0" ]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Bedrock Model Usage",
                    "period": 300
                }
            },
            {
                "type": "metric",
                "x": 12, "y": 12, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        [ "AWS/ES", "SearchRate", "DomainName", "svl-chatbot-vector-search", "ClientId", "AWS_ACCOUNT_ID" ],
                        [ "AWS/ES", "SearchLatency", "DomainName", "svl-chatbot-vector-search", "ClientId", "AWS_ACCOUNT_ID" ]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "OpenSearch Performance",
                    "period": 300
                }
            },
            
            # Logs Insights Queries Row
            {
                "type": "log",
                "x": 0, "y": 18, "width": 12, "height": 6,
                "properties": {
                    "query": "SOURCE '/aws/lambda/svl-chatbot/api-requests' | fields @timestamp, trace_id, operation_name, duration_ms, status\n| filter event_type = \"trace_end\"\n| sort @timestamp desc\n| limit 20",
                    "region": "us-east-1",
                    "title": "Recent API Traces",
                    "view": "table"
                }
            },
            {
                "type": "log",
                "x": 12, "y": 18, "width": 12, "height": 6,
                "properties": {
                    "query": "SOURCE '/aws/lambda/svl-chatbot/errors' | fields @timestamp, error_type, error_message, context\n| filter event_type = \"error\"\n| sort @timestamp desc\n| limit 20",
                    "region": "us-east-1",
                    "title": "Recent Errors",
                    "view": "table"
                }
            },
            
            # Bedrock Operations Detail Row
            {
                "type": "log",
                "x": 0, "y": 24, "width": 8, "height": 6,
                "properties": {
                    "query": "SOURCE '/aws/lambda/svl-chatbot/bedrock-operations' | fields @timestamp, operation_type, model_id, metrics.duration_seconds\n| filter event_type = \"bedrock_operation\"\n| stats avg(metrics.duration_seconds) by operation_type\n| sort avg desc",
                    "region": "us-east-1", 
                    "title": "Bedrock Operation Performance",
                    "view": "table"
                }
            },
            {
                "type": "log",
                "x": 8, "y": 24, "width": 8, "height": 6,
                "properties": {
                    "query": "SOURCE '/aws/lambda/svl-chatbot/knowledge-base' | fields @timestamp, operation_type, results_count, metrics.duration_seconds\n| filter event_type = \"knowledge_base_operation\"\n| stats avg(results_count) as avg_results, avg(metrics.duration_seconds) as avg_duration by operation_type",
                    "region": "us-east-1",
                    "title": "Knowledge Base Performance",
                    "view": "table"
                }
            },
            {
                "type": "log",
                "x": 16, "y": 24, "width": 8, "height": 6,
                "properties": {
                    "query": "SOURCE '/aws/lambda/svl-chatbot/vector-search' | fields @timestamp, search_type, results_count, opensearch_metrics.average_score\n| filter event_type = \"vector_search\"\n| stats avg(results_count) as avg_results, avg(opensearch_metrics.average_score) as avg_score by search_type",
                    "region": "us-east-1",
                    "title": "Vector Search Analytics",
                    "view": "table"
                }
            },
            
            # Ingestion Pipeline Row
            {
                "type": "log",
                "x": 0, "y": 30, "width": 12, "height": 6,
                "properties": {
                    "query": "SOURCE '/aws/lambda/svl-chatbot/ingestion-pipeline' | fields @timestamp, stage, document_info, metrics\n| filter event_type = \"ingestion_pipeline\"\n| sort @timestamp desc\n| limit 20",
                    "region": "us-east-1",
                    "title": "Recent Ingestion Pipeline Activity",
                    "view": "table"
                }
            },
            {
                "type": "log",
                "x": 12, "y": 30, "width": 12, "height": 6,
                "properties": {
                    "query": "SOURCE '/aws/lambda/svl-chatbot/performance' | fields @timestamp, operation_name, duration_ms, status, metadata\n| filter event_type = \"span\"\n| stats avg(duration_ms) as avg_duration, count() as operations by operation_name\n| sort avg_duration desc",
                    "region": "us-east-1",
                    "title": "Operation Performance Breakdown",
                    "view": "table"
                }
            }
        ]
    }
    
    try:
        # Create the dashboard
        response = cloudwatch.put_dashboard(
            DashboardName='SVL-Chatbot-Comprehensive-Monitoring',
            DashboardBody=json.dumps(dashboard_body)
        )
        
        logger.info("✅ CloudWatch dashboard created successfully!")
        logger.info(f"Dashboard response: {response}")
        
        # Create additional alarms
        create_monitoring_alarms(cloudwatch)
        
        return "dashboard_created_successfully"
        
    except Exception as e:
        logger.error(f"❌ Failed to create dashboard: {e}")
        raise

def create_monitoring_alarms(cloudwatch):
    """Create CloudWatch alarms for critical metrics"""
    
    alarms = [
        {
            'AlarmName': 'SVL-Chatbot-High-Error-Rate',
            'ComparisonOperator': 'GreaterThanThreshold',
            'EvaluationPeriods': 2,
            'MetricName': 'ErrorCount',
            'Namespace': 'SVL/svl-chatbot',
            'Period': 300,
            'Statistic': 'Sum',
            'Threshold': 10.0,
            'ActionsEnabled': True,
            'AlarmDescription': 'SVL Chatbot error rate is too high',
            'Unit': 'Count'
        },
        {
            'AlarmName': 'SVL-Chatbot-High-Response-Time',
            'ComparisonOperator': 'GreaterThanThreshold',
            'EvaluationPeriods': 2,
            'MetricName': 'RequestDuration',
            'Namespace': 'SVL/svl-chatbot',
            'Period': 300,
            'Statistic': 'Average',
            'Threshold': 10000.0,  # 10 seconds
            'ActionsEnabled': True,
            'AlarmDescription': 'SVL Chatbot response time is too high',
            'Unit': 'Milliseconds',
            'Dimensions': [
                {
                    'Name': 'Operation',
                    'Value': 'conversation_request'
                }
            ]
        },
        {
            'AlarmName': 'SVL-Chatbot-Guardrails-High-Block-Rate',
            'ComparisonOperator': 'GreaterThanThreshold',
            'EvaluationPeriods': 1,
            'MetricName': 'GuardrailsActivation',
            'Namespace': 'SVL/svl-chatbot',
            'Period': 300,
            'Statistic': 'Sum',
            'Threshold': 50.0,
            'ActionsEnabled': True,
            'AlarmDescription': 'SVL Chatbot guardrails blocking too many requests',
            'Unit': 'Count',
            'Dimensions': [
                {
                    'Name': 'Action',
                    'Value': 'blocked'
                }
            ]
        }
    ]
    
    for alarm_config in alarms:
        try:
            cloudwatch.put_metric_alarm(**alarm_config)
            logger.info(f"✅ Created alarm: {alarm_config['AlarmName']}")
        except Exception as e:
            logger.error(f"❌ Failed to create alarm {alarm_config['AlarmName']}: {e}")

def create_log_groups():
    """Ensure all required log groups exist"""
    
    cloudwatch_logs = boto3.client('logs', region_name='us-east-1')
    
    log_groups = [
        '/aws/lambda/svl-chatbot',
        '/aws/lambda/svl-chatbot/api-requests',
        '/aws/lambda/svl-chatbot/bedrock-operations',
        '/aws/lambda/svl-chatbot/knowledge-base',
        '/aws/lambda/svl-chatbot/vector-search',
        '/aws/lambda/svl-chatbot/ingestion-pipeline',
        '/aws/lambda/svl-chatbot/performance',
        '/aws/lambda/svl-chatbot/errors'
    ]
    
    for log_group in log_groups:
        try:
            cloudwatch_logs.create_log_group(logGroupName=log_group)
            # Set retention policy (30 days)
            cloudwatch_logs.put_retention_policy(
                logGroupName=log_group,
                retentionInDays=30
            )
            logger.info(f"✅ Created log group: {log_group}")
        except cloudwatch_logs.exceptions.ResourceAlreadyExistsException:
            logger.info(f"📋 Log group already exists: {log_group}")
        except Exception as e:
            logger.error(f"❌ Failed to create log group {log_group}: {e}")

def create_custom_metrics_filter():
    """Create CloudWatch metrics filters for custom metrics"""
    
    cloudwatch_logs = boto3.client('logs', region_name='us-east-1')
    
    # Metric filter for API response times
    try:
        cloudwatch_logs.put_metric_filter(
            logGroupName='/aws/lambda/svl-chatbot/api-requests',
            filterName='APIResponseTimeFilter',
            filterPattern='{ $.event_type = "trace_end" && $.duration_ms = * }',
            metricTransformations=[
                {
                    'metricName': 'APIResponseTime',
                    'metricNamespace': 'SVL/svl-chatbot/Custom',
                    'metricValue': '$.duration_ms',
                    'defaultValue': 0
                }
            ]
        )
        logger.info("✅ Created API response time metric filter")
    except Exception as e:
        logger.error(f"❌ Failed to create metric filter: {e}")

if __name__ == "__main__":
    logger.info("🔧 Setting up SVL Chatbot monitoring infrastructure...")
    
    try:
        # Create log groups
        logger.info("📁 Creating CloudWatch log groups...")
        create_log_groups()
        
        # Create metric filters
        logger.info("📊 Creating custom metric filters...")
        create_custom_metrics_filter()
        
        # Create dashboard
        logger.info("📈 Creating monitoring dashboard...")
        dashboard_arn = create_svl_monitoring_dashboard()
        
        logger.info("✅ SVL Chatbot monitoring setup complete!")
        logger.info(f"🌐 Dashboard URL: https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=SVL-Chatbot-Comprehensive-Monitoring")
        logger.info("\n📋 Monitoring Components Created:")
        logger.info("  - CloudWatch Dashboard: SVL-Chatbot-Comprehensive-Monitoring")
        logger.info("  - Log Groups: 8 specialized log groups")
        logger.info("  - Alarms: Error rate, response time, guardrails")
        logger.info("  - Custom Metrics: API performance tracking")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        raise 