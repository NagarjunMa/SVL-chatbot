import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# DynamoDB Configuration
DYNAMODB_CONFIG = {
    "region_name": os.environ.get("AWS_REGION", "us-east-1"),
    "endpoint_url": os.environ.get("DYNAMODB_ENDPOINT_URL"),  # For local development
}

# Table Schemas
TABLES = {
    "conversations": {
        "TableName": "svl-conversations",
        "KeySchema": [
            {"AttributeName": "conversation_id", "KeyType": "HASH"},
            {"AttributeName": "timestamp", "KeyType": "RANGE"}
        ],
        "AttributeDefinitions": [
            {"AttributeName": "conversation_id", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "S"},
            {"AttributeName": "user_id", "AttributeType": "S"}
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "user_id-index",
                "KeySchema": [
                    {"AttributeName": "user_id", "KeyType": "HASH"},
                    {"AttributeName": "timestamp", "KeyType": "RANGE"}
                ],
                "Projection": {"ProjectionType": "ALL"}
            }
        ],
        "BillingMode": "PAY_PER_REQUEST"
    },
    "tickets": {
        "TableName": "svl-tickets",
        "KeySchema": [
            {"AttributeName": "ticket_id", "KeyType": "HASH"},
            {"AttributeName": "created_at", "KeyType": "RANGE"}
        ],
        "AttributeDefinitions": [
            {"AttributeName": "ticket_id", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"}
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "user_id-index",
                "KeySchema": [
                    {"AttributeName": "user_id", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"}
                ],
                "Projection": {"ProjectionType": "ALL"}
            },
            {
                "IndexName": "status-index",
                "KeySchema": [
                    {"AttributeName": "status", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"}
                ],
                "Projection": {"ProjectionType": "ALL"}
            }
        ],
        "BillingMode": "PAY_PER_REQUEST"
    }
}

# Data Validation Rules
VALIDATION_RULES = {
    "ticket_id_format": r"^SVL-\d{8}-\d{6}-\d{3}$",
    "max_message_length": 10000,
    "max_conversation_turns": 100,
    "allowed_statuses": ["open", "in_progress", "resolved", "closed"]
}

# Encryption Configuration
ENCRYPTION_CONFIG = {
    "algorithm": "AES-256-GCM",
    "key_derivation": "PBKDF2",
    "iterations": 100000
}

# Audit Configuration
AUDIT_CONFIG = {
    "log_all_operations": True,
    "retention_days": 90,
    "sensitive_fields": ["phone", "email", "address", "vin"]
} 