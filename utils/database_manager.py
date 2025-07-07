import boto3
import json
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union
from botocore.exceptions import ClientError, NoCredentialsError
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

from config.database_config import DYNAMODB_CONFIG, TABLES, VALIDATION_RULES, ENCRYPTION_CONFIG, AUDIT_CONFIG
from data.models import Conversation, Ticket, Message, AuditLog, VehicleInfo, OwnerInfo, IncidentInfo, InsuranceInfo
from utils.logger import get_logger

logger = get_logger("database_manager")

class EncryptionManager:
    """Handles encryption/decryption of sensitive data"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        if encryption_key:
            self.key = base64.urlsafe_b64encode(encryption_key.encode())
        else:
            # Generate a key for development (use environment variable in production)
            self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

class DatabaseManager:
    """Manages all DynamoDB operations for the SVL chatbot"""
    
    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        try:
            self.dynamodb = boto3.resource('dynamodb', **DYNAMODB_CONFIG)
            self.client = boto3.client('dynamodb', **DYNAMODB_CONFIG)
            self.encryption_manager = encryption_manager or EncryptionManager()
            self._ensure_tables_exist()
            logger.info("DatabaseManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseManager: {e}")
            raise
    
    def _ensure_tables_exist(self):
        """Ensure required DynamoDB tables exist"""
        existing_tables = [table.name for table in self.dynamodb.tables.all()]
        for table_name, table_config in TABLES.items():
            table_name_actual = table_config["TableName"]
            if table_name_actual in existing_tables:
                # Check if we need to recreate due to schema changes
                try:
                    table = self.dynamodb.Table(table_name_actual)
                    table_description = table.meta.client.describe_table(TableName=table_name_actual)
                    
                    # Check for the old ticket_id index in conversations table
                    if table_name == "conversations":
                        gsi_names = []
                        if 'GlobalSecondaryIndexes' in table_description['Table']:
                            gsi_names = [gsi['IndexName'] for gsi in table_description['Table']['GlobalSecondaryIndexes']]
                        
                        # If old ticket_id index exists, delete and recreate table
                        if 'ticket_id-index' in gsi_names:
                            logger.info(f"Deleting table {table_name_actual} due to schema changes")
                            table.delete()
                            # Wait for deletion
                            waiter = self.client.get_waiter('table_not_exists')
                            waiter.wait(TableName=table_name_actual)
                            # Remove from existing_tables so it gets recreated
                            existing_tables.remove(table_name_actual)
                            
                except Exception as e:
                    logger.warning(f"Could not check table schema for {table_name_actual}: {e}")
            
            if table_name_actual not in existing_tables:
                logger.info(f"Creating table: {table_name_actual}")
                self.dynamodb.create_table(**table_config)
                # Wait for table to be active
                waiter = self.client.get_waiter('table_exists')
                waiter.wait(TableName=table_name_actual)
    
    def generate_ticket_id(self) -> str:
        """Generate unique ticket ID in format: SVL-YYYYMMDD-HHMMSS-XXX"""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        sequence = str(uuid.uuid4().int % 1000).zfill(3)
        return f"SVL-{date_str}-{time_str}-{sequence}"
    
    def generate_conversation_id(self) -> str:
        """Generate unique conversation ID"""
        return f"conv-{uuid.uuid4().hex[:8]}"
    
    # --- Conversation Operations ---
    
    async def create_conversation(self, user_id: str) -> Conversation:
        """Create a new conversation"""
        try:
            conversation_id = self.generate_conversation_id()
            conversation = Conversation(
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            item = conversation.dict()
            # Convert datetime objects to ISO format strings for DynamoDB
            item['timestamp'] = conversation.created_at.isoformat()
            item['created_at'] = conversation.created_at.isoformat()
            item['updated_at'] = conversation.updated_at.isoformat()
            
            # Convert message timestamps
            for i, message in enumerate(item.get('messages', [])):
                if 'timestamp' in message and hasattr(message['timestamp'], 'isoformat'):
                    item['messages'][i]['timestamp'] = message['timestamp'].isoformat()
            
            table = self.dynamodb.Table(TABLES['conversations']['TableName'])
            table.put_item(Item=item)
            
            self._log_audit("CREATE", "conversations", conversation_id, user_id)
            logger.info(f"Created conversation: {conversation_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by ID"""
        try:
            table = self.dynamodb.Table(TABLES['conversations']['TableName'])
            # Query by conversation_id since timestamp is a range key
            response = table.query(
                KeyConditionExpression='conversation_id = :conv_id',
                ExpressionAttributeValues={':conv_id': conversation_id},
                Limit=1
            )
            
            if 'Items' in response and response['Items']:
                item = response['Items'][0]  # Get the first (and only) item
                # Convert timestamp strings back to datetime
                item['created_at'] = datetime.fromisoformat(item['timestamp'])
                item['updated_at'] = datetime.fromisoformat(item['updated_at'])
                for message in item.get('messages', []):
                    message['timestamp'] = datetime.fromisoformat(message['timestamp'])
                
                return Conversation(**item)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            raise
    
    async def add_message_to_conversation(self, conversation_id: str, message: Message) -> bool:
        """Add a message to an existing conversation"""
        try:
            table = self.dynamodb.Table(TABLES['conversations']['TableName'])
            
            # Get current conversation
            conversation = await self.get_conversation(conversation_id)
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            # Add message and update
            conversation.add_message(message)
            
            # Update in DynamoDB
            table.update_item(
                Key={
                    'conversation_id': conversation_id,
                    'timestamp': conversation.created_at.isoformat()
                },
                UpdateExpression='SET messages = :messages, updated_at = :updated_at',
                ExpressionAttributeValues={
                    ':messages': [msg.dict() for msg in conversation.messages],
                    ':updated_at': conversation.updated_at.isoformat()
                }
            )
            
            self._log_audit("UPDATE", "conversations", conversation_id, message.role)
            logger.info(f"Added message to conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
            raise
    
    async def get_user_conversations(self, user_id: str) -> List[Conversation]:
        """Get all conversations for a user"""
        try:
            table = self.dynamodb.Table(TABLES['conversations']['TableName'])
            response = table.query(
                IndexName='user_id-index',
                KeyConditionExpression='user_id = :user_id',
                ExpressionAttributeValues={':user_id': user_id}
            )
            
            conversations = []
            for item in response.get('Items', []):
                # Convert timestamps
                item['created_at'] = datetime.fromisoformat(item['timestamp'])
                item['updated_at'] = datetime.fromisoformat(item['updated_at'])
                conversations.append(Conversation(**item))
            
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to get conversations for user {user_id}: {e}")
            raise
    
    # --- Ticket Operations ---
    
    async def create_ticket(self, user_id: str, vehicle_info: VehicleInfo, 
                          owner_info: OwnerInfo, incident_info: IncidentInfo, 
                          insurance_info: InsuranceInfo) -> Ticket:
        """Create a new ticket"""
        try:
            ticket_id = self.generate_ticket_id()
            
            # Encrypt sensitive information
            encrypted_owner_info = OwnerInfo(
                name=owner_info.name,
                phone=self.encryption_manager.encrypt(owner_info.phone),
                email=self.encryption_manager.encrypt(owner_info.email),
                address=self.encryption_manager.encrypt(owner_info.address)
            )
            
            ticket = Ticket(
                ticket_id=ticket_id,
                user_id=user_id,
                vehicle_info=vehicle_info,
                owner_info=encrypted_owner_info,
                incident_info=incident_info,
                insurance_info=insurance_info
            )
            
            item = ticket.dict()
            item['created_at'] = ticket.created_at.isoformat()
            item['updated_at'] = ticket.updated_at.isoformat()
            
            table = self.dynamodb.Table(TABLES['tickets']['TableName'])
            table.put_item(Item=item)
            
            self._log_audit("CREATE", "tickets", ticket_id, user_id)
            logger.info(f"Created ticket: {ticket_id}")
            return ticket
            
        except Exception as e:
            logger.error(f"Failed to create ticket: {e}")
            raise
    
    async def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Retrieve a ticket by ID"""
        try:
            table = self.dynamodb.Table(TABLES['tickets']['TableName'])
            # Query by ticket_id since created_at is a range key
            response = table.query(
                KeyConditionExpression='ticket_id = :tid',
                ExpressionAttributeValues={':tid': ticket_id},
                Limit=1
            )
            
            if 'Items' in response and response['Items']:
                item = response['Items'][0]
                # Convert timestamps
                item['created_at'] = datetime.fromisoformat(item['created_at'])
                item['updated_at'] = datetime.fromisoformat(item['updated_at'])
                
                # Decrypt sensitive information
                if 'owner_info' in item:
                    owner_info = item['owner_info']
                    owner_info['phone'] = self.encryption_manager.decrypt(owner_info['phone'])
                    owner_info['email'] = self.encryption_manager.decrypt(owner_info['email'])
                    owner_info['address'] = self.encryption_manager.decrypt(owner_info['address'])
                
                return Ticket(**item)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get ticket {ticket_id}: {e}")
            raise
    
    async def update_ticket_status(self, ticket_id: str, status: str, assigned_to: Optional[str] = None) -> bool:
        """Update ticket status"""
        try:
            table = self.dynamodb.Table(TABLES['tickets']['TableName'])
            
            update_expression = 'SET #status = :status, updated_at = :updated_at'
            expression_attrs = {
                '#status': 'status',
                ':status': status,
                ':updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            if assigned_to:
                update_expression += ', assigned_to = :assigned_to'
                expression_attrs[':assigned_to'] = assigned_to
            
            # First get the ticket to get the created_at timestamp
            ticket = await self.get_ticket(ticket_id)
            if not ticket:
                raise ValueError(f"Ticket {ticket_id} not found")
            
            table.update_item(
                Key={
                    'ticket_id': ticket_id,
                    'created_at': ticket.created_at.isoformat()
                },
                UpdateExpression=update_expression,
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues=expression_attrs
            )
            
            self._log_audit("UPDATE", "tickets", ticket_id, assigned_to or "system")
            logger.info(f"Updated ticket {ticket_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update ticket {ticket_id}: {e}")
            raise
    
    async def get_tickets_by_status(self, status: str) -> List[Ticket]:
        """Get all tickets with a specific status"""
        try:
            table = self.dynamodb.Table(TABLES['tickets']['TableName'])
            response = table.query(
                IndexName='status-index',
                KeyConditionExpression='#status = :status',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':status': status}
            )
            
            tickets = []
            for item in response.get('Items', []):
                # Convert timestamps and decrypt sensitive data
                item['created_at'] = datetime.fromisoformat(item['created_at'])
                item['updated_at'] = datetime.fromisoformat(item['updated_at'])
                
                if 'owner_info' in item:
                    owner_info = item['owner_info']
                    owner_info['phone'] = self.encryption_manager.decrypt(owner_info['phone'])
                    owner_info['email'] = self.encryption_manager.decrypt(owner_info['email'])
                    owner_info['address'] = self.encryption_manager.decrypt(owner_info['address'])
                
                tickets.append(Ticket(**item))
            
            return tickets
            
        except Exception as e:
            logger.error(f"Failed to get tickets by status {status}: {e}")
            raise
    
    # --- Data Export and Backup ---
    
    async def export_conversations(self, user_id: Optional[str] = None, 
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export conversations for reporting"""
        try:
            table = self.dynamodb.Table(TABLES['conversations']['TableName'])
            
            if user_id:
                response = table.query(
                    IndexName='user_id-index',
                    KeyConditionExpression='user_id = :user_id',
                    ExpressionAttributeValues={':user_id': user_id}
                )
            else:
                response = table.scan()
            
            conversations = []
            for item in response.get('Items', []):
                # Filter by date if specified
                if start_date or end_date:
                    conv_date = datetime.fromisoformat(item['timestamp'])
                    if start_date and conv_date < start_date:
                        continue
                    if end_date and conv_date > end_date:
                        continue
                
                # Remove sensitive data for export
                if 'messages' in item:
                    for message in item['messages']:
                        # Remove PII from message content
                        pii_patterns = [r'\b\d{3}-\d{2}-\d{4}\b', r'\b\d{10}\b']
                        content = message['content']
                        for pattern in pii_patterns:
                            content = re.sub(pattern, '[REDACTED]', content)
                        message['content'] = content
                
                conversations.append(item)
            
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to export conversations: {e}")
            raise
    
    async def backup_data(self, backup_path: str) -> bool:
        """Create a backup of all data"""
        try:
            backup_data = {
                'conversations': [],
                'tickets': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Backup conversations
            conv_table = self.dynamodb.Table(TABLES['conversations']['TableName'])
            conv_response = conv_table.scan()
            backup_data['conversations'] = conv_response.get('Items', [])
            
            # Backup tickets (encrypted)
            ticket_table = self.dynamodb.Table(TABLES['tickets']['TableName'])
            ticket_response = ticket_table.scan()
            backup_data['tickets'] = ticket_response.get('Items', [])
            
            # Write backup to file
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    # --- Audit Logging ---
    
    def _log_audit(self, operation: str, table_name: str, record_id: str, user_id: Optional[str] = None):
        """Log audit trail for database operations"""
        if not AUDIT_CONFIG['log_all_operations']:
            return
        
        try:
            audit_log = AuditLog(
                log_id=f"audit-{uuid.uuid4().hex[:8]}",
                operation=operation,
                table_name=table_name,
                record_id=record_id,
                user_id=user_id,
                details={
                    'timestamp': datetime.utcnow().isoformat(),
                    'environment': os.environ.get('ENVIRONMENT', 'development')
                }
            )
            
            # Store audit log (could be a separate table or external service)
            logger.info(f"AUDIT: {operation} on {table_name} record {record_id} by {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to log audit: {e}")
    
    # --- Session Management ---
    
    async def get_or_create_session(self, user_id: str) -> str:
        """Get existing conversation or create new one for user"""
        try:
            # Get recent active conversation
            conversations = await self.get_user_conversations(user_id)
            active_conversations = [c for c in conversations if c.status == 'active']
            
            if active_conversations:
                # Return most recent active conversation
                return max(active_conversations, key=lambda x: x.updated_at).conversation_id
            else:
                # Create new conversation
                conversation = await self.create_conversation(user_id)
                return conversation.conversation_id
                
        except Exception as e:
            logger.error(f"Failed to get/create session for user {user_id}: {e}")
            raise 