from utils.logger import get_logger
from utils.conversation_manager import ConversationManager
import uuid

logger = get_logger("aws_bedrock")

def get_conversation_manager(session_id: str = None) -> ConversationManager:
    """
    Get a conversation manager instance for the given session ID.
    If no session_id is provided, generates a new one.
    """
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())
        return ConversationManager(session_id)
    except Exception as e:
        logger.error(f"Failed to initialize ConversationManager: {e}")
        raise 