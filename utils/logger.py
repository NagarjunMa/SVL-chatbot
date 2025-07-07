import logging
import os

def get_logger(name: str = "svl_chatbot"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    return logger 