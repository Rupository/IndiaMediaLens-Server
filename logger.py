import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logging.basicConfig(
        handlers=[RotatingFileHandler('runs.log', maxBytes=1024, backupCount=3)],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )