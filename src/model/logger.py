# logger.py
import logging
import os
from datetime import datetime

def get_logger(name: str, log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    """
    Create and configure a logger.
    
    Args:
        name (str): Logger name (usually __name__ or script name)
        log_dir (str): Directory to store log files
        level (int): Logging level (default=logging.INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Ensure logs directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Log file name: time-name.log
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(log_dir, f"{timestamp}-{name}.log")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Avoid duplicate logs

    # Formatter with date, time, and log level
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
