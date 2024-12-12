import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Optional


class RotatingFileLogger(logging.Logger):
    """Logger with rotating file handling configured by default."""

    def __init__(
        self,
        name: str,
        log_level: int = logging.INFO
    ):
        super().__init__(name)
        self.setLevel(log_level)

        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # Set default log file if none provided
        log_file = os.path.join('logs', f'{name}.log')

        # Configure rotating file handler
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024  # 10 MB
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.addHandler(handler)


# Register our logger class
logging.setLoggerClass(RotatingFileLogger)
