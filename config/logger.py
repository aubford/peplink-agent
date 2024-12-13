import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys


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

        # Create common formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Configure rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024  # 10 MB
        )
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)
