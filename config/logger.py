import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys


class RotatingFileLogger(logging.Logger):
    """Logger with rotating file handling configured by default."""

    def __init__(self, name: str, log_level: int = logging.INFO):
        super().__init__(name)
        self.setLevel(log_level)

        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Create common formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Configure rotating file handler
        file_handler = RotatingFileHandler(
            os.path.join("logs", f"{name}.log"),
            maxBytes=20 * 1024 * 1024,  # 20 MB
        )
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

        # Configure console handler with same formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)  # Set same level as logger
        self.addHandler(console_handler)
