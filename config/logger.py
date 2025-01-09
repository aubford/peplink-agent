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
            "%(asctime)s (%(name)s) %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",  # Specify date format without subseconds
        )

        # Configure rotating file handler
        file_handler = RotatingFileHandler(
            os.path.join("logs", f"{name}.log"),
            maxBytes=20 * 1024 * 1024,  # 20 MB
            backupCount=2
        )
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

        # Configure console handler with same formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)  # Set same level as logger
        self.addHandler(console_handler)

    def n_info(self, msg: str) -> None:
        self.info("")
        self.info(msg)

    def br_info(self, msg: str) -> None:
        self.info("-" * 100)
        self.info(msg)



class RotatingFileLogWriter(logging.Logger):
    """Logger with rotating file handling configured by default."""

    def __init__(self, name: str, max_bytes: int = 20 * 1024 * 1024):
        super().__init__(name)
        self.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Create common formatter
        self.default_formatter = logging.Formatter(
            "\n%(asctime)s: %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )

        # Configure rotating file handler
        self.file_handler = RotatingFileHandler(
            os.path.join("logs", f"{name}.log"),
            maxBytes=max_bytes,
            backupCount=2,
        )
        self.file_handler.setFormatter(self.default_formatter)
        self.addHandler(self.file_handler)

    def print(self, msg: str) -> None:
        # self.file_handler.setFormatter(self.simple_formatter)
        print(msg)
        self.info(msg)
        # self.file_handler.setFormatter(self.default_formatter)

    def log_header(self, msg: str) -> None:
        log_msg = f"\n----------------------- {msg}"
        self.info(log_msg)

    def print_header(self, msg: str) -> None:
        log_msg = f"\n----------------------- {msg}"
        print(log_msg)
        self.info(log_msg)
