import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import time


class RotatingFileLogger(logging.Logger):
    """Logger with rotating file handling configured by default."""

    def __init__(
        self,
        name: str,
        log_level: int = logging.INFO,
        silent: bool = False,
        log_to_console: bool = True,
    ):
        super().__init__(name)
        self.setLevel(log_level)
        self.silent = silent
        self.root_dir = Path(__file__).parent.parent

        if not self.silent:
            # Create logs directory if it doesn't exist
            log_dir = self.root_dir / "logs"
            log_dir.mkdir(exist_ok=True)

            # Create common formatter
            formatter = logging.Formatter(
                "%(asctime)s (%(name)s) %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",  # Specify date format without subseconds
            )

            # Configure rotating file handler
            file_handler = RotatingFileHandler(
                os.path.join("logs", f"{name}.log"),
                maxBytes=40 * 1024 * 1024,  # 40 MB
                backupCount=0,
            )
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)

            if log_to_console:
                # Configure console handler with same formatter
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(log_level)  # Set same level as logger
                console_handler.setFormatter(formatter)
                self.addHandler(console_handler)

    def n_info(self, msg: str) -> None:
        if not self.silent:
            self.info("")
            self.info(msg)

    def br_info(self, msg: str) -> None:
        if not self.silent:
            self.info("-" * 100)
            self.info(msg)


class RotatingFileLogWriter(logging.Logger):
    """Logger with rotating file handling configured by default."""

    def __init__(
        self,
        name: str,
        *,
        backup_count=1,
        max_bytes: int = 75 * 1024 * 1024,
        silent: bool = False,
    ):
        super().__init__(name)
        self.setLevel(logging.DEBUG)
        self.silent = silent
        self.root_dir = Path(__file__).parent.parent

        if not self.silent:
            # Create logs directory if it doesn't exist
            log_dir = self.root_dir / "logs"
            log_dir.mkdir(exist_ok=True)

            self.file_handler = RotatingFileHandler(
                os.path.join(
                    "logs",
                    f"{name}_{time.strftime('%m-%d_%H_%M')}.log",
                ),
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            self.addHandler(self.file_handler)

    def log_and_print(self, msg: str) -> None:
        if not self.silent:
            print(msg)
            self.info(msg)

    def log_header(self, msg: str) -> None:
        if not self.silent:
            log_msg = f"\n----------------------- {msg}"
            self.info(log_msg)

    def log_and_print_header(self, msg: str) -> None:
        if not self.silent:
            log_msg = f"\n----------------------- {msg}"
            print(log_msg)
            self.info(log_msg)
