import logging

from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
from typing import Any, Dict
from config.logger import RotatingFileLogger


class Config:
    _instance = None
    _env_vars: Dict[str, Any] = {}
    root_dir: Path

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration by loading environment variables."""
        root_dir = Path(__file__).parent.parent
        os.chdir(root_dir)
        self.root_dir = root_dir

        load_dotenv()
        pd.options.io.parquet.engine = 'fastparquet'
        logging.setLoggerClass(RotatingFileLogger)
        self._load_environment_variables()

    def _load_environment_variables(self):
        """Load all environment variables and set them in the config."""
        # Load all environment variables into the _env_vars dictionary
        for key, value in os.environ.items():
            self._env_vars[key] = value

        # Set commonly used environment variables
        self._set_langchain_env_vars()

    def _set_langchain_env_vars(self):
        """Set required environment variables for LangChain."""
        langchain_vars = {
            "LANGCHAIN_API_KEY": self.get("LANGCHAIN_API_KEY"),
            "OPENAI_API_KEY": self.get("OPENAI_API_KEY"),
            "LANGCHAIN_TRACING_V2": self.get("LANGCHAIN_TRACING_V2", "true"),
            "USER_AGENT": self.get("USER_AGENT")
        }

        for key, value in langchain_vars.items():
            if value is not None:
                os.environ[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: The configuration key to look up
            default: Default value if key is not found

        Returns:
            The configuration value or default if not found
        """
        return self._env_vars.get(key, os.getenv(key, default))

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: The configuration key
            value: The value to set
        """
        self._env_vars[key] = value
        os.environ[key] = str(value)

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._env_vars.copy()


# Create a global instance
config = Config()

# check that the correct work dir is set
def main():
    print(config.root_dir)
    print(Path().resolve())


if __name__ == "__main__":
    main()
