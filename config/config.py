from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Any, Dict


class Config:
    _instance = None
    root_dir: Path

    def __new__(cls):
        """ensure only one instance of the config can be created"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration and set root dir to the project root."""
        root_dir = Path(__file__).parent.parent
        os.chdir(root_dir)
        self.root_dir = root_dir
        # Load environment variables from .env file into os.environ
        load_dotenv()

    def get(self, key: str, default: Any = None) -> Any:
        return os.getenv(key, default)

    def set(self, key: str, value: Any) -> None:
        os.environ[key] = str(value)

    def get_all(self) -> Dict[str, Any]:
        return dict(os.environ)


# Create a global instance
global_config = Config()

# check that the correct work dir is set
def main():
    print(global_config.root_dir)
    print(Path().resolve())


if __name__ == "__main__":
    main()
