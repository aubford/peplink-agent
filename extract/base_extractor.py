from abc import ABC, abstractmethod
import inflection
from pathlib import Path
from typing import Dict, Any, TypeVar, Type, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, ValidationError
from config import RotatingFileLogger, global_config, ConfigType
from langchain_core.load import dumps
from util.util_main import sanitize_filename
from pandas import DataFrame, read_json
import logging


T = TypeVar("T", bound=BaseModel)


class Ldoc(BaseModel):
    page_content: str
    metadata: Dict[str, Any]


class BaseExtractor(ABC):
    """Base class for all data extractors.  Enforces a common folders structure in the data dir."""

    source_name: str = NotImplemented

    def __init__(self):
        """Initialize the base extractor."""
        self._active_streams: Dict[str, Tuple[Type[T], Path]] = {}
        self.logger = RotatingFileLogger(sanitize_filename(self.source_name))
        self.validation_error_items = []

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    @abstractmethod
    def extract(self) -> None:
        """Extract data according to implementation-specific rules.

        Returns:
            The extracted data in implementation-specific format.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass

    def set_logger(self, name: str):
        self.logger = RotatingFileLogger(f"extract__{sanitize_filename(name)}")

    def _ensure_dir(self, dir_type: str) -> Path:
        """Create and return path to a data directory of specified type."""
        dir_path = Path("data") / self.source_name / dir_type
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _get_filename(self, identifier: str) -> str:
        sanitized_identifier = sanitize_filename(identifier)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.source_name}_{sanitized_identifier}__T_{timestamp}"

    def write_json(self, data: list[dict], identifier: str):
        file_path = self._ensure_dir("raw") / f"{self._get_filename(identifier)}.json"
        print(f"Writing {len(data)} documents to {file_path}")
        with open(file_path, "w") as json_file:
            json_file.write(dumps(data, ensure_ascii=False))

    def start_stream(
        self, model_class: Type[T], *, identifier: Optional[str] = None
    ) -> str:
        """
        Start a new streaming session for a specific model type.
        Creates new file for raw data.

        Args:
            model_class: The Pydantic model class to stream
            identifier: label
        """
        modelname = inflection.underscore(model_class.__name__)
        safe_identifier = sanitize_filename(
            f"{modelname}_{identifier if identifier else ''}"
        )
        filename = self._get_filename(safe_identifier)

        # Check if stream already exists using the stream_key
        if safe_identifier in self._active_streams:
            raise ValueError(f"Stream for {safe_identifier} already exists")

        # Set up raw data file (JSONL)
        raw_path = self._ensure_dir("raw") / f"{filename}.jsonl"

        self._active_streams[safe_identifier] = (model_class, raw_path)

        self.logger.info({datetime.now().strftime("%Y-%m-%d %H:%M")})
        self.logger.br_info(f"Started stream for {safe_identifier}:\nRaw: {raw_path}")
        return safe_identifier

    def stream_item(self, data: Dict[str, Any], stream_key: str) -> None:
        """
        Validate and stream a single item to raw file.

        Args:
            data: Dictionary containing the item data
            stream_key: Stream key
        """
        if stream_key not in self._active_streams:
            raise RuntimeError(
                f"Must call start_stream() for {stream_key} before streaming items"
            )

        model_class, raw_path = self._active_streams[stream_key]

        # Validate data
        try:
            validated_model = model_class.model_validate(data)
            processed_data = validated_model.model_dump()
        except ValidationError as e:
            self.validation_error_items.append(data)
            error_message = (
                f"Validation error for model: {model_class.__name__}\n" f"Error: {e}"
            )
            self.logger.error(error_message)
            return

        # Stream raw data
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(dumps(processed_data, ensure_ascii=False) + "\n")

    def end_stream(self, stream_key: str) -> None:
        """
        End a specific streaming session.

        Args:
            stream_key: Stream key
        """
        if stream_key not in self._active_streams:
            raise ValueError(f"No active stream for {stream_key}")

        del self._active_streams[stream_key]
        print(f"Ended stream for {stream_key}")

    def end_all_streams(self) -> None:
        """End all active streaming sessions."""
        for stream_key in list(self._active_streams.keys()):
            self.end_stream(stream_key)

    @classmethod
    def get_rawfiles(cls) -> list[Path]:
        """
        Get all files created by this extractor in the specified directory type.

        Returns:
            List of Path objects for files matching this extractor's source name
        """
        dir_path = Path("data") / cls.source_name / "raw"
        if not dir_path.exists():
            return []

        return sorted(p for p in dir_path.glob(f"{cls.source_name}_*") if p.is_file())

    @classmethod
    def get_rawfile_dataframes(cls) -> list[tuple[str, DataFrame]]:
        """
        Load all files created by this extractor into pandas DataFrames.

        Returns:
            List of tuples containing (filename, DataFrame) for each file
        """
        files = cls.get_rawfiles()
        dataframes = []

        for file_path in files:
            try:
                if file_path.suffix == ".jsonl":
                    # Read JSONL file line by line
                    df = read_json(file_path, lines=True)
                else:
                    # Read regular JSON file
                    df = read_json(file_path)

                dataframes.append((file_path.name, df))

            except Exception as e:
                logging.error(f"Failed to load DataFrame from {file_path}: {str(e)}")

        return dataframes
