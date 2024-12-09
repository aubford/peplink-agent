from pathlib import Path
from typing import List, Dict, Any, TypeVar, Type, Optional, Tuple
from datetime import datetime
import json
import pandas as pd
from pydantic import BaseModel, ValidationError
import logging
from logging.handlers import RotatingFileHandler
import inflection
import os

T = TypeVar('T', bound=BaseModel)

# Configure rotating file handler
handler = RotatingFileHandler(
    os.path.join('logs', 'base_extractor.log'),  # Log file name
    maxBytes=10*1024*1024,    # Maximum file size in bytes (10 MB)
    backupCount=0             # No backup files
)

# Set up logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Get the logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logger.addHandler(handler)

class BaseExtractor:
    """Base class for all data extractors."""

    def __init__(self, source_name: str):
        """
        Initialize the base extractor.

        Args:
            source_name: Name of the data source (e.g., 'youtube', 'twitter', etc.)
        """
        self.source_name = source_name
        # Dictionary to track active streams: filename -> (raw_path, parquet_path)
        self._active_streams: Dict[str, Tuple[Type[T], Path, Path]] = {}

    def _ensure_dir(self, dir_type: str) -> Path:
        """Create and return path to a data directory of specified type."""
        dir_path = Path("data") / self.source_name / dir_type
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize the identifier to be safe for use in filenames across all systems.
        Removes/replaces invalid filename characters using a standard approach.
        """
        # Common invalid filename characters
        invalid_chars = '<>:"/\\|?*'

        # First handle leading special characters
        while filename and (filename[0] in invalid_chars or not filename[0].isalnum()):
            filename = filename[1:]

        # Then replace remaining invalid characters with underscore
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Replace spaces with underscore and remove any duplicate underscores
        filename = '_'.join(filename.split())
        while '__' in filename:
            filename = filename.replace('__', '_')

        return filename.strip('_')

    def start_stream(self, model_class: Type[T], *, identifier: Optional[str] = None) -> str:
        """
        Start a new streaming session for a specific model type.
        Creates new files for both raw and processed data.

        Args:
            model_class: The Pydantic model class to stream
        """
        modelname = inflection.underscore(model_class.__name__)
        safe_identifier = self._sanitize_filename(modelname) + (f"_{identifier}" if identifier else "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.source_name}_{safe_identifier}_{timestamp}"

        # Check if stream already exists using the stream_key
        if safe_identifier in self._active_streams:
            raise ValueError(f"Stream for {safe_identifier} already exists")

        # Set up raw data file (JSONL)
        raw_path = self._ensure_dir("raw") / f"{filename}.jsonl"
        # Set up processed data file (Parquet)
        parquet_path = self._ensure_dir("parquet") / f"{filename}.parquet"

        self._active_streams[safe_identifier] = (model_class, raw_path, parquet_path)

        print(f"Started stream for {safe_identifier}:\nRaw: {raw_path}\nProcessed: {parquet_path}")
        return safe_identifier

    def stream_item(self, data: Dict[str, Any], stream_key: str) -> None:
        """
        Validate and stream a single item to both raw and processed files.

        Args:
            data: Dictionary containing the item data
            model_class: The Pydantic model class to validate against
        """
        if stream_key not in self._active_streams:
            raise RuntimeError(f"Must call start_stream() for {stream_key} before streaming items")

        model_class, raw_path, parquet_path = self._active_streams[stream_key]

        # Stream raw data
        with open(raw_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

        # Validate data
        try:
            validated_model = model_class.model_validate(data)
            processed_data = validated_model.model_dump()
        except ValidationError as e:
            error_message = (
                f"Validation error for {model_class.__name__} with data: {json.dumps(data, ensure_ascii=False, indent=2)} \n"
                f"Error: {e}"
            )
            print(error_message)
            logger.error(error_message)
            return

        # Stream processed data
        df = pd.DataFrame([processed_data])
        # If file doesn't exist, write with schema, otherwise append
        if not parquet_path.exists():
            df.to_parquet(parquet_path)
        else:
            df.to_parquet(parquet_path, append=True)

    def end_stream(self, stream_key: str) -> None:
        """
        End a specific streaming session.

        Args:
            model_class: The Pydantic model class whose stream to end
        """
        if stream_key not in self._active_streams:
            raise ValueError(f"No active stream for {stream_key}")

        del self._active_streams[stream_key]
        print(f"Ended stream for {stream_key}")

    def end_all_streams(self) -> None:
        """End all active streaming sessions."""
        for stream_key in list(self._active_streams.keys()):
            self.end_stream(stream_key)
