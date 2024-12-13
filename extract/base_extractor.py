from pathlib import Path
from typing import Dict, Any, TypeVar, Type, Optional, Tuple
from datetime import datetime
import json
from pydantic import BaseModel, ValidationError
import logging
import inflection

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger('extractor')


def sanitize_filename(filename: str) -> str:
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


class BaseExtractor:
    """Base class for all data extractors."""

    def __init__(self, source_name: str):
        """
        Initialize the base extractor.

        Args:
            source_name: Name of the data source (e.g., 'youtube', 'twitter', etc.)
        """
        self.source_name = source_name
        # Dictionary to track active streams: filename -> (model_class, raw_path)
        self._active_streams: Dict[str, Tuple[Type[T], Path]] = {}

    def _ensure_dir(self, dir_type: str) -> Path:
        """Create and return path to a data directory of specified type."""
        dir_path = Path("data") / self.source_name / dir_type
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def start_stream(self, model_class: Type[T], *, identifier: Optional[str] = None) -> str:
        """
        Start a new streaming session for a specific model type.
        Creates new file for raw data.

        Args:
            model_class: The Pydantic model class to stream
            identifier: label
        """
        modelname = inflection.underscore(model_class.__name__)
        safe_identifier = sanitize_filename(modelname (f"_{identifier}" if identifier else "")) 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.source_name}_{safe_identifier}__T_{timestamp}"

        # Check if stream already exists using the stream_key
        if safe_identifier in self._active_streams:
            raise ValueError(f"Stream for {safe_identifier} already exists")

        # Set up raw data file (JSONL)
        raw_path = self._ensure_dir("raw") / f"{filename}.jsonl"

        self._active_streams[safe_identifier] = (model_class, raw_path)

        print(f"Started stream for {safe_identifier}:\nRaw: {raw_path}")
        return safe_identifier

    def stream_item(self, data: Dict[str, Any], stream_key: str) -> None:
        """
        Validate and stream a single item to raw file.

        Args:
            data: Dictionary containing the item data
            stream_key: Stream key
        """
        if stream_key not in self._active_streams:
            raise RuntimeError(f"Must call start_stream() for {stream_key} before streaming items")

        model_class, raw_path = self._active_streams[stream_key]

        # Validate data
        try:
            validated_model = model_class.model_validate(data)
            processed_data = validated_model.model_dump()
        except ValidationError as e:
            error_message = (
                f"Validation error for model: {model_class.__name__}\n"
                f"Error: {e}"
            )
            logger.error(error_message)
            return

        # Stream raw data
        with open(raw_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(processed_data, ensure_ascii=False) + '\n')

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
