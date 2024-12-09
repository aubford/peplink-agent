from pathlib import Path
from typing import List, Dict, Any, TypeVar, Type, Optional
from datetime import datetime
import json
import pandas as pd
from pydantic import BaseModel, ValidationError
import logging
import inflection
import os

T = TypeVar('T', bound=BaseModel)

# Configure logging at the beginning of your module
logging.basicConfig(
    filename=os.path.join('logs', 'base_extractor.log'),
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BaseExtractor:
    """Base class for all data extractors."""

    def __init__(self, source_name: str):
        """
        Initialize the base extractor.

        Args:
            source_name: Name of the data source (e.g., 'youtube', 'twitter', etc.)
        """
        self.source_name = source_name
        self.raw_data: Dict[str, Any] = {}
        self.data: Dict[str, T] = {}

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

    def _generate_filename(self, filename: str, extension: str) -> str:
        """Generate a timestamped filename with the given extension."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_identifier = self._sanitize_filename(filename)
        return f"{self.source_name}_{safe_identifier}_{timestamp}.{extension}"

    def save_raw_data(self, data: List[Dict[str, Any]], filename: str) -> Path:
        """
        Save raw data to a JSON file in the data directory.

        Args:
            data: Data to save (any JSON-serializable object)
            identifier: Unique identifier for the data (e.g., channel name, user handle)
        """
        json_path = self._ensure_dir("raw") / self._generate_filename(filename, "json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON file to: {json_path}")
        return json_path

    def save_data(self, data: List[Dict[str, T]], filename: str) -> None:
        """
        Save data to Parquet file in the data directory.

        Args:
            data: List of data items to save
            filename: Unique identifier for the data (e.g., channel name, user handle)
        """
        if not data:
            print("No data to save.")
            return

        parquet_path = self._ensure_dir("parquet") / self._generate_filename(filename, "parquet")
        pd.DataFrame(data).to_parquet(parquet_path)
        print(f"Saved Parquet file to: {parquet_path}")

    def save_data_to_file(self) -> None:
        """
        Save data to a file in the data directory.
        """
        for filename, data in self.data.items():
            self.save_data(data, filename)
        for filename, data in self.raw_data.items():
            self.save_raw_data(data, filename)


    def validate_and_convert_model(self, data: Dict[str, Any], model_class: Type[T], *, filename: Optional[str] = None) -> None:
        """
        Validate dictionary data against a Pydantic model and store in self.data and self.raw_data.

        Args:
            data: Dictionary containing the data
            model_class: The Pydantic model class to validate against
            identifier: Unique identifier to use as key in self.data and self.raw_data dictionaries
        """
        key = filename if filename else inflection.underscore(model_class.__name__)
        # save raw data regardless of validation for inspection (being lazy)
        self.raw_data[key] = data
        try:
            validated_model = model_class.model_validate(data)
            self.data[key] = validated_model.model_dump()
        except ValidationError as e:
            error_message = (
                f"Validation error for identifier '{key}' with data: {json.dumps(data, ensure_ascii=False, indent=2)} \n"
                f"Error: {e}"
            )
            print(error_message)
            logging.error(error_message)
