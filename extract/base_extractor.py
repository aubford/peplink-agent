from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json
import pandas as pd

class BaseExtractor:
    """Base class for all data extractors."""

    def __init__(self, source_name: str):
        """
        Initialize the base extractor.

        Args:
            source_name: Name of the data source (e.g., 'youtube', 'twitter', etc.)
        """
        self.source_name = source_name

    def _ensure_dir(self, dir_type: str) -> Path:
        """Create and return path to a data directory of specified type."""
        dir_path = Path("data") / self.source_name / dir_type
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _generate_filename(self, identifier: str, extension: str) -> str:
        """Generate a timestamped filename with the given extension."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.source_name}_{identifier}_{timestamp}.{extension}"

    def save_raw_data(self, data: Any, identifier: str) -> Path:
        """
        Save raw data to a JSON file in the data directory.

        Args:
            data: Data to save (any JSON-serializable object)
            identifier: Unique identifier for the data (e.g., channel name, user handle)
        """
        json_path = self._ensure_dir("raw") / self._generate_filename(identifier, "json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON file to: {json_path}")
        return json_path

    def save_data(self, data: List[Dict[Any, Any]], identifier: str) -> None:
        """
        Save data to Parquet file in the data directory.

        Args:
            data: List of data items to save
            identifier: Unique identifier for the data (e.g., channel name, user handle)
        """
        if not data:
            print("No data to save.")
            return

        parquet_path = self._ensure_dir("parquet") / self._generate_filename(identifier, "parquet")
        pd.DataFrame(data).to_parquet(parquet_path)
        print(f"Saved Parquet file to: {parquet_path}")
