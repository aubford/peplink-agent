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

    def save_data(self, data: List[Dict[Any, Any]], identifier: str) -> None:
        """
        Save data to JSON and Parquet files in the data directories.

        Args:
            data: List of data items to save
            identifier: Unique identifier for the data (e.g., channel name, user handle)
        """
        if not data:
            print("No data to save.")
            return

        # Ensure data directories exist
        data_dir = Path("data")
        target_dir = data_dir / self.source_name
        json_dir = target_dir / "json"
        parquet_dir = target_dir / "parquet"

        for dir_path in [data_dir, target_dir, json_dir, parquet_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.source_name}_{identifier}_{timestamp}"

        # Save JSON
        json_path = json_dir / f"{base_filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON file to: {json_path}")

        # Save Parquet
        df = pd.DataFrame(data)
        parquet_path = parquet_dir / f"{base_filename}.parquet"
        df.to_parquet(parquet_path)
        print(f"Saved Parquet file to: {parquet_path}")
