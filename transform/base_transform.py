import uuid
from abc import abstractmethod
import logging
from typing import Any, Dict
from config import global_config, ConfigType
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from pathlib import Path


logger = logging.getLogger("transformer")


class BaseTransform:
    """Base class for all data transformers."""

    def __init__(self, folder_name: str):
        self.folder_name = folder_name

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    @abstractmethod
    def transform_file(self, data: Dict[str, Any]) -> pd.DataFrame:
        pass

    def transform(self) -> None:
        """Process all files in raw directory and save to documents."""
        raw_dir = Path("data") / self.folder_name / "raw"
        staging_dir = self._ensure_dir()

        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

        for file_path in raw_dir.glob("*"):
            try:
                df = self.transform_file(file_path)
                output_path = staging_dir / f"{file_path.stem}.parquet"
                pq.write_table(pyarrow.Table.from_pandas(df), output_path)

            except Exception as e:
                logger.error(f"Error processing {file_path}")
                raise e

    def _ensure_dir(self) -> Path:
        """Create and return path to documents directory."""
        dir_path = Path("data") / self.folder_name / "documents"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def add_required_columns(
        self,
        columns: dict,
        page_content: str,
        file_path: Path,
        doc_id: str | uuid.UUID | None = None,
    ) -> dict:
        columns["id"] = str(uuid.uuid4()) if doc_id is None else doc_id
        columns["source_file"] = self.get_stem(file_path)
        columns["page_content"] = page_content
        return columns

    @staticmethod
    def get_stem(file_path: Path) -> str:
        """Get the stem of a file path."""
        return Path(file_path.name).stem
