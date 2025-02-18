import uuid
from abc import ABC, abstractmethod
from typing import override

from config import global_config, ConfigType, RotatingFileLogger
import pandas as pd
from pathlib import Path
from util.util_main import (
    set_string_columns,
    sanitize_filename,
)
from util.document_utils import load_parquet_files, get_all_parquet_in_dir
from enum import StrEnum
import pymongo


class SubjectMatter(StrEnum):
    PEPWAVE = "PEPWAVE"
    IT_NETWORKING = "IT_NETWORKING"
    MOBILE_INTERNET = "MOBILE_INTERNET"


class BaseTransform(ABC):
    """Base class for all data transformers."""

    folder_name: str = NotImplemented
    subject_matter: SubjectMatter = NotImplemented

    def __init__(self):
        if not isinstance(self.subject_matter, SubjectMatter):
            raise TypeError(
                f"subject_matter must be an instance of SubjectMatter, got {type(self.subject_matter)}"
            )
        self.row_count = None
        self.logger = RotatingFileLogger(f"transform__{self.folder_name}")

    def set_logger(self, name: str):
        self.logger = RotatingFileLogger(f"transform__{sanitize_filename(name)}")

    def log_df(self, df: pd.DataFrame, file_name: Path | str) -> None:
        self.logger.br_info(f"DF Transformed: {file_name}\n")
        df.info(show_counts=True)

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    @abstractmethod
    def transform_file(self, file_path: Path) -> pd.DataFrame:
        """Transform a single file into a pandas DataFrame."""
        pass

    def transform(self) -> None:
        """Process all files in raw directory and save to documents."""
        raw_dir = Path("data") / self.folder_name / "raw"
        parquet_dir = self.ensure_dir()

        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

        for file_path in raw_dir.glob("*"):
            # Skip system files
            if file_path.name.startswith("."):
                continue

            file_name = file_path.name.split("__T")[0]
            self.logger.br_info(f"\n\n******Transforming file: {file_name}\n\n")

            try:
                df = self.transform_file(file_path)
                output_path = parquet_dir / f"{file_path.stem}.parquet"
                self.log_df(df, file_name)
                df.to_parquet(
                    output_path, index=True, compression="snappy", engine="pyarrow"
                )

            except Exception as e:
                self.logger.error(f"Error processing {file_name}")
                raise e

    def ensure_dir(self) -> Path:
        """Create and return path to documents directory."""
        dir_path = Path("data") / self.folder_name / "documents"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def add_required_columns(
        self,
        columns: dict,
        *,
        page_content: str,
        file_path: Path | str,
        doc_id: str | uuid.UUID | None = None,
    ) -> dict:
        if isinstance(file_path, Path):
            file_path = self.get_stem(file_path)

        columns["id"] = str(uuid.uuid4()) if doc_id is None else doc_id
        columns["source_file"] = file_path
        columns["page_content"] = page_content
        columns["subject_matter"] = self.subject_matter
        return columns

    @staticmethod
    def get_stem(file_path: Path) -> str:
        """Get the stem of a file path."""
        return Path(file_path.name).stem

    def make_df(self, data: list[dict]) -> pd.DataFrame:
        """Make a DataFrame from a list of dictionaries."""
        df = pd.DataFrame(data).set_index("id", verify_integrity=True, drop=False)
        set_string_columns(df, ["page_content"])
        set_string_columns(df, ["id", "source_file"], False)
        self.row_count = df.shape[0]
        self.logger.info(f"DataFrame length: {self.row_count}\n")
        return df

    def notify_dropped_rows(self, df: pd.DataFrame, operation_name: str) -> None:
        """Get the number of rows dropped during a given operation."""
        new_count = df.shape[0]
        dropped_rows = self.row_count - new_count
        self.row_count = new_count
        self.logger.info(f"Dropped {dropped_rows} rows for rule: {operation_name}")

    @classmethod
    def get_artifact_file_paths(cls) -> list[Path]:
        """
        Get all parquet files created by this transformer.

        Returns:
            List of Path objects for parquet files in this transformer's directory
        """
        dir_path = Path("data") / cls.folder_name / "documents"
        return get_all_parquet_in_dir(dir_path)

    @classmethod
    def get_artifacts(cls) -> list[pd.DataFrame]:
        """
        Load all parquet files created by this transformer into pandas DataFrames.

        Returns:
            List of DataFrames, one for each successfully loaded parquet file
        """
        files = cls.get_artifact_file_paths()
        return load_parquet_files(files)


class BaseMongoTransform(BaseTransform, ABC):
    """Base class for transformers that retrieve data from MongoDB."""

    def __init__(self, db_uri: str, db_name: str):
        super().__init__()
        self.client = pymongo.MongoClient(db_uri)
        self.db_name = db_name
        self.db = self.client[db_name]

    transform_file = NotImplemented

    @abstractmethod
    def transform_db(self) -> pd.DataFrame:
        """Retrieve data from MongoDB, transform, and save to parquet files."""
        pass

    @override
    def transform(self) -> None:
        """Retrieve data from MongoDB, transform, and save to parquet files."""
        parquet_dir = self.ensure_dir()

        try:
            df = self.transform_db()
            self.log_df(df, f"{self.folder_name}_{self.subject_matter}")

            output_path = parquet_dir / f"{self.folder_name}_{self.db_name}.parquet"
            df.to_parquet(
                output_path, index=True, compression="snappy", engine="pyarrow"
            )

        except Exception as e:
            self.logger.error(f"Error processing MongoDB data for {self.folder_name}")
            raise e
