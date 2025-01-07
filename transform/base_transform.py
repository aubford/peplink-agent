import uuid
from abc import abstractmethod
from typing import Any, Dict
from config import global_config, ConfigType, RotatingFileLogger
import pandas as pd
from pathlib import Path
from util.util_main import set_string_columns, sanitize_filename


class BaseTransform:
    """Base class for all data transformers."""

    folder_name: str = NotImplemented

    def __init__(self):
        self.row_count = None
        self.logger = RotatingFileLogger(self.folder_name)

    def set_logger(self, name: str):
        self.logger = RotatingFileLogger(f"extract__{sanitize_filename(name)}")

    def log_df(self, df: pd.DataFrame, file_name: Path) -> None:
        self.logger.br_info(f"DF Transformed: {file_name}\n")
        self.logger.n_info(df.info(show_counts=True))

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
        parquet_dir = self._ensure_dir()

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

    def _ensure_dir(self) -> Path:
        """Create and return path to documents directory."""
        dir_path = Path("data") / self.folder_name / "documents"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _add_required_columns(
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

    def _make_df(self, data: list[dict]) -> pd.DataFrame:
        """Make a DataFrame from a list of dictionaries."""
        df = pd.DataFrame(data).set_index("id", verify_integrity=True, drop=False)
        set_string_columns(df, ["page_content"])
        set_string_columns(df, ["id", "source_file"], False)
        self.row_count = df.shape[0]
        self.logger.info(f"Initial length: {self.row_count}\n")
        return df

    def _notify_dropped_rows(self, df: pd.DataFrame, operation_name: str) -> None:
        """Get the number of rows dropped during a given operation."""
        new_count = df.shape[0]
        dropped_rows = self.row_count - new_count
        self.row_count = new_count
        self.logger.info(f"Dropped {dropped_rows} rows for rule: {operation_name}")

    @classmethod
    def get_parquet_files(cls) -> list[Path]:
        """
        Get all parquet files created by this transformer.

        Returns:
            List of Path objects for parquet files in this transformer's directory
        """
        dir_path = Path("data") / cls.folder_name / "parquet"
        if not dir_path.exists():
            return []

        return sorted(p for p in dir_path.glob("*.parquet") if p.is_file())

    @classmethod
    def get_parquet_dfs(cls) -> list[tuple[str, pd.DataFrame]]:
        """
        Load all parquet files created by this transformer into pandas DataFrames.

        Returns:
            List of tuples containing (filename, DataFrame) for each parquet file
        """
        files = cls.get_parquet_files()
        dataframes = []

        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                dataframes.append((file_path.name, df))
            except Exception as e:
                print(f"Failed to load DataFrame from {file_path}: {str(e)}")

        return dataframes
