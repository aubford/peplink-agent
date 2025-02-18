import pandas as pd
import numpy as np
from typing import List, Dict, Any
from langchain.docstore.document import Document
from pathlib import Path


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays in the metadata dictionary to lists."""
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        else:
            sanitized[key] = value
    return sanitized


def df_to_documents(df: pd.DataFrame) -> List[Document]:
    """Convert a DataFrame to a list of Document objects."""
    documents: List[Document] = []
    for _, row in df.iterrows():
        metadata = row.drop(["page_content", "id"]).to_dict()
        metadata = sanitize_metadata(metadata)
        doc = Document(
            id=row["id"], page_content=row["page_content"], metadata=metadata
        )
        documents.append(doc)
    return documents


def load_parquet_files(files: List[Path]) -> List[pd.DataFrame]:
    """
    Load multiple parquet files into pandas DataFrames.

    Args:
        files: List of Path objects pointing to parquet files

    Returns:
        List of DataFrames, one for each successfully loaded parquet file
    """
    dataframes = []

    for idx, file_path in enumerate(files):
        try:
            df = pd.read_parquet(file_path)
            dataframes.append(df)
            print(f"{idx}: {file_path.name}")
        except Exception as e:
            print(f"Failed to load DataFrame from {file_path}: {str(e)}")

    return dataframes


def get_all_parquet_in_dir(dir_path: Path) -> List[Path]:
    """
    Get all parquet files in a directory.

    Args:
        dir_path: Path to directory to search

    Returns:
        List of Path objects for parquet files in the directory
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{dir_path}' does not exist")

    return sorted(p for p in dir_path.glob("*.parquet") if p.is_file())
