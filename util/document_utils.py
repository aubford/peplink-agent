import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any
from langchain_core.documents import Document
from pathlib import Path


def deserialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays and JSON serializable strings in the metadata dictionary to Python objects."""
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif isinstance(value, str) and (
            (value.startswith('{') and value.endswith('}'))
            or (value.startswith('[') and value.endswith(']'))
        ):
            try:
                sanitized[key] = json.loads(value)
            except json.JSONDecodeError:
                sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized


def df_to_documents(df: pd.DataFrame) -> List[Document]:
    """Convert a DataFrame to a list of Document objects."""
    documents: List[Document] = []
    for _, row in df.iterrows():
        metadata = row.drop(["page_content"]).to_dict()
        metadata = deserialize_metadata(metadata)
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


def document_to_dict(doc: Document) -> Dict[str, Any]:
    """Convert a Document object to a dict for state serialization."""
    result = {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }
    if hasattr(doc, "id") and doc.id is not None:
        result["id"] = doc.id
    return result


def documents_to_dicts(docs: List[Document]) -> List[Dict[str, Any]]:
    """Convert a list of Document objects to dicts for state serialization."""
    return [document_to_dict(doc) for doc in docs]


def dict_to_document(doc_dict: Dict[str, Any]) -> Document:
    """Convert a dict back to a Document object."""
    doc_kwargs = {
        "page_content": doc_dict.get("page_content", ""),
        "metadata": doc_dict.get("metadata", {}),
    }
    if "id" in doc_dict:
        doc_kwargs["id"] = doc_dict["id"]
    return Document(**doc_kwargs)
