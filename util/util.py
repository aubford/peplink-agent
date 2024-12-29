from typing import Dict, Any, List
from langchain_core.documents import Document
from difflib import SequenceMatcher
from pandas import DataFrame


def serialize_document(document: Document) -> Dict[str, Any]:
    return {"page_content": document.page_content, "metadata": document.metadata}


def deduplicate_page_content(documents: List[Document]) -> List[Document]:
    """Remove documents with duplicate page_content while preserving order.

    Args:
        documents: List of serialized Document dicts

    Returns:
        List of deduplicated Document dicts
    """
    seen = set()
    deduped = []
    for doc in documents:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            deduped.append(doc)
    return deduped


def empty_document_dict(metadata: Dict[str, Any] = {}) -> Dict[str, Any]:
    return {"page_content": "", "metadata": metadata}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the identifier to be safe for use in filenames across all systems.
    Removes/replaces invalid filename characters using a standard approach.
    """
    # Common invalid filename characters including period
    invalid_chars = '<>:"/\\|?*@.-'

    # First handle leading special characters
    while filename and (filename[0] in invalid_chars or not filename[0].isalnum()):
        filename = filename[1:]

    # Then replace remaining invalid characters with underscore
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Replace spaces with underscore and remove any duplicate underscores
    filename = "_".join(filename.split())
    while "__" in filename:
        filename = filename.replace("__", "_")

    return filename.strip("_")


def similarity_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

def group_strings_return_longest(string_list: List[str], similarity_threshold: float = 0.75) -> List[str]:
    """
    Groups similar strings and returns the longest string from each group.
    Returns a list of representative strings, one per group.
    """
    if not string_list:
        return []

    groups: List[List[str]] = []

    for s in string_list:
        found_group = False
        for group in groups:
            if any(similarity_ratio(s, existing) > similarity_threshold for existing in group):
                group.append(s)
                found_group = True
                break
        if not found_group:
            groups.append([s])

    return [max(group, key=len) for group in groups]

def deduplicate_df_page_content(df: DataFrame, similarity_threshold: float = 0.85) -> DataFrame:
    """
    Deduplicate a dataframe based on the page_content column.
    Uses similarity_threshold to determine how to deduplicate.
    """
    count = df.shape[0]
    content = df["page_content"].tolist()
    unique_content = group_strings_return_longest(content, similarity_threshold)
    filtered = df[df["page_content"].isin(unique_content)]
    print(f"removed {count - filtered.shape[0]} entries to {filtered.shape[0]} unique entries")
    return filtered
