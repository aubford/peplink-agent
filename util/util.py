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

def similarity_count(string_list: List[str]) -> int:
    """
    Count the number of strings that are similar to another string in the list.
    """
    similar_count = 0

    for i in range(len(string_list)):
        for j in range(i + 1, len(string_list)):
            if similarity_ratio(string_list[i], string_list[j]) > 0.75:
                similar_count += 1
                break

    return similar_count


def get_dissimilar_content_count(df: DataFrame) -> int:
    content = df["page_content"].tolist()
    if len(content) == 0:
        return 0
    similar_count = similarity_count(content)
    print(f"total entries: {len(content)}")
    print(f"unique-ish entries: {len(content) - similar_count + 1}")
    return len(content) - similar_count + 1
