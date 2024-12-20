from typing import Dict, Any, List
from langchain_core.documents import Document


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
