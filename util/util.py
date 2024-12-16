from typing import Dict, Any, List
from langchain_core.documents import Document


def serialize_document(document: Document) -> Dict[str, Any]:
    return {
        "page_content": document.page_content,
        "metadata": document.metadata
    }


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
