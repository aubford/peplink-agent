from typing import Dict, Any
from langchain_core.documents import Document


def serialize_document(document: Document) -> Dict[str, Any]:
    return {
        "page_content": document.page_content,
        "metadata": document.metadata
    }
