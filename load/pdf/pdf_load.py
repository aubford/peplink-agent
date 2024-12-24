from typing import List
from langchain.docstore.document import Document
import pandas as pd
from load.base_load import BaseLoad
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader


class PdfLoad(BaseLoad):
    def __init__(self):
        super().__init__("pdf")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

    def stage_documents(self, docs: List[Document]) -> None:
        for doc in docs:
            print(f"\n{doc.page_content}\n-------\n")

    def load_file(self, file_path: str) -> List[Document]:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()

        # Inspect the extracted elements
        for doc in documents:
            print(f"Page {doc.metadata.get('page')}")
            print(f"Type: {type(doc.metadata.get('plumber_table'))}")  # Will be None if not a table
            print(f"Content: {doc.page_content[:100]}...\n")

        return documents


if __name__ == "__main__":
    pdf_load = PdfLoad()
    pdf_load.load()
