from typing import List
from langchain.docstore.document import Document
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

    def load_docs(self, documents: List[Document]) -> List[Document]:
        loader = PDFPlumberLoader(documents)
        documents = loader.load()

        for doc in documents:
            print(f"Page {doc.metadata.get('page')}")
            print(
                f"Type: {type(doc.metadata.get('plumber_table'))}"
            )  # Will be None if not a table
            print(f"Content: {doc.page_content[:100]}...\n")

        return documents


if __name__ == "__main__":
    pdf_load = PdfLoad()
    pdf_load.load()
