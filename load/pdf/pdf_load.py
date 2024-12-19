import pandas as pd
from pathlib import Path
from load.base_load import BaseLoad
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class PdfLoad(BaseLoad):
    def __init__(self):
        super().__init__("pdf")

    def load_file(self, file_path: Path) -> pd.DataFrame:

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        print(documents)

        # split_docs = text_splitter.split_documents(documents)
        # self.store_documents(split_docs)


if __name__ == "__main__":
    pdf_load = PdfLoad()
    pdf_load.load()
