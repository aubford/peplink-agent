from typing import List
from langchain.docstore.document import Document
from load.base_load import BaseLoad
from langchain.text_splitter import RecursiveCharacterTextSplitter


class WebLoad(BaseLoad):
    def __init__(self):
        super().__init__("web")

    def load_docs(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100, length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs


if __name__ == "__main__":
    youtube_load = WebLoad()
    youtube_load.load()
