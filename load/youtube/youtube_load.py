import pandas as pd
from pathlib import Path
from load.base_load import BaseLoad
from langchain.text_splitter import RecursiveCharacterTextSplitter


class YoutubeLoad(BaseLoad):
    def __init__(self):
        super().__init__("youtube")

    def load_file(self, file_path: Path):
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len
        )

        documents = self.parquet_to_documents(file_path)
        split_docs = text_splitter.split_documents(documents)
        self.stage_documents(split_docs)


if __name__ == "__main__":
    youtube_load = YoutubeLoad()
    youtube_load.load()
