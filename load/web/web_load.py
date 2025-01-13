from typing import List
from langchain.docstore.document import Document
from load.base_load import BaseLoad
from util.nlp import deduplication_pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd


class WebLoad(BaseLoad):
    def __init__(self):
        super().__init__("web")

    def create_staging_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        deduped_dfs = []
        for df in dfs:
            deduped = deduplication_pipeline(
                df, precision_threshold=0.80, precision_ngram=1
            )
            deduped_dfs.append(deduped)
        staging_df = pd.concat(deduped_dfs)
        staging_df = staging_df.set_index("id", drop=False, verify_integrity=True)
        return staging_df

    def load_docs(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100, length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs


if __name__ == "__main__":
    youtube_load = WebLoad()
    youtube_load.load()
