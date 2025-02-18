# %%
from typing import List
from langchain.docstore.document import Document
from load.base_load import BaseLoad
from langchain.text_splitter import RecursiveCharacterTextSplitter
from util.util_main import dedupe_df_ids
import pandas as pd


class YoutubeLoad(BaseLoad):
    folder_name = "youtube"

    def __init__(self):
        super().__init__()

    def create_merged_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        combined_df = pd.concat(dfs)
        combined_df = dedupe_df_ids(combined_df)
        deduped = self.deduplication_pipeline.run(
            combined_df, precision_threshold=0.75, precision_ngram=1
        )
        return deduped

    def load_docs(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=3000
        )
        split_docs = self._split_docs(documents, text_splitter)
        return split_docs
