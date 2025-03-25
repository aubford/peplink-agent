# %%
from typing import List
from langchain_core.documents import Document
from load.base_load import BaseLoad
import pandas as pd


class WebLoad(BaseLoad):
    folder_name = "web"

    def __init__(self):
        super().__init__()

    def create_merged_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        deduped_dfs = []
        for df in dfs:
            deduped = self.deduplication_pipeline.run(
                df, precision_threshold=0.8, precision_ngram=1
            )
            deduped_dfs.append(deduped)
        staging_df = pd.concat(deduped_dfs)
        staging_df = staging_df.set_index("id", drop=False, verify_integrity=True)
        return staging_df

    def load_docs(self, documents: List[Document]) -> List[Document]:
        text_splitter = self._get_default_text_splitter()
        split_docs = self._split_docs(documents, text_splitter)
        return split_docs
