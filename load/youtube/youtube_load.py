# %%
from typing import List
from langchain.docstore.document import Document
from load.base_load import BaseLoad
from langchain.text_splitter import RecursiveCharacterTextSplitter
from util.deduplication_pipeline import DeduplicationPipeline
from util.util_main import dedupe_df_ids
import pandas as pd


class YoutubeLoad(BaseLoad):
    def __init__(self):
        super().__init__("youtube")

    def create_staging_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        combined_df = pd.concat(dfs)
        combined_df = dedupe_df_ids(combined_df)
        deduped = self.deduplication_pipeline.run(
            combined_df, precision_threshold=0.75, precision_ngram=1
        )
        return deduped

    def load_docs(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=300, length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs


loader = YoutubeLoad()

# %%

loader.load()

# %%

loader.staging_to_vector_store()
