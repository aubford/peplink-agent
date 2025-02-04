# %%
from typing import List
from langchain.docstore.document import Document
from load.base_load import BaseLoad
from util.deduplication_pipeline import DeduplicationPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd


class WebLoad(BaseLoad):
    def __init__(self):
        super().__init__("web")

    def create_staging_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100, length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs


loader = WebLoad()

# %%

loader.load()

# %%

loader.staging_to_vector_store()
