# %%

from typing import List
from langchain.docstore.document import Document
from load.base_load import BaseLoad
import pandas as pd


class MongoLoad(BaseLoad):
    folder_name = "mongo"

    def __init__(self):
        super().__init__()

    def create_merged_df(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets into a single dataframe and perform operations."""
        df = pd.concat(dfs)
        df = self.ner(df)
        return df

    def load_docs(self, documents: List[Document]) -> List[Document]:
        # Remove unwanted metadata fields
        for doc in documents:
            doc.metadata.pop("post_tags", None)
            doc.metadata.pop("post_category_id", None)
        return documents
