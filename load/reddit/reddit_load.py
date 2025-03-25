from langchain_core.documents import Document
from load.base_load import BaseLoad
from load.synthetic_data_loaders import ForumSyntheticDataLoader
import pandas as pd


class RedditLoad(BaseLoad, ForumSyntheticDataLoader):
    folder_name = "reddit"

    def __init__(self):
        super().__init__()

    def create_merged_df(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets into a single dataframe and perform operations."""
        df = pd.concat(dfs)
        df = self.normalize_columns(df)
        df = self.ner(df)
        return df

    def load_docs(self, documents: list[Document]) -> list[Document]:
        """
        Create a batch job for documents.

        Args:
            documents: List of documents to process

        Returns:
            Documents.
        """
        self.create_batch_job(documents)
        return documents
