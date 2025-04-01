import pandas as pd
from typing import List
from langchain_core.documents import Document
from load.base_load import BaseLoad
from langchain.text_splitter import RecursiveCharacterTextSplitter
from util.util_main import dedupe_df_ids
from load.synthetic_data_loaders import YouTubeSyntheticDataLoader


class YoutubeLoad(BaseLoad, YouTubeSyntheticDataLoader):
    folder_name = "youtube"

    def __init__(self):
        super().__init__()

    def create_merged_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(dfs)
        df = dedupe_df_ids(df)
        df = self.deduplication_pipeline.run(
            df, precision_threshold=0.75, precision_ngram=1
        )
        df = self.normalize_columns(df)
        df = self.ner(df)
        return df

    def load_docs(self, documents: List[Document]) -> List[Document]:
        """
        Create a batch job for documents and perform text splitting.

        Args:
            documents: List of documents to process

        Returns:
            Documents after text splitting.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=3000
        )
        split_docs = self._split_docs(documents, text_splitter)
        # primary content should be the split page content for YouTube
        for doc in split_docs:
            doc.metadata["primary_content"] = doc.page_content

        self.create_batchfile(split_docs, max_tokens=800)
        return split_docs
