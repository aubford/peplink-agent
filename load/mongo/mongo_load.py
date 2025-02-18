# %%

from typing import List
from langchain.docstore.document import Document
from load.base_load import BaseLoad
import pandas as pd


class MongoLoad(BaseLoad):
    def __init__(self):
        super().__init__("mongo")

    def create_merged_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(dfs)
        # Drop rows with less than 80 words
        df = df[df["page_content"].str.strip().str.split(r"\s+").str.len() >= 80]
        return df

    def load_docs(self, documents: List[Document]) -> List[Document]:
        # Remove unwanted metadata fields
        for doc in documents:
            doc.metadata.pop("topic_tags", None)
            doc.metadata.pop("topic_category_id", None)
        return documents


loader = MongoLoad()
loader.load()

# %%

# loader.staging_to_vector_store()
