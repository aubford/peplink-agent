# simply merge the parquet files in data/html/documents into a staging.parquet??
# maybe we just skip staging.parquet and just upload the parquet files in data/html/documents straight to vectorstore

# %%

from typing import List
from langchain.docstore.document import Document
from load.base_load import BaseLoad
import pandas as pd
import sys


class HtmlLoad(BaseLoad):
    def __init__(self):
        super().__init__("html")

    def create_staging_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(dfs)
        # Add section as header in page_content.  Do this here instead of transform so we can experiment.
        mask = df["section"].str.strip().astype(bool)
        df.loc[mask, "page_content"] = df.loc[mask, "section"] + " \n " + df.loc[mask, "page_content"]
        return df

    def load_docs(self, documents: List[Document]) -> List[Document]:
        return documents


if __name__ == "__main__":
    loader = HtmlLoad()
    loader.load()
