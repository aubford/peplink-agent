# simply merge the parquet files in data/html/documents into a staging.parquet??
# maybe we just skip staging.parquet and just upload the parquet files in data/html/documents straight to vectorstore


from typing import List
from langchain.docstore.document import Document
from load.base_load import BaseLoad
import pandas as pd

class HtmlLoad(BaseLoad):
    def __init__(self):
        super().__init__("html")

    def create_staging_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(dfs)

    def load_docs(self, file_path: str) -> List[Document]:
        df = self.parquet_to_df(file_path)
        print(df)


if __name__ == "__main__":
    loader = HtmlLoad()
    loader.load()
