import pandas as pd
from pathlib import Path
from load.base_load import BaseLoad
from langchain_pinecone import PineconeVectorStore

class YoutubeLoad(BaseLoad):
    def __init__(self):
        super().__init__("youtube")

    def load_file(self, file_path: Path) -> pd.DataFrame:
        df = self.parquet_to_df(file_path)

        documents = self.df_to_documents(df)
        
        self.store_documents(documents)

        return df
