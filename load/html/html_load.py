# simply merge the parquet files in data/html/documents into a staging.parquet??
# maybe we just skip staging.parquet and just upload the parquet files in data/html/documents straight to vectorstore


from typing import List
from langchain.docstore.document import Document
from load.base_load import BaseLoad


class HtmlLoad(BaseLoad):
    def __init__(self):
        super().__init__("html")
        
        
    # todo: temp for testing
    def stage_documents(self, docs: List[Document]) -> None:
        for doc in docs:
            print(f"\n{doc.page_content}\n-------\n")

    def load_file(self, file_path: str) -> List[Document]:
        df = self.parquet_to_df(file_path)
        print(df)


if __name__ == "__main__":
    loader = HtmlLoad()
    loader.load()
