from abc import abstractmethod
from typing import Any, Dict
from config import global_config, ConfigType
import pandas as pd
from pathlib import Path
from langchain.docstore.document import Document
from typing import List
from types import SimpleNamespace
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from config.logger import RotatingFileLogger

index_namespaces = SimpleNamespace(
    PEPWAVE="pepwave",
    NETWORKING="networking"
)


class BaseLoad:
    """Base class for all data transformers."""

    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.index_namespaces = index_namespaces
        self.logger = RotatingFileLogger(name=f"load_{self.folder_name}")

        pc = Pinecone(api_key=self.config.get("PINECONE_API_KEY"))
        index = pc.Index("pepwave")
        self.vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    @abstractmethod
    def load_file(self, data: Dict[str, Any]) -> pd.DataFrame:
        pass

    def load(self) -> None:
        documents_dir = Path("data") / self.folder_name / "documents"

        if not documents_dir.exists():
            raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")

        for file_path in documents_dir.glob("*"):
            try:
                self.load_file(file_path)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}")
                raise e

    def parquet_to_df(self, file_path: Path) -> pd.DataFrame:
        if not str(file_path).endswith('.parquet'):
            raise FileNotFoundError(f"File {file_path} is not a parquet file")

        df = pd.read_parquet(file_path)

        if df.empty:
            raise ValueError(f"File {file_path} is empty")

        return df

    def df_to_documents(self, df: pd.DataFrame) -> List[Document]:
        documents = []
        for _, row in df.iterrows():
            metadata = row.drop(["page_content", "id"]).to_dict()
            doc = Document(
                id=row["id"],
                page_content=row["page_content"],
                metadata=metadata
            )
            documents.append(doc)
        return documents

    def parquet_to_documents(self, file_path: Path) -> List[Document]:
        df = self.parquet_to_df(file_path)
        return self.df_to_documents(df)

    def store_documents(self, documents: List[Document], namespace: str = index_namespaces.PEPWAVE) -> None:
        """Store documents in vector store."""
        self.vector_store.add_documents(documents, namespace=namespace)
