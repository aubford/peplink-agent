from abc import abstractmethod
from typing import Any, Dict
from config import global_config, ConfigType
import pandas as pd
from pathlib import Path
from langchain.docstore.document import Document
from typing import List
from types import SimpleNamespace
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from config.logger import RotatingFileLogger

index_namespaces = SimpleNamespace(PEPWAVE="pepwave", NETWORKING="networking")
cat = 'test'


class BaseLoad:
    """Base class for all data transformers."""

    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.index_namespaces = index_namespaces
        self.logger = RotatingFileLogger(name=f"load_{self.folder_name}")
        self.vector_store: PineconeVectorStore = self.initialize_pinecone_index()

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    @abstractmethod
    def load_file(self, data: Dict[str, Any]) -> pd.DataFrame:
        pass

    def initialize_pinecone_index(self) -> PineconeVectorStore:
        """Initialize or create Pinecone index for pepwave namespace."""
        pc = Pinecone(api_key=self.config.get("PINECONE_API_KEY"))

        # Check if index exists
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if "pepwave" not in existing_indexes:
            # Create new index
            pc.create_index(name="pepwave", dimension=1536,  # OpenAI embeddings dimension
                            metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            self.logger.info("Created new Pinecone index: pepwave")

        # Initialize the index
        index = pc.Index("pepwave")
        vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())
        self.logger.info("Initialized Pinecone vector store")
        return vector_store

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
        if not str(file_path).endswith(".parquet"):
            raise FileNotFoundError(f"File {file_path} is not a parquet file")

        df = pd.read_parquet(file_path)

        if df.empty:
            raise ValueError(f"File {file_path} is empty")

        return df

    def df_to_documents(self, df: pd.DataFrame) -> List[Document]:
        documents = []
        for _, row in df.iterrows():
            metadata = row.drop(["page_content", "id"]).to_dict()
            metadata['record_id'] = row["id"]
            doc = Document(id=row["id"], page_content=row["page_content"], metadata=metadata)
            documents.append(doc)
        return documents

    def parquet_to_documents(self, file_path: Path) -> List[Document]:
        df = self.parquet_to_df(file_path)
        return self.df_to_documents(df)

    def log_documents(self, docs: List[Document]) -> None:
        doc = docs[0]
        print(f"Storing {len(docs)} documents.")
        print(f"First document:")
        self.logger.info(f"Doc.id: {doc.id}")
        self.logger.info(f"Metadata: \n{doc.metadata}\n")
        self.logger.info(f"Content: \n{doc.page_content}")
        self.logger.info("\n\n-----------")

    def store_documents(self, docs: List[Document], namespace: str = index_namespaces.PEPWAVE) -> None:
        """Store documents in vector store."""
        self.log_documents(docs)
        self.vector_store.add_documents(docs, namespace=namespace)
