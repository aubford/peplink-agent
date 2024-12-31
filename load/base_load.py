from abc import abstractmethod
from typing import List
from config import global_config, ConfigType
import pandas as pd
from pathlib import Path
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from config.logger import RotatingFileLogger
from langchain.vectorstores import VectorStore
from types import SimpleNamespace
from uuid import uuid4

index_namespaces = SimpleNamespace(PEPWAVE="pepwave", NETWORKING="networking")

class BaseLoad:
    """Base class for all data transformers."""

    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.index_namespaces = index_namespaces
        self.logger = RotatingFileLogger(name=f"load_{self.folder_name}")
        self.vector_store: VectorStore | None = None
        self.staging_path = Path("load") / self.folder_name / "staging.parquet"

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    @abstractmethod
    def load_file(self, file_path: str) -> List[Document]:
        pass

    def initialize_pinecone_index(self) -> None:
        """Initialize or create Pinecone index for pepwave namespace."""
        pc = Pinecone(api_key=self.config.get("PINECONE_API_KEY"))

        # Check if index exists
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if "pepwave" not in existing_indexes:
            # Create new index
            pc.create_index(
                name="pepwave",
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            self.logger.info("Created new Pinecone index: pepwave")

        # Initialize the index
        index = pc.Index("pepwave")
        vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())
        self.logger.info("Initialized Pinecone vector store")
        self.vector_store = vector_store

    def staging_to_vector_store(self, namespace: str = index_namespaces.PEPWAVE) -> None:
        """Upload staged documents to Pinecone."""

        if self.vector_store is None:
            self.initialize_pinecone_index()
        if not self.staging_path.exists():
            raise FileNotFoundError(f"No staged documents found at {self.staging_path}")

        docs = self.parquet_to_documents(self.staging_path)
        self.log_documents(docs)
        self.vector_store.add_documents(docs, namespace=namespace)
        self.logger.info(f"Uploaded {len(docs)} documents to Pinecone namespace: {namespace}")

    def stage_documents(self, docs: List[Document]) -> None:
        """Store documents to local parquet file."""
        self.log_documents(docs)

        records = []
        for doc in docs:
            record = {
                "id": str(uuid4()),
                "page_content": doc.page_content,
                **doc.metadata,
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.to_parquet(self.staging_path)
        self.logger.info(f"Saved {len(docs)} documents to {self.staging_path}")

    def load(self) -> None:
        documents_dir = Path("data") / self.folder_name / "documents"
        # todo: to enable namespaces, iterate over subfolders in documents_dir

        if not documents_dir.exists():
            raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")

        if self.staging_path.exists():
            raise FileNotFoundError(f"Staging data file already exists at {self.staging_path}")

        all_documents = []
        for file_path in documents_dir.glob("*"):
            # Skip system files
            if file_path.name.startswith("."):
                continue

            try:
                print(f"--------------- Loading file: {file_path}---------\n\n")
                docs = self.load_file(file_path)
                all_documents.extend(docs)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}")
                raise e
        self.stage_documents(all_documents)

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
            metadata["record_id"] = row["id"]
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
