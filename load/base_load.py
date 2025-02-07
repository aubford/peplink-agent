from abc import abstractmethod
from datetime import datetime
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
from uuid import uuid4
from util.deduplication_pipeline import DeduplicationPipeline
from util.document_utils import df_to_documents


class BaseLoad:
    """Base class for all data transformers."""

    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.logger = RotatingFileLogger(name=f"load_{self.folder_name}")
        self.vector_store: VectorStore | None = None
        self.staging_path = Path("load") / self.folder_name / "staging.parquet"
        self.deduplication_pipeline = DeduplicationPipeline(self.folder_name)

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    @abstractmethod
    def create_staging_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_docs(self, documents: List[Document]) -> List[Document]:
        pass

    @staticmethod
    def _simple_dedupe(df: pd.DataFrame) -> None:
        df.drop_duplicates(subset=["page_content"], keep="first", inplace=True)

    def _initialize_pinecone_index(self, versioned_index_name: str) -> None:
        """Initialize or create Pinecone index. Note: DO NOT USE NAMESPACES!

        Args:
            versioned_index_name (str): The name of the index to upload to. We should version these so we can A/B test different index versions.
        """
        pc = Pinecone(api_key=self.config.get("PINECONE_API_KEY"))

        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if versioned_index_name not in existing_indexes:
            # Create new index
            pc.create_index(
                name=versioned_index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            self.logger.info(f"Created new Pinecone index: {versioned_index_name}")

        # Initialize the index
        index = pc.Index(versioned_index_name)
        vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())
        self.logger.info(f"Initialized Pinecone vector store for index: {versioned_index_name}")
        self.vector_store = vector_store

    def staging_to_vector_store(self, versioned_index_name: str) -> None:
        """Upload staged documents to a new, versioned Pinecone index.

        Args:
            versioned_index_name (str): The name of the index to upload to. We should version these so we can A/B test different index versions.
        """
        timestamp = datetime.now().strftime("%y_%m_%d")
        versioned_index_name = f"{versioned_index_name}__{timestamp}"

        if self.vector_store is None:
            self._initialize_pinecone_index(versioned_index_name)
        if not self.staging_path.exists():
            raise FileNotFoundError(f"No staged documents found at {self.staging_path}")

        docs = self._parquet_to_documents(self.staging_path)
        self._log_documents(docs)
        self.vector_store.add_documents(docs)
        self.logger.info(f"Uploaded {len(docs)} documents to Pinecone index: {versioned_index_name}")

    def _stage_documents(self, docs: List[Document]) -> None:
        self._log_documents(docs)

        records = []
        for doc in docs:
            record = {
                "id": str(uuid4()),
                "page_content": doc.page_content,
                "type": self.folder_name,
                **doc.metadata,
            }
            records.append(record)

        df = pd.DataFrame(records).set_index("id", drop=False, verify_integrity=True)
        df.to_parquet(self.staging_path)
        self.logger.info(f"Saved {len(docs)} documents to {self.staging_path}")

    def load(self) -> None:
        documents_dir = Path("data") / self.folder_name / "documents"

        if not documents_dir.exists():
            raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")

        if self.staging_path.exists():
            raise FileNotFoundError(f"Staging data file already exists at {self.staging_path}")

        dfs = []
        for file_path in documents_dir.glob("*"):
            # Skip system files
            if file_path.name.startswith("."):
                continue

            try:
                self.logger.info(f"Loading file: {file_path}")
                df = self._parquet_to_df(file_path)
                dfs.append(df)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}")
                raise e

        # Combine all dataframes and convert to documents
        staging_df = self.create_staging_df(dfs)
        self._simple_dedupe(staging_df)
        all_documents = df_to_documents(staging_df)
        staging_docs = self.load_docs(all_documents)
        self._stage_documents(staging_docs)

    @staticmethod
    def _parquet_to_df(file_path: Path) -> pd.DataFrame:
        if not str(file_path).endswith(".parquet"):
            raise FileNotFoundError(f"File {file_path} is not a parquet file")

        df = pd.read_parquet(file_path)
        df = df.set_index("id", drop=False, verify_integrity=True)

        if df.empty:
            raise ValueError(f"File {file_path} is empty")

        return df

    def _parquet_to_documents(self, file_path: Path) -> List[Document]:
        df = self._parquet_to_df(file_path)
        return df_to_documents(df)

    def _log_documents(self, docs: List[Document]) -> None:
        doc = docs[0]
        self.logger.info("=" * 100)
        self.logger.info(f"\n\nFinal documents: {len(docs)}")
        self.logger.info(f"First document:")
        self.logger.info(f"Doc.id: {doc.id}")
        self.logger.info(f"Metadata: \n{doc.metadata}\n")
        self.logger.info(f"Content: \n{doc.page_content}")
        self.logger.info("\n\n-----------")
        self.logger.info("=" * 100)
