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
from langchain.text_splitter import TextSplitter


class BaseLoad:
    """Base class for all data transformers."""

    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.index_name = self.config.get("VERSIONED_PINECONE_INDEX_NAME")
        self.logger = RotatingFileLogger(name=f"load_{self.folder_name}")
        self.vector_store: VectorStore | None = None
        self.staging_folder = Path("load") / self.folder_name
        self.staging_path = self.staging_folder / "staging.parquet"
        self.deduplication_pipeline = DeduplicationPipeline(self.folder_name)

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    def create_merged_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(dfs)

    def load_docs(self, documents: List[Document]) -> List[Document]:
        return documents

    @staticmethod
    def _simple_dedupe(df: pd.DataFrame) -> None:
        df.drop_duplicates(subset=["page_content"], keep="first", inplace=True)

    def _initialize_pinecone_index(self) -> None:
        """Initialize or create Pinecone index. Note: DO NOT USE NAMESPACES!"""
        pc = Pinecone(api_key=self.config.get("PINECONE_API_KEY"))

        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if self.index_name not in existing_indexes:
            # Create new index
            pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            self.logger.info(f"Created new Pinecone index: {self.index_name}")

        # Initialize the index
        index = pc.Index(self.index_name)
        vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())
        self.logger.info(
            f"Initialized Pinecone vector store for index: {self.index_name}"
        )
        self.vector_store = vector_store

    def staging_to_vector_store(self) -> None:
        """Upload staged documents to a new, versioned Pinecone index. DON'T FORGET TO CHANGE VERSION IN .env!!!"""
        if self.vector_store is None:
            self._initialize_pinecone_index()
        if not self.staging_path.exists():
            raise FileNotFoundError(f"No staged documents found at {self.staging_path}")

        docs = self._parquet_to_documents(self.staging_path)
        self._log_documents(docs)
        self.vector_store.add_documents(docs)
        self.logger.info(
            f"Uploaded {len(docs)} documents to Pinecone index: {self.index_name}"
        )

    def _stage_documents(self, docs: List[Document]) -> None:
        self._log_documents(docs)

        records = []
        for doc in docs:
            record = {
                "id": doc.id,
                "page_content": doc.page_content,
                "type": self.folder_name,
                **doc.metadata,
            }
            records.append(record)

        df = pd.DataFrame(records).set_index("id", drop=False, verify_integrity=True)
        df.to_parquet(self.staging_path)
        self.logger.info(f"Saved {len(docs)} documents to {self.staging_path}")

    def _write_merged_df_artifact(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.staging_folder / "merged.parquet")
        self.logger.info(
            f"Saved {len(df)} documents to {self.staging_folder / 'merged.parquet'}"
        )

    @staticmethod
    def _split_docs(docs: List[Document], splitter: TextSplitter) -> List[Document]:
        split_docs = splitter.split_documents(docs)
        for doc in split_docs:
            # apply the ID from the original document as "record_id" so we can provide a unique uuid for each
            # split document
            doc.metadata["record_id"] = doc.id
            doc.id = str(uuid4())
        return split_docs

    def load(self) -> None:
        documents_dir = Path("data") / self.folder_name / "documents"

        if not documents_dir.exists():
            raise FileNotFoundError(
                f"Documents directory does not exist: {documents_dir}"
            )

        if self.staging_path.exists():
            raise FileNotFoundError(
                f"Staging data file already exists at {self.staging_path}"
            )

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
        staging_df = self.create_merged_df(dfs)
        self._simple_dedupe(staging_df)
        self._write_merged_df_artifact(staging_df)
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
