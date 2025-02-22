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
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter

# Split on sentences before spaces. Will false positive on initials like J. Robert Oppenheimer so room for improvement.
DEFAULT_TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+(?=[A-Z])", " ", ""]

class BaseLoad:
    """Base class for all data transformers."""

    folder_name: str = NotImplemented

    def __init__(self):
        if not isinstance(self.folder_name, str):
            raise TypeError(
                f"folder_name must be defined as a string in the subclass, got {type(self.folder_name)}"
            )
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
                **doc.metadata,
                "id": doc.id,
                "page_content": doc.page_content,
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

    def _get_default_text_splitter(
        self, chunk_size: int = 3000, chunk_overlap: int = 500
    ) -> TextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=DEFAULT_TEXT_SPLITTER_SEPARATORS,
            add_start_index=True,
            is_separator_regex=True,
        )

    @staticmethod
    def _split_docs(docs: List[Document], splitter: TextSplitter) -> List[Document]:
        # Store original IDs in metadata before splitting
        for doc in docs:
            doc.metadata["record_id"] = doc.id

        split_docs = splitter.split_documents(docs)
        for doc in split_docs:
            doc.id = str(uuid4())
        return split_docs

    def load_from_merged(self) -> None:
        merged_path = self.staging_folder / "merged.parquet"
        if not merged_path.exists():
            raise FileNotFoundError(f"No merged document found at {merged_path}")

        df = self._parquet_to_df(merged_path)
        all_documents = df_to_documents(df)
        staging_docs = self.load_docs(all_documents)
        self._stage_documents(staging_docs)

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
        for df in dfs:
            df["type"] = self.folder_name
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

    @classmethod
    def get_artifact(cls, select_merged: bool = False) -> pd.DataFrame:
        """
        Load the staging.parquet file for this loader into a pandas DataFrame.

        Returns:
            DataFrame containing the staged documents

        Raises:
            FileNotFoundError: If staging.parquet does not exist
        """
        filename = "merged.parquet" if select_merged else "staging.parquet"
        staging_path = Path("load") / cls.folder_name / filename
        if not staging_path.exists():
            raise FileNotFoundError(f"No staging file found at {staging_path}")
        return pd.read_parquet(staging_path)
