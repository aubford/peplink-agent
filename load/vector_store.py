from typing import Literal, Optional
from config import global_config
import pandas as pd
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from pinecone.data.index import Index
from util.util_main import drop_embedding_columns
from load.document_index import DocumentIndex
import json


class VectorStore:
    """Base class for all data transformers."""

    this_dir: Path = Path(__file__).parent

    def __init__(self, index_name: str):
        self.index_name = index_name
        self.vector_store: Optional[Index] = None
        self.postprocess_path: Path = DocumentIndex.document_index_path

    def _initialize_pinecone_index(self, alt_index: str | None = None) -> None:
        """Initialize or create Pinecone index. Note: DO NOT USE NAMESPACES!"""
        if self.vector_store is not None:
            return

        pc = Pinecone(api_key=global_config.get("PINECONE_API_KEY"))

        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        index_name = (
            f"{self.index_name}-{alt_index.replace('_', '-')}"
            if alt_index
            else self.index_name
        )
        # Limit index name to 45 characters
        index_name = index_name[:45]

        if index_name not in existing_indexes:
            # Create new index
            pc.create_index(
                name=index_name,
                dimension=3072,  # OpenAI embeddings dimension for text-embedding-3-large
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Created new Pinecone index: {index_name}")

        # Initialize the index
        self.vector_store = pc.Index(index_name)
        print(f"Initialized Pinecone vector store for index: {index_name}")

    def _clean_metadata_for_vector_store(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean duplicate columns for vector store."""
        df = df.drop(
            columns=[
                "post_title",
                "post_content",
                "comment_content",
                "page_content_embedding_clean",
                "dirty_page_content",
                "token_count",
                "entities_dirty",
            ],
            errors="ignore",
        )
        # Replace all null values with empty string
        df = df.fillna("")
        return df

    def staging_to_vector_store(
        self,
        alt_column: Literal[
            "page_content_embedding",
            "primary_content_embedding",
            "technical_summary_embedding",
            "title_embedding",
        ],
    ) -> None:
        """Upload staged documents to a new, versioned Pinecone index."""
        self._initialize_pinecone_index(alt_column)
        if not self.postprocess_path.exists():
            raise FileNotFoundError(
                f"No staged documents found at {self.postprocess_path}"
            )
        if self.vector_store is None:
            raise ValueError("Vector store initialization failed")

        df = self._parquet_to_df(self.postprocess_path)
        metadata_df = self._parquet_to_df(self.postprocess_path, drop_embeddings=True)
        metadata_df = self._clean_metadata_for_vector_store(metadata_df)

        ids = df.index.tolist()
        column = (
            alt_column
            if alt_column and alt_column in df.columns
            else "primary_content_embedding"
        )
        vectors = df[column].apply(json.loads).tolist()
        metadata_dict = metadata_df.to_dict(orient="records")

        docs = []
        for id, vector, metadata in zip(ids, vectors, metadata_dict):
            docs.append(
                {
                    "id": str(id),
                    "values": vector,
                    "metadata": metadata,
                }
            )

        print(f"Uploading {len(docs)}")
        self.vector_store.upsert(docs, batch_size=50)
        print(f"Uploaded {len(docs)} documents to Pinecone index: {self.index_name}")

    @staticmethod
    def _parquet_to_df(file_path: Path, drop_embeddings: bool = False) -> pd.DataFrame:
        if not str(file_path).endswith(".parquet"):
            raise FileNotFoundError(f"File {file_path} is not a parquet file")

        df = pd.read_parquet(file_path)
        df = df.set_index("id", drop=False, verify_integrity=True)

        if df.empty:
            raise ValueError(f"File {file_path} is empty")

        if drop_embeddings:
            df = drop_embedding_columns(df)

        return df

    def validate_pinecone_index(self) -> None:
        """Validate that all staging IDs exist in the Pinecone index."""
        self._initialize_pinecone_index()
        if not self.vector_store:
            raise ValueError("Vector store initialization failed")

        vector_store: Index = self.vector_store

        vector_store_staging_data = self._parquet_to_df(self.postprocess_path)
        staging_ids = vector_store_staging_data.index.tolist()

        # Get all IDs from Pinecone using pagination
        pinecone_ids = []

        try:
            for ids in vector_store.list():
                pinecone_ids.extend(ids)

            # Compare the sets
            missing_from_pinecone = set(staging_ids) - set(pinecone_ids)
            extra_in_pinecone = set(pinecone_ids) - set(staging_ids)

            if missing_from_pinecone or extra_in_pinecone:
                print(
                    f"Index validation failed:\n"
                    f"IDs missing from Pinecone: {len(missing_from_pinecone)}\n"
                    f"Extra IDs in Pinecone: {len(extra_in_pinecone)}"
                )

            # compare lengths
            if len(staging_ids) != len(pinecone_ids):
                print(
                    f"Index validation failed:\n"
                    f"Staging IDs: {len(staging_ids)}\n"
                    f"Pinecone IDs: {len(pinecone_ids)}"
                )

        except Exception as e:
            print(f"Error validating Pinecone index: {e}")
            raise
