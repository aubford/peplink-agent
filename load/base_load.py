import tiktoken
from config import global_config, ConfigType
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from config.logger import RotatingFileLogger
from uuid import uuid4
from util.deduplication_pipeline import DeduplicationPipeline
from util.document_utils import df_to_documents
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
import spacy
from load.html.html_util import get_settings_entities
from load.batch_manager import BatchManager
from util.util_main import (
    to_serialized_parquet,
    drop_embedding_columns,
    get_chunk_size,
)
import json

# Split on sentences before spaces. Will false positive on initials like J. Robert Oppenheimer so room for improvement.
DEFAULT_TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+(?=[A-Z])", " ", ""]

"""Standard columns in both merged.parquet and staging.parquet

- id: Id and index. Same between merged.parquet and staging.parquet except for youtube.
- page_content: The main content.
- type: The folder name of the data source in load dir (html, reddit, reddit_general, youtube, mongo).
- source_file: The file name from the data/ folder.
- subject_matter: The subject matter type of the data source (PEPWAVE, IT_NETWORKING, MOBILE_INTERNET).

Normalized Columns:
- title: Html section, post title, video title.
- primary_content: YouTube transcript, comment content, html section content.
Not in HTML:
- lead_content: YouTube description, topic content.
- score: YouTube video likes, comment likes.
- author_id: YouTube channel id, comment author id.
- author_name: YouTube channel title, comment author name.
"""

"""Synthetic Columns:
- entities: Entities in primary_content and lead_content. (HTML: Only primary_content)
    Inference:
    - When comparing a non-HTML with HTML: Compare with both HTML.entities and HTML.settings_entity_list.
    - Don't compare HTML with itself?
    - Always try to find an HTML relationship via this column.
From LLM: (EX: HTML)
- themes: Themes from primary_content and lead_content. (YouTube: Only primary_content)
- is_useful: Whether the primary_content contains useful technical information.
- technical_summary:
    - For YouTube: Summary of the primary_content.
    - For forums: Summary of the primary_content and lead_content. Use special prompt to explain the two documents and how to summarize them.
From Embedding Model:
- technical_summary_embedding: Embedding of the technical_summary.
- title_embedding: Embedding of the title column. (filter out short titles)
- primary_content_embedding: Embedding of the primary_content.
- lead_content_embedding: Embedding of the lead_content.
"""


class DeduplicationLockPipeline:
    """Reuse previous deduplication results to save time."""

    def __init__(self, lock_file_path: Path):
        df = pd.read_parquet(lock_file_path)
        df = df.set_index("id", drop=False, verify_integrity=True)
        self.valid_ids = set(df.index)

    def run(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Filter the input dataframe to only include rows with IDs in the valid_ids set.
        These IDs represent documents that were previously processed by the deduplication pipeline.
        """
        original_count = len(df)
        df = df[df.index.isin(self.valid_ids)]
        retained_count = len(df)
        print(
            f"Deduplication lock: Kept {retained_count} documents out of {original_count} original documents."
        )
        return df


class BaseLoad:
    """Base class for all data transformers."""

    folder_name: str = NotImplemented
    this_dir: Path = Path(__file__).parent

    def __init__(self):
        if not isinstance(self.folder_name, str):
            raise TypeError(
                f"folder_name must be defined as a string in the subclass, got {type(self.folder_name)}"
            )
        self.logger: RotatingFileLogger = RotatingFileLogger(
            name=f"load_{self.folder_name}"
        )
        self.staging_folder: Path = self.this_dir / self.folder_name
        self.staging_path: Path = self.staging_folder / "staging.parquet"
        self.synth_data_path: Path = self.staging_folder / "synth_data.parquet"
        deduplication_lock_path: Path = (
            self.staging_folder / "deduplication_lock.parquet"
        )
        self.deduplication_pipeline = (
            DeduplicationLockPipeline(deduplication_lock_path)
            if deduplication_lock_path.exists()
            else DeduplicationPipeline(self.folder_name)
        )
        self.embedding_model: OpenAIEmbeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        self.batch_manager: BatchManager = BatchManager(self.staging_folder)
        self.init_spacy()

    def apply_synth_data_to_staging(self):
        """Apply the generated data from LLM to the staging file."""
        synthetic_data = pd.read_parquet(self.synth_data_path)
        # Filter out rows where is_useful is False
        original_count = len(synthetic_data)
        synthetic_data = synthetic_data[synthetic_data["is_useful"] == True]
        filtered_count = len(synthetic_data)
        print(
            f"Filtered out {original_count - filtered_count} non-useful documents, {filtered_count} remaining."
        )
        # merge with staging
        staging = pd.read_parquet(self.staging_path)
        staging.drop(columns=synthetic_data.columns, inplace=True, errors="ignore")

        merged = pd.merge(staging, synthetic_data, left_index=True, right_index=True)
        # generate the title embeddings
        merged = self._generate_embeddings(merged, "title")
        merged = self._generate_embeddings(merged, "technical_summary")
        merged = self._generate_embeddings(merged, "primary_content")
        to_serialized_parquet(merged, self.staging_path)

    def create_synth_data_from_batch_results(self) -> None:
        """
        Create a parquet file from the batch results JSON file.
        Extracts content from each result and flattens into a dataframe structure.
        """
        output_file_name = self.batch_manager.output_file_name
        if not output_file_name.exists():
            raise FileNotFoundError(f"Results file not found at {output_file_name}")

        with open(output_file_name) as f:
            data = json.load(f)

        # Extract and process the data
        processed_data = []
        for item in data:
            record = {
                "id": item["custom_id"],
            }

            # Extract message content and parse it
            try:
                message = item["response"]["body"]["choices"][0]["message"]
                content = json.loads(message["content"])
                record.update(content)
            except (KeyError, json.JSONDecodeError) as e:
                self.logger.error(
                    f"Error processing item {item.get('custom_id', 'unknown')}: {e}"
                )
                continue

            processed_data.append(record)

        # Create DataFrame and save as parquet
        df = pd.DataFrame(processed_data)
        df = df.set_index("id", verify_integrity=True)
        to_serialized_parquet(df, self.synth_data_path)

    def _generate_embeddings(
        self, df: pd.DataFrame, column_name: str, chunk_size: int = 1000
    ) -> pd.DataFrame:
        """Generate embeddings for a specific column in a dataframe."""
        texts = df[column_name].tolist()
        optimal_chunk_size = get_chunk_size(texts)
        print(f"Using chunk size {optimal_chunk_size}")
        chunk_size = min(chunk_size, optimal_chunk_size)

        embeddings = self.embedding_model.embed_documents(texts, chunk_size=chunk_size)
        df[f"{column_name}_embedding"] = embeddings
        return df

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    def create_merged_df(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets into a single dataframe and perform operations like deduplication."""
        return pd.concat(dfs)

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count the tokens in a text."""
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(str(text)))

    def load_docs(self, documents: list[Document]) -> list[Document]:
        """Perform final operations like text splitting, synthetic data generation, and knowledge_graph final product for upload to vector store."""
        return documents

    def init_spacy(self):
        """Initialize SpaCy model."""
        nlp = spacy.load("en_core_web_trf")
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        all_entities = get_settings_entities()
        patterns = [
            {"label": "PEPWAVE_SETTINGS_ENTITY", "pattern": entity, "id": "LOWER"}
            for entity in all_entities
        ]
        ruler.add_patterns(patterns)
        self.nlp = nlp

    def extract_entities(self, text: str) -> list[str]:
        """Use SpaCy NER to extract entities from text, including custom PEPWAVE_SETTINGS_ENTITY."""
        if not text or pd.isna(text):
            return []

        # Process the text with the pipeline that includes our EntityRuler
        doc = self.nlp(text)
        entities: set[str] = {
            ent.text
            for ent in doc.ents
            if ent.label_
            in {
                "URL",
                "FAC",
                "ORG",
                "GPE",
                "PRODUCT",
                "LOC",
                "WORK_OF_ART",
                "EVENT",
                "PEPWAVE_SETTINGS_ENTITY",  # Our custom entity type
            }
            and any(c.isalpha() for c in ent.text)
        }

        return list(entities)

    def ner(
        self,
        df: pd.DataFrame,
        primary_content_col: str = "primary_content",
        lead_content_col: str = "lead_content",
    ) -> pd.DataFrame:
        """
        Use SpaCy NER feature to extract an "entities" column from the text in the specified columns.
        The "entities" column contains a list of identified entities.

        Args:
            df: The dataframe to process
            primary_content_col: Column name for primary content
            lead_content_col: Column name for lead content (optional - can be None)

        Returns:
            DataFrame with added entities column
        """
        df = df.copy()
        # Process primary content column
        df["primary_entities"] = df[primary_content_col].apply(self.extract_entities)

        # Process lead content column if it exists
        if lead_content_col in df.columns:
            df["lead_entities"] = df[lead_content_col].apply(self.extract_entities)
        else:
            df["lead_entities"] = df.apply(lambda _: [], axis=1)

        # Combine entities from both columns, removing duplicates
        def combine_entities(row: pd.Series) -> list[str]:
            all_entities = []
            if row["primary_entities"]:
                all_entities.extend(row["primary_entities"])
            if row["lead_entities"]:
                all_entities.extend(row["lead_entities"])

            return list(set(all_entities))

        df["entities"] = df.apply(combine_entities, axis=1)

        # Drop intermediate columns
        df = df.drop(columns=["primary_entities", "lead_entities"])

        return df

    @staticmethod
    def _simple_dedupe(df: pd.DataFrame) -> None:
        df.drop_duplicates(subset=["page_content"], keep="first", inplace=True)

    def _stage_documents(self, docs: list[Document]) -> None:
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
        to_serialized_parquet(df, self.staging_folder / "staging.parquet")

    def _write_merged_df_artifact(self, df: pd.DataFrame) -> None:
        to_serialized_parquet(df, self.staging_folder / "merged.parquet")

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
    def _split_docs(docs: list[Document], splitter: TextSplitter) -> list[Document]:
        # Store original IDs in metadata before splitting
        for doc in docs:
            doc.metadata["parent_doc_id"] = doc.id

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
        """
        Generates the staged.parquet file that should be a local record of what was uploaded to the
        vector store. It should reflect the current vector store state for that dataset exactly.
        """
        documents_dir = self.config.root_dir / "data" / self.folder_name / "documents"

        if not documents_dir.exists():
            raise FileNotFoundError(
                f"Documents directory does not exist: {documents_dir}"
            )

        if self.staging_path.exists():
            raise FileNotFoundError(
                f"Staging data file already exists at {self.staging_path}"
            )

        dfs: list[pd.DataFrame] = []
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

    def normalize_columns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rename the following columns to be consistent across all datasets:
        section, post_title -> title
        post_content, description -> lead_content
        comment_content -> primary_content
        like_count, number_of_likes -> score
        creator_id, comment_author_id, channel_id -> author_id
        creator_name, comment_author_name, channel_title -> author_name

        Creates a copy to avoid mutating the original DataFrame.

        Args:
            df: The DataFrame to normalize

        Returns:
            New DataFrame with standardized columns added
        """
        # Create a copy at the beginning
        df = df.copy()

        # Map original column names to standardized names
        column_mappings = {
            "section": "title",  # html
            "post_title": "title",  # mongo/reddit
            "post_content": "lead_content",  # mongo/reddit
            "description": "lead_content",  # youtube
            "like_count": "score",  # youtube
            "number_of_likes": "score",  # mongo
            "creator_id": "author_id",  # mongo
            "comment_author_id": "author_id",  # reddit
            "channel_id": "author_id",  # youtube
            "creator_name": "author_name",  # mongo
            "comment_author_name": "author_name",  # reddit
            "channel_title": "author_name",  # youtube
        }

        df["primary_content"] = (
            df["comment_content"]
            if "comment_content" in df.columns
            else df["page_content"]
        )

        # Iterate through mappings and add standardized columns while preserving originals
        for original_col, standard_col in column_mappings.items():
            if original_col in df.columns and standard_col not in df.columns:
                df[standard_col] = df[original_col]
            elif original_col in df.columns and standard_col in df.columns:
                # If both columns exist, don't overwrite the standard column
                self.logger.info(
                    f"Both {original_col} and {standard_col} exist, keeping existing {standard_col}"
                )
        return df

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

    def _parquet_to_documents(
        self, file_path: Path, drop_embeddings: bool = False
    ) -> list[Document]:
        df = self._parquet_to_df(file_path, drop_embeddings)
        return df_to_documents(df)

    def _log_documents(self, docs: list[Document]) -> None:
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
        staging_path = cls.this_dir / cls.folder_name / filename
        if not staging_path.exists():
            raise FileNotFoundError(f"No staging file found at {staging_path}")
        return pd.read_parquet(staging_path)
