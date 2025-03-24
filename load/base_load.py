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
import spacy
from load.html.html_util import get_settings_entities

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
    - When comparing a non-HTML with HTML: Compare with both HTML.entities and HTML.settings_entity_list.
    - Don't compare HTML with itself?
    - Always try to find an HTML relationship via this column.
From LLM:
- themes: Themes from primary_content and lead_content. (EX: HTML)
- is_useful: Whether the primary_content contains useful technical information.
- summary: (EX: HTML)
    - For YouTube: Summary of the primary_content.
    - For forums: Summary of the primary_content and lead_content. Use special prompt to explain the two documents and how to summarize them.
From Embedding Model:
- summary_embedding: Embedding of the summary. (use embedding extractor)
- title_embedding: Embedding of the title column. (filter out short titles) (use embedding extractor)
- lead_content_embedding: Embedding of the lead_content. (use embedding extractor) (EX: HTML)
"""


class ForumSyntheticDataPrompt:
    """
    Class to generate prompts for OpenAI API to extract structured data from forum posts.
    The prompt facilitates transformation of a document's primary_content and lead_content
    into a structured format for analysis.
    """

    @staticmethod
    def create_system_prompt() -> str:
        """Creates the system prompt that instructs the model on its task."""
        return """You are an expert technical content analyzer specializing in IT networking and Pepwave products.
Your task is to analyze forum conversations and extract key information.

You will be provided with a forum conversation consisting of:
1. The original forum post/question.
2. A response to the original post.

Together, these form a single conversation turn between two forum users.

Analyze this conversation and provide a structured output containing:
1. A list of technical themes discussed.
2. A summary of the technical information that can be learned from the conversation. Provide this in the form of sentences, not lists or other special formatting.
3. An assessment of whether there is useful technical information related to Pepwave products or IT networking.

Important guidelines:
- Focus only on technical content and information in your analysis.
- "Useful technical information" means factual statements, not questions asking for information.
- Be specific and precise in identifying themes.
- Provide a concise, factual summary that captures the key technical points.
- Base your analysis only on the provided content, do not make assumptions.
"""

    @staticmethod
    def create_prompt(lead_content: str, primary_content: str) -> str:
        """Creates a prompt that includes the forum post content."""

        return f"""# Forum Post (Original Post/Question)
{lead_content}

# Response to the Original Post
{primary_content}

Analyze this conversation according to the guidelines provided.
"""

    @staticmethod
    def get_examples() -> list[dict]:
        """
        Provides examples of conversations and ideal responses.

        Returns:
            List of example dictionaries.
        """
        return [
            {
                "lead_content": "I just installed a Pepwave MAX Transit Duo-CAT12 in my RV, but I'm having trouble with the cellular connection. The signal strength is showing only 2 bars even though my phone gets 4 bars in the same location. Does anyone know why this might be happening?",
                "primary_content": "Check your antenna connections first. The MAX Transit Duo requires proper external antennas to get the best signal. Make sure you're using the right cellular antennas and they're properly connected to the correct ports (they're labeled CELL on the router). Also, try changing the SIM priority in the admin panel - go to Network > Mobile > Settings, and you can change which SIM card is used or enable band locking for better performance on specific carriers. If you're in a fringe area, enabling band locking to the lower frequencies (like Band 12, 13, or 71 depending on your carrier) might help with penetration and range.",
                "expected_output": {
                    "themes": [
                        "Cellular signal strength",
                        "Antenna configuration",
                        "Pepwave MAX Transit Duo",
                        "SIM card settings",
                        "Band locking",
                        "RV networking",
                    ],
                    "summary": "The conversation discusses troubleshooting poor cellular signal on a Pepwave MAX Transit Duo-CAT12 installed in an RV. The response suggests checking antenna connections to the correct ports, adjusting SIM priority in the admin panel, and enabling band locking for specific carriers, particularly lower frequency bands for better range in fringe areas.",
                    "is_useful": True,
                },
            },
            {
                "lead_content": "Anyone have recommendations for a good backup internet solution? I work from home and need something reliable when my main fiber connection goes down.",
                "primary_content": "I've been there! After trying several options, I settled on a Peplink Balance 20X with a 5G capable modem. The SpeedFusion technology in the Peplink devices is amazing for combining connections. I use it with both my fixed connection and a cellular backup, and the handover between them is completely seamless. I can be on a Zoom call and if my main connection fails, the call doesn't drop at all because of the Hot Failover feature. It's not cheap but worth every penny for reliability.",
                "expected_output": {
                    "themes": [
                        "Backup internet solutions",
                        "Peplink Balance 20X",
                        "SpeedFusion technology",
                        "Connection bonding",
                        "Hot Failover",
                        "Work from home setup",
                    ],
                    "summary": "The conversation covers backup internet solutions for remote work. The response recommends the Peplink Balance 20X with a 5G modem, highlighting its SpeedFusion technology for combining connections and Hot Failover feature that provides seamless transition between primary and backup connections, maintaining continuity for applications like video calls.",
                    "is_useful": True,
                },
            },
            {
                "lead_content": "Does anyone know if Peplink routers work with AT&T FirstNet? I'm looking to set up a mobile command center for our emergency response team.",
                "primary_content": "No idea, I've never used FirstNet. Have you tried contacting Peplink support directly? They might have better information about carrier compatibility.",
                "expected_output": {
                    "themes": [
                        "FirstNet compatibility",
                        "Peplink routers",
                        "Emergency response equipment",
                    ],
                    "summary": "The conversation inquires about compatibility between Peplink routers and AT&T FirstNet for an emergency response mobile command center. The response does not provide any specific technical information about compatibility, only suggesting to contact Peplink support directly.",
                    "is_useful": False,
                },
            },
        ]

    @staticmethod
    def create_system_prompt_with_examples(num_examples: int = 2) -> str:
        """
        Creates a system prompt that includes both instructions and examples.

        Args:
            num_examples: Number of examples to include (default: 2)

        Returns:
            Complete system prompt with instructions and examples
        """
        base_prompt = ForumSyntheticDataPrompt.create_system_prompt()
        examples = ForumSyntheticDataPrompt.get_examples()

        # Limit to the requested number of examples
        examples = examples[: min(num_examples, len(examples))]

        examples_text = (
            "\n\nHere are some examples of conversations and expected analyses:\n\n"
        )

        for i, example in enumerate(examples, 1):
            # Format the example conversation
            example_prompt = ForumSyntheticDataPrompt.create_prompt(
                lead_content=example["lead_content"],
                primary_content=example["primary_content"],
            )

            # Format the expected output
            output = example["expected_output"]
            themes_str = ", ".join([f'"{theme}"' for theme in output["themes"]])

            expected_output = (
                f'Expected output:\n'
                f'{{\n'
                f'  "themes": [{themes_str}],\n'
                f'  "summary": "{output["summary"]}",\n'
                f'  "is_useful": {output["is_useful"]}\n'
                f'}}\n'
            )

            examples_text += (
                f"EXAMPLE {i}:\n\n{example_prompt}\n\n{expected_output}\n{'=' * 40}\n\n"
            )

        return base_prompt + examples_text


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
        self.index_name = self.config.get("VERSIONED_PINECONE_INDEX_NAME")
        self.logger = RotatingFileLogger(name=f"load_{self.folder_name}")
        self.vector_store: PineconeVectorStore | None = None
        self.staging_folder = self.this_dir / self.folder_name
        self.staging_path = self.staging_folder / "staging.parquet"
        self.generated_data_path = self.staging_folder / "generated_data.parquet"
        deduplication_lock_path = self.staging_folder / "deduplication_lock.parquet"
        self.deduplication_pipeline = (
            DeduplicationLockPipeline(deduplication_lock_path)
            if deduplication_lock_path.exists()
            else DeduplicationPipeline(self.folder_name)
        )

    # todo
    def apply_generated_data(self):
        """Apply the generated data from LLM to the staging file."""
        pass

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    def create_merged_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets into a single dataframe and perform operations like deduplication."""
        return pd.concat(dfs)

    def load_docs(self, documents: List[Document]) -> List[Document]:
        """Perform final operations like text splitting and output final product for upload to vector store."""
        return documents

    def _init_spacy(self):
        """Initialize SpaCy model."""
        nlp = spacy.load("en_core_web_trf")
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        all_entities = get_settings_entities()
        patterns = [
            {"label": "PEPWAVE_SETTINGS_ENTITY", "pattern": entity, "id": "LOWER"}
            for entity in all_entities
        ]
        ruler.add_patterns(patterns)
        return nlp

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
                dimension=3072,  # OpenAI embeddings dimension for text-embedding-3-large
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            self.logger.info(f"Created new Pinecone index: {self.index_name}")

        # Initialize the index
        index = pc.Index(self.index_name)
        vector_store = PineconeVectorStore(
            index=index, embedding=OpenAIEmbeddings(model="text-embedding-3-large")
        )
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
        if self.vector_store is None:
            raise ValueError("Vector store initialization failed")
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

        # todo: reset!!!!!
        # if self.staging_path.exists():
        #     raise FileNotFoundError(
        #         f"Staging data file already exists at {self.staging_path}"
        #     )

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
        self._normalize_columns(staging_df)
        self._write_merged_df_artifact(staging_df)
        all_documents = df_to_documents(staging_df)
        staging_docs = self.load_docs(all_documents)
        self._stage_documents(staging_docs)

    def _normalize_columns(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Rename the following columns to be consistent across all datasets:
        section, post_title -> title
        post_content, description -> lead_content
        comment_content -> primary_content
        like_count, number_of_likes -> score
        creator_id, comment_author_id, channel_id -> author_id
        creator_name, comment_author_name, channel_title -> author_name

        Keep the original columns as well.
        """
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
        staging_path = cls.this_dir / cls.folder_name / filename
        if not staging_path.exists():
            raise FileNotFoundError(f"No staging file found at {staging_path}")
        return pd.read_parquet(staging_path)
