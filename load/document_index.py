import re
from langchain_openai import OpenAIEmbeddings
from load.html.html_load import HtmlLoad
from load.mongo.mongo_load import MongoLoad
from load.reddit.reddit_load import RedditLoad
from load.reddit_general.reddit_general_load import RedditGeneralLoad
from load.youtube.youtube_load import YoutubeLoad
from util.nlp import normalize_entities_and_themes
from util.util_main import (
    count_tokens,
    clean_text_for_embedding,
    collapse_blank_lines,
    get_chunk_size,
    to_serialized_parquet,
)
import pandas as pd
import tiktoken
from pathlib import Path
import json


def clean_text_for_inference(text: str) -> str:
    """
    Remove Markdown headers, XML/HTML tags, and image insertions from the text.
    Args:
        text: The input text to clean.
    Returns:
        Cleaned text content that is not useful for LLM.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove Markdown image insertions: ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)

    # Remove entire lines containing [IMG]... image inserts
    text = re.sub(r"^.*\[IMG\]\S*.*$\n?", "", text, flags=re.MULTILINE)

    # Collapse runs of 3+ blank lines (optionally with whitespace) into exactly 2 newlines
    text = collapse_blank_lines(text)

    text = text.strip()
    return text


class DocumentIndex:
    """
    Handle any processing that needs to be done after the original load the current
    Knowledge Graph is based on. Create the final single document index to be used
    going forward.
    """

    this_dir = Path(__file__).parent
    document_index_path = this_dir / "document_index.parquet"

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedding_model: OpenAIEmbeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        self.loaders = [
            RedditLoad(),
            RedditGeneralLoad(),
            MongoLoad(),
            YoutubeLoad(),
            HtmlLoad(),
        ]
        self.artifacts = [loader.get_artifact() for loader in self.loaders]
        self.staging_data = pd.concat(self.artifacts, verify_integrity=True)

    def _generate_embeddings_for_column(
        self, df: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        # remove markdown headers, xml/html tags, and other content that should be in
        # page_content for LLM inference but carries no semantic meaning when embedded
        clean_column_name = f"{column_name}_embedding_clean" # should be "page_content_clean_for_embedding" instead
        df[clean_column_name] = df[column_name].apply(clean_text_for_embedding)
        texts = df[clean_column_name].tolist()

        # Determine optimal chunk size
        optimal_chunk_size = get_chunk_size(texts)
        print(f"Using chunk size {optimal_chunk_size}")

        embeddings = self.embedding_model.embed_documents(
            texts, chunk_size=optimal_chunk_size
        )
        df[f"{column_name}_embedding"] = embeddings
        return df

    def _clean_page_content(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Drop the 10 or so rows where page_content is > 4000 tokens because they have
        # wacky content that isn't worth the effort
        before_count = len(df)
        df = df[df["page_content"].apply(lambda x: count_tokens(x) <= 4000)]
        dropped_count = before_count - len(df)
        print(f"Dropped {dropped_count} rows where page_content > 4000 tokens.")

        # Remove text that is of no use for LLM inference
        df["page_content_dirty"] = df["page_content"]
        df["page_content"] = df["page_content"].apply(clean_text_for_inference)

        return df

    def create(self) -> None:
        df = self._clean_page_content(self.staging_data)
        df["token_count"] = df["page_content"].apply(count_tokens)
        df = self._generate_embeddings_for_column(df, "page_content")
        df = normalize_entities_and_themes(df)

        # Save with columns in alphabetical order
        df = df[sorted(df.columns)]
        df = df.drop(columns=["id"])
        to_serialized_parquet(df, self.document_index_path)

    @classmethod
    def get_document_index(cls) -> pd.DataFrame:
        df = pd.read_parquet(cls.document_index_path)
        # fill missing technical summaries embeddings in HTML elements with page_content embedding
        # this is also reflected in the vector store
        for col in [
            "page_content_embedding",
            "technical_summary_embedding",
            "primary_content_embedding",
            "title_embedding",
            "entities",
            "entities_pre_normalization",
            "themes",
        ]:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        return df
