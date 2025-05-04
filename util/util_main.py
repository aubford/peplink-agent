import re
from typing import Dict, Any, List
from langchain_core.documents import Document
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

import tiktoken
from itertools import accumulate


def serialize_document(document: Document) -> Dict[str, Any]:
    return {"page_content": document.page_content, "metadata": document.metadata}


def empty_document_dict(metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if metadata is None:
        metadata = {}
    return {"page_content": "", "metadata": metadata}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the identifier to be safe for use in filenames across all systems.
    Removes/replaces invalid filename characters using a standard approach.
    """
    # Common invalid filename characters including period
    invalid_chars = '<>:"/\\|?*@.-'

    # First handle leading special characters
    while filename and (filename[0] in invalid_chars or not filename[0].isalnum()):
        filename = filename[1:]

    # Then replace remaining invalid characters with underscore
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Replace spaces with underscore and remove any duplicate underscores
    filename = "_".join(filename.split())
    while "__" in filename:
        filename = filename.replace("__", "_")

    return filename.strip("_")


def print_replace(text: str) -> None:
    """
    Print text and replace the previous line.
    """
    print(f"{text}", end="\r", flush=True)


def get_column_word_count(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Calculate word count for a text column in a DataFrame.

    Args:
        df: DataFrame containing the text column
        column: Name of the column to count words from

    Returns:
        Series containing word counts as Int64 type
    """
    return df[column].str.split().str.len().astype("Int64")


def validate_string_column(
    df: pd.DataFrame, column: str, allow_empty: bool = True
) -> None:
    """
    Validate that all values in a column are non-empty strings.
    Raises ValueError for any invalid values.
    """
    log_path = Path("logs/validate_string_column_error_rows.parquet")
    log_path.parent.mkdir(exist_ok=True)

    if column not in df.columns:
        print(f"Available columns: {', '.join(df.columns)}")
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if df[column].apply(lambda x: not isinstance(x, str)).any():
        non_string_rows = df[df[column].apply(lambda x: not isinstance(x, str))]
        print(non_string_rows[column])
        raise ValueError(f"Column '{column}' contains non-string values")
    if df[column].isna().any():
        nan_rows = df[df[column].isna()]
        nan_rows.to_parquet(log_path)
        raise ValueError(f"Column '{column}' contains NaN values")
    if not allow_empty:
        if (df[column] == "").any():
            empty_rows = df[df[column] == ""]
            empty_rows.to_parquet(log_path)
            print(f"Found {len(empty_rows)} rows with empty strings in '{column}'")
            raise ValueError(f"Column '{column}' contains empty strings")

        spaces = df[column].apply(lambda x: x.isspace())
        if spaces.any():
            whitespace_rows = df[spaces]
            whitespace_rows.to_parquet(log_path)
            raise ValueError(f"Column '{column}' contains whitespace-only strings")


def validate_string_columns(
    df: pd.DataFrame, columns: List[str], allow_empty: bool = True
) -> None:
    """
    Validate that all values in a list of columns are non-empty strings.
    Raises ValueError for any invalid values.
    """
    for column in columns:
        validate_string_column(df, column, allow_empty)


def set_string_columns(
    df: pd.DataFrame, columns: List[str], allow_empty: bool = True
) -> None:
    """
    Set the string columns to the strict string type.
    """
    validate_string_columns(df, columns, allow_empty)
    for column in columns:
        df[column] = df[column].astype("string[pyarrow]", errors="raise")


def dedupe_df_ids(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["id"]).set_index(
        "id", drop=False, verify_integrity=True
    )


def serialize_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe for parquet serialization by converting complex data types to JSON strings.

        Args:
            df: The dataframe to prepare

        Returns:
        A copy of the dataframe with complex types serialized
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # Convert lists, sets, and other JSON serializable types to JSON strings
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, set, dict))).any():
            df[col] = df[col].apply(
                lambda x: (
                    json.dumps(list(x))
                    if isinstance(x, set)
                    else json.dumps(x) if isinstance(x, (list, dict)) else x
                )
            )

    return df


def drop_embedding_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all columns whose name contain the string 'embed' in the name."""
    return df.loc[:, ~df.columns.str.contains("embed")]


def to_serialized_parquet(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = serialize_df_for_parquet(df)
    df.to_parquet(path)
    print(f"Saved {len(df)} documents to {path}")
    return df


def count_tokens(text: str) -> int:
    """Count the tokens in a text."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(str(text)))


def collapse_blank_lines(text: str) -> str:
    """
    Replace runs of 3 or more consecutive blank lines (optionally with whitespace) with exactly two newlines.
    """
    return re.sub(r'((?:[ \t]*\n){3,})', '\n\n', text)


def clean_text_for_embedding(text: str) -> str:
    """
    Remove Markdown headers, XML/HTML tags, and image insertions from the text.
    Args:
        text: The input text to clean.
    Returns:
        Cleaned text with non-semantic content removed.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove Markdown image insertions: ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)

    # Remove Markdown headers (lines starting with #), except those starting with '## Title: '
    text = re.sub(r"^(?:(?!## Title: ).)#.*$", "", text, flags=re.MULTILINE)

    # Remove XML/HTML tags (e.g., <tag> or </tag> or <tag attr="val">)
    text = re.sub(r"<[^>]+>", "", text)

    # Remove excessive spaces/tabs (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse runs of 3+ blank lines (optionally with whitespace) into exactly 2 newlines
    text = collapse_blank_lines(text)

    text = text.strip()
    return text


def get_chunk_size(texts: list[str], token_limit: int = 300_000) -> int:
    """
    Determine the optimal chunk size so that the sum of the largest chunk_size token counts is under the token_limit.
    Returns the largest chunk size that keeps max_possible_tokens < token_limit.
    """
    token_counts = [count_tokens(text) for text in texts]
    total_tokens = sum(token_counts)
    if total_tokens < token_limit:
        return len(texts)
    sorted_counts = sorted(token_counts, reverse=True)
    for i, running_sum in enumerate(accumulate(sorted_counts), 1):
        if running_sum >= token_limit:
            return i - 1
    return len(texts)


def handle_file_exists(path: Path, should_raise: bool = False) -> Path | None:
    """
    If the given file exists, rename it by appending the current time (HH_MM) to its stem.
    Returns the new backup path if the file was renamed, otherwise None.
    """
    if path.exists():
        if should_raise:
            raise FileExistsError(f"File already exists: {path}")
        backup_path = path.with_stem(f"{path.stem}__{datetime.now().strftime('%H_%M')}")
        path.rename(backup_path)
        return backup_path
    return None
