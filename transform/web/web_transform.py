import json
import pandas as pd
from pathlib import Path
from transform.base_transform import BaseTransform
from util.util_main import (
    get_column_word_count,
    set_string_columns,
)


class WebTransform(BaseTransform):
    """Transform web page data from JSONL files into a structured DataFrame."""

    folder_name = "web"

    def __init__(self):
        super().__init__()

    def transform_file(self, file_path: Path) -> pd.DataFrame:
        """Transform web page data from a JSONL file.

        Args:
            file_path: Path to the JSONL file containing web page data

        Returns:
            DataFrame containing transformed web page data
        """
        pages = []
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                meta = data["metadata"]
                page = self._add_required_columns(
                    columns={"url": meta["source"], "title": meta.get("title", "")},
                    page_content=data["page_content"],
                    file_path=file_path,
                )
                pages.append(page)
        df = self._make_df(pages)

        # filter out pages with empty title
        df = df[df["title"].str.strip().str.len() > 0]
        self._notify_dropped_rows(df, "empty title")

        set_string_columns(df, ["url", "title"], False)

        df["word_count"] = get_column_word_count(df, "page_content")

        # filter out pages with less than 100 words
        df = df[df["word_count"] >= 100]
        self._notify_dropped_rows(df, ">100 words")

        return df


if __name__ == "__main__":
    transformer = WebTransform()
    transformer.transform()
