import pandas as pd
from pathlib import Path
from transform.base_transform import BaseTransform
from transform.html.langchain_splitter_fork import HTMLSemanticPreservingSplitter
from util.util import set_string_columns

class HTMLTransform(BaseTransform):
    """Transform and chunk HTML documents."""

    folder_name = "html"

    def __init__(self):
        super().__init__()
        self.splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h3", "section")],
            max_chunk_size=3000,
            elements_to_preserve=["table", "ul", "ol"],
            tags_to_preserve=["table", "tr", "td", "th", "li"],
            preserve_image_metadata=True,
        )

    def transform_file(self, file_path: Path) -> pd.DataFrame:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        chunks = self.splitter.split_text(html_content)

        documents = []
        for chunk in chunks:
            doc = self._add_required_columns(
                columns={
                    "section": chunk.metadata.get("section", ""),
                    "images": chunk.metadata.get("images", []),
                },
                page_content=chunk.page_content,
                file_path=file_path,
            )
            documents.append(doc)
        df = self._make_df(documents)
        set_string_columns(df, ["section"], False)
        return df


if __name__ == "__main__":
    transformer = HTMLTransform()
    transformer.transform()
