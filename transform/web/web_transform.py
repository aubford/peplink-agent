import json
import pandas as pd
from pathlib import Path
from transform.base_transform import BaseTransform
import uuid

class WebTransform(BaseTransform):
    """Transform web page data from JSONL files into a structured DataFrame."""

    def __init__(self):
        super().__init__("web")

    def transform_file(self, file_path: Path) -> pd.DataFrame:
        """Transform web page data from a JSONL file.

        Args:
            file_path: Path to the JSONL file containing web page data

        Returns:
            DataFrame containing transformed web page data
        """
        pages = []

        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                metadata = data['metadata']

                page = {
                    # Document properties
                    'id': str(uuid.uuid4()),
                    'page_content': data['page_content'],

                    # Source tracking
                    'source_file': file_path.name,

                    # Metadata
                    'url': metadata['source'],
                    'title': metadata.get('title', ''),
                    'word_count': len(data['page_content'].split())
                }
                pages.append(page)

        df = pd.DataFrame(pages)
        df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce').fillna(0).astype('Int64')

        # Filter out pages with less than 100 words and require "pep" in content
        df = df[(df['word_count'] >= 100)].reset_index(drop=True)
        return df


if __name__ == "__main__":
    transformer = WebTransform()
    transformer.transform()
