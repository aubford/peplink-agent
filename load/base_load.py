from abc import abstractmethod
import logging
from typing import Any, Dict
from config import global_config, ConfigType
import pandas as pd
from pathlib import Path
from langchain.docstore.document import Document


logger = logging.getLogger('loader')


class BaseLoader:
    """Base class for all data transformers."""

    def __init__(self, folder_name: str):
        self.folder_name = folder_name

    @property
    def config(self) -> ConfigType:
        """Get the global config singleton."""
        return global_config

    @abstractmethod
    def load_file(self, data: Dict[str, Any]) -> pd.DataFrame:
        pass

    def load(self) -> None:
        documents_dir = Path("data") / self.folder_name / "documents"

        if not documents_dir.exists():
            raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")

        for file_path in documents_dir.glob("*"):
            try:
                self.load_file(file_path)

            except Exception as e:
                logger.error(f"Error processing {file_path}")
                raise e

    def df_to_documents(self, df: pd.DataFrame) -> List[Document]:
        documents = []
        for _, row in df.iterrows():
            metadata = row.drop(["page_content", "id"]).to_dict()
            doc = Document(
                id=row["id"],
                page_content=row["page_content"],
                metadata=metadata
            )
            documents.append(doc)
        return documents
