from langchain.docstore.document import Document
from pydantic import BaseModel
from load.base_load import BaseLoad
from load.forum_synthetic_data_mixin import ForumSyntheticDataMixin, ModelResponse
import pandas as pd

class RedditLoad(BaseLoad, ForumSyntheticDataMixin):
    folder_name = "reddit"

    def __init__(self):
        super().__init__()

    def create_merged_df(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets into a single dataframe and perform operations."""
        df = pd.concat(dfs)
        df = self.ner(df)
        return df

    def load_docs(self, documents: list[Document]) -> list[Document]:
        """
        Process documents using the BatchManager and ForumSyntheticDataMixin.

        Args:
            documents: List of documents to process

        Returns:
            Processed documents with additional metadata
        """
        # Create batch items from documents
        batch_items = []
        for doc in documents:
            if not doc.id:
                raise ValueError("Document ID is required for batch processing")
            lead_content = doc.metadata.get("lead_content", "")
            primary_content = doc.metadata.get("primary_content", "")

            batch_items.append(
                {
                    "id": doc.id,
                    "prompt": self.create_prompt(lead_content, primary_content),
                }
            )

        # If we have batch items, create and run a batch job
        if batch_items:
            system_prompt = self.create_system_prompt_with_examples()
            self.batch_manager.create_batch_tasks(
                items=batch_items,
                schema=ModelResponse,
                system_prompt=system_prompt,
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=2040,
            )
            self.batch_manager.test_batchfile()

        return documents
