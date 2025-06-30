from typing import Any
from langchain_core.documents import Document
from pinecone import Pinecone
from openai import OpenAI


class PineconeRetriever:
    """Retriever that uses Pinecone for vector search with built-in reranking."""

    def __init__(
        self,
        index_name: str,
        embedding_model: str = "text-embedding-3-large",
        rerank_model: str = "bge-reranker-v2-m3",
        fields: list[str] = [
            "id",
            "page_content",
            "technical_summary",
            "subject_matter",
            "type",
            "post_category_name",
            "title",
            "lead_content",
            "primary_content",
            "score",
            "creator_is_star",  # pep forum only
            "themes",
            "entities",
            "created_at",  # standardized post/video created date
            # html only
            "settings_entities",
            "settings_entity_list",
            "all_settings_entities",
        ],
    ):
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.namespace = ""
        self.fields = fields

        self.openai = OpenAI()
        self.pinecone = Pinecone()
        self.pinecone_index = self.pinecone.Index(index_name)

    def get_query_embedding(self, query: str) -> list[float]:
        vector_response = self.openai.embeddings.create(
            input=query, model=self.embedding_model
        )
        return vector_response.data[0].embedding

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 100,
        rerank_top_n: int = 40,
        rank_field: str = "technical_summary",
    ) -> list[Document]:
        pc_query: Any = {"vector": {"values": query_embedding}, "top_k": top_k}
        rerank: Any = {
            "query": query,
            "top_n": rerank_top_n,
            "rank_fields": [rank_field],  # only one field currently suported by pc
            "model": self.rerank_model,
        }

        # Search Pinecone with reranking
        response = self.pinecone_index.search(
            namespace=self.namespace,
            query=pc_query,
            fields=self.fields,
            rerank=rerank,
        )

        documents = []
        for match in response.result.hits:
            metadata = match.fields.copy()
            page_content = metadata.pop("page_content")

            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)

        return documents


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    retriever = PineconeRetriever(
        index_name="pepwave-early-april-page-content-embedding",
    )
    retriever.retrieve(query="What is a Pepwave?")
