import pandas as pd
import typing as t
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from util.document_utils import df_to_documents
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms
from datetime import datetime
from ragas.testset.transforms.base import BaseGraphTransformation
from ragas.utils import num_tokens_from_string
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)
from ragas.utils import num_tokens_from_string
from ragas.testset.transforms import Parallel
from evals.evals_utils import node_meta
from langsmith import tracing_context


kg_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-large")
)
latest_kg_path = "evals/output/kg_output_LATEST.json"


def construct_kg_nodes(docs: list[Document]) -> KnowledgeGraph:
    """Apply the following columns from df to nodes.properties:
    - entities: List[str]
    - themes: List[str]
    - title_embedding: List[float]
    - technical_summary: str
    - technical_summary_embedding: List[float]
    """
    knowledge_graph = KnowledgeGraph()
    for doc in docs:
        knowledge_graph.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                    "entities": doc.metadata["entities"],
                    "themes": doc.metadata["themes"],
                    "title_embedding": doc.metadata["title_embedding"],
                    "technical_summary": doc.metadata["technical_summary"],
                    "technical_summary_embedding": doc.metadata[
                        "technical_summary_embedding"
                    ],
                },
            )
        )
    print(f"Constructed KG has {len(knowledge_graph.nodes)} nodes")
    return knowledge_graph


def clean_meta(doc: Document) -> Document:
    # Filter out metadata fields with null-like values
    doc.metadata = {
        k: v for k, v in doc.metadata.items() if v is not None and not pd.isna(v)
    }
    return doc


def create_kg(
    df: pd.DataFrame, transforms: list[BaseGraphTransformation], label: str
) -> None:
    docs = df_to_documents(df)
    docs = [clean_meta(doc) for doc in docs]
    num_docs = len(docs)
    print(f"Constructing KG with {num_docs} docs")
    kg = construct_kg_nodes(docs)
    print(f"KG has {len(kg.nodes)} nodes")
    apply_transforms(kg, transforms)
    print(f"Transformed KG has {len(kg.nodes)} nodes")
    print(f"Transformed KG has {len(kg.relationships)} relationships")
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    kg.save(f"evals/output/kg_output__n_{num_docs}__{label}__{timestamp}.json")


################################### MAIN ###################################

"""
1. Create relationships for all properties:
- entities
- themes
- title_embedding
- technical_summary
- technical_summary_embedding
2. Normalize the scores for each between 0 and 1
3. Create a new KG that merges relationships with same source and target into a single relationship with a score based 
on a heuristic combining them.
4. Filter out low scoring relationships (take the top half)?
"""

with tracing_context(enabled=False):

    def filter_youtube(node):
        return node_meta(node)["type"] == "youtube"

    def filter_min_tokens(node, property_name: str, min_tokens: int = 100):
        return num_tokens_from_string(node.properties[property_name]) > min_tokens

    def get_transforms() -> t.List[BaseGraphTransformation]:
        cosine_sim_builder = CosineSimilarityBuilder(
            property_name="technical_summary_embedding",
            new_property_name="summary_similarity",
            threshold=0.9,
            filter_nodes=lambda node: filter_min_tokens(node, "technical_summary", 100),
        )
        ner_overlap_sim = OverlapScoreBuilder(
            threshold=0.5,
            filter_nodes=lambda node: len(node.properties["entities"]) > 3,
        )
        transforms = Parallel(cosine_sim_builder, ner_overlap_sim)
        return transforms  # type: ignore

    sample_df = pd.read_parquet("evals/sample_df.parquet")
    sample_df = sample_df.iloc[0:5]
    create_kg(sample_df, get_transforms(), "sample_with_custom_transforms")
