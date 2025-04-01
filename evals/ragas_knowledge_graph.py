from pathlib import Path
import numpy as np
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


this_file_path = Path(__file__).parent


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
        # Convert entities from comma-separated string to list if it's a string
        entities = doc.metadata["entities"]
        if isinstance(entities, str):
            entities = [entity.strip() for entity in entities.split(",")]

        knowledge_graph.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                    "entities": entities,
                    "themes": doc.metadata.get("themes", []),
                    "title_embedding": doc.metadata.get("title_embedding", []),
                    "technical_summary": doc.metadata.get("technical_summary", ""),
                    "technical_summary_embedding": doc.metadata.get(
                        "technical_summary_embedding", []
                    ),
                },
            )
        )
    print(f"Constructed KG has {len(knowledge_graph.nodes)} nodes")
    return knowledge_graph


def is_valid_metadata(v):
    if v is None:
        return False
    try:
        return not pd.isna(v)
    except:
        return True


def clean_meta(doc: Document) -> Document:
    # Filter out metadata fields with null-like values
    doc.metadata = {k: v for k, v in doc.metadata.items() if is_valid_metadata(v)}
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
    kg.save(
        f"{this_file_path}/output/kg_output__n_{num_docs}__{label}__{timestamp}.json"
    )


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

    def filter_out_html(node):
        return node_meta(node)["type"] != "html"

    def filter_min_tokens(node, property_name: str, min_tokens: int = 100):
        return num_tokens_from_string(node.properties[property_name]) > min_tokens

    def get_transforms() -> t.List[BaseGraphTransformation]:
        cosine_sim_builder = CosineSimilarityBuilder(
            property_name="title_embedding",
            new_property_name="title_similarity",
            threshold=0.8,
            filter_nodes=filter_out_html,
        )
        summary_cosine_sim_builder = CosineSimilarityBuilder(
            property_name="technical_summary_embedding",
            new_property_name="summary_similarity",
            threshold=0.8,
            filter_nodes=filter_out_html,
        )
        themes_overlap_sim = OverlapScoreBuilder(
            property_name="themes",
            new_property_name="themes_overlap_score",
            threshold=0.2,
            filter_nodes=lambda node: len(node.properties["themes"]) > 3
            and filter_out_html(node),
        )
        entities_overlap_sim = OverlapScoreBuilder(
            property_name="entities",
            new_property_name="entities_overlap_score",
            threshold=0.2,
            filter_nodes=lambda node: len(node.properties["entities"]) > 3,
        )
        transforms = Parallel(
            cosine_sim_builder,
            summary_cosine_sim_builder,
            entities_overlap_sim,
            themes_overlap_sim,
        )
        return transforms  # type: ignore

    sample_df = pd.read_parquet(this_file_path / "sample_df.parquet")
    create_kg(sample_df, get_transforms(), "subsample_with_four_transforms")
