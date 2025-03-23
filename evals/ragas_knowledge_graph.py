import pandas as pd
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from util.document_utils import df_to_documents
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from datetime import datetime
from ragas.testset.transforms.base import BaseGraphTransformation
from ragas.utils import num_tokens_from_string

from load.reddit_general.reddit_general_load import RedditGeneralLoad
from load.reddit.reddit_load import RedditLoad
from load.html.html_load import HtmlLoad
from load.youtube.youtube_load import YoutubeLoad
from load.mongo.mongo_load import MongoLoad

from ragas.testset.graph import NodeType
from ragas.testset.transforms.extractors import (
    EmbeddingExtractor,
    HeadlinesExtractor,
    SummaryExtractor,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)
from ragas.testset.transforms.splitters import HeadlineSplitter
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
    knowledge_graph = KnowledgeGraph()
    for doc in docs:
        knowledge_graph.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
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

# mongo -> thin out rows randomly?
# should we skip the general youtubes and reddits? probably not....
# how to handle html?

# create the pipeline then run it on the dataset while doing thorough debugging breakpoints.
# Check for duplicates after the headline stage (and the others if this isn't the cause)
# todo: Manually add headlines/themes using metadata?


def filter_youtube(node):
    return node_meta(node)["type"] == "youtube"


def filter_min_tokens(node):
    return num_tokens_from_string(node.properties["page_content"]) > 100


def filter_chunks(node):
    return node.type == NodeType.CHUNK


def get_transforms():
    headline_extractor = HeadlinesExtractor(
        llm=kg_llm, filter_nodes=lambda node: filter_youtube(node)
    )
    splitter = HeadlineSplitter(
        min_tokens=500, filter_nodes=lambda node: filter_youtube(node)
    )

    summary_extractor = SummaryExtractor(
        llm=kg_llm, filter_nodes=lambda node: filter_min_tokens(node)
    )

    node_filter = CustomNodeFilter(
        llm=kg_llm, filter_nodes=lambda node: filter_chunks(node)
    )

    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embeddings,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=lambda node: filter_min_tokens(node),
    )
    theme_extractor = ThemesExtractor(
        llm=kg_llm, filter_nodes=lambda node: filter_min_tokens(node)
    )
    ner_extractor = NERExtractor(
        llm=kg_llm, filter_nodes=lambda node: filter_min_tokens(node)
    )

    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.7,
        filter_nodes=lambda node: filter_min_tokens(node),
    )
    ner_overlap_sim = OverlapScoreBuilder(
        threshold=0.01, filter_nodes=lambda node: filter_min_tokens(node)
    )
    return [
        headline_extractor,
        splitter,
        summary_extractor,
        node_filter,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]


with tracing_context(enabled=False):
    sample_df = pd.read_parquet("evals/sample_df_55.parquet")
    create_kg(sample_df, get_transforms(), "sample_with_custom_transforms")
