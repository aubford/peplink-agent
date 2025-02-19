# %%
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
import pandas as pd
from util.document_utils import df_to_documents
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from datetime import datetime
from ragas.testset.transforms.extractors import (
    EmbeddingExtractor,
    HeadlinesExtractor,
    SummaryExtractor,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.utils import num_tokens_from_string

from load.reddit_general.reddit_general_load import RedditGeneralLoad
from load.reddit.reddit_load import RedditLoad
from load.html.html_load import HtmlLoad
from load.web.web_load import WebLoad
from load.youtube.youtube_load import YoutubeLoad
from load.mongo.mongo_load import MongoLoad


LLM = "gpt-4o-mini"
sample_size = 1


def get_dataset_df(sample: bool = False) -> pd.DataFrame:
    dfs = [
        HtmlLoad.get_artifact(select_merged=True),
        MongoLoad.get_artifact(select_merged=True),
        RedditLoad.get_artifact(select_merged=True),
        RedditGeneralLoad.get_artifact(select_merged=True),
        WebLoad.get_artifact(select_merged=True),
        YoutubeLoad.get_artifact(select_merged=True),
    ]

    if sample:
        for df in dfs:
            df = df.sample(sample_size)

    return pd.concat(dfs, ignore_index=True)


# apply transforms in place and save backup file
def transform(kg: KnowledgeGraph, documents: list[Document]) -> None:
    kg_llm = LangchainLLMWrapper(ChatOpenAI(model_name=LLM))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    # Apply transformations to create relationships
    transforms = default_transforms(
        documents=documents, llm=kg_llm, embedding_model=embeddings
    )

    apply_transforms(kg, transforms)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    kg.save(f"evals/backup/knowledge_graph_{timestamp}.json")


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

    print(knowledge_graph.nodes)
    return knowledge_graph


def create_knowledge_graph() -> None:
    df = get_dataset_df(sample=True)
    docs = df_to_documents(df)
    knowledge_graph = construct_kg_nodes(docs)
    transform(knowledge_graph, docs)
    # Save knowledge graph to staged file
    knowledge_graph.save("evals/staged_knowledge_graph.json")


def estimate_ragas_eval_cost(
    documents: list[Document], num_test_cases: int = 200
) -> None:
    """
    Estimates the cost of running Ragas evaluations based on:
    1. Knowledge graph creation cost
    2. Test case generation cost
    3. Evaluation metrics cost
    """
    # Calculate total document tokens
    total_text = " ".join([doc.page_content for doc in documents])
    doc_tokens = num_tokens_from_string(total_text)

    # Estimate knowledge graph creation cost (processes each doc ~3 times)
    kg_tokens = doc_tokens * 3

    # Estimate test case generation (each test case uses ~1000 tokens)
    testset_tokens = num_test_cases * 1000

    # Estimate evaluation cost (6 metrics, each using ~500 tokens per test case)
    eval_tokens = num_test_cases * 6 * 500

    total_tokens = kg_tokens + testset_tokens + eval_tokens
    estimated_cost = (
        total_tokens / 1_000_000
    ) * 0.15  # gpt-4o-mini rate $0.15/1M tokens

    kg_cost = kg_tokens / 1_000_000 * 0.15
    testset_cost = testset_tokens / 1_000_000 * 0.15
    eval_cost = eval_tokens / 1_000_000 * 0.15

    print(f"Total docs: {len(documents):,}")
    print(f"Dataset tokens: {doc_tokens:,}")
    print(f"Estimated tokens for knowledge graph: {kg_tokens:,}")
    print(f"estimated cost for knowledge graph: ${kg_cost:.2f}")
    print(f"Estimated tokens for test generation: {testset_tokens:,}")
    print(f"estimated cost for test generation: ${testset_cost:.2f}")
    print(f"Estimated tokens for evaluation: {eval_tokens:,}")
    print(f"estimated cost for evaluation: ${eval_cost:.2f}")
    print(f"Total estimated tokens: {total_tokens:,}")
    print(f"Estimated cost: ${estimated_cost:.2f}")


df = get_dataset_df()
dataset_docs = df_to_documents(df)
estimate_ragas_eval_cost(dataset_docs, 500)
