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
from ragas.testset.transforms.base import BaseGraphTransformation
from ragas.utils import num_tokens_from_string

from load.reddit_general.reddit_general_load import RedditGeneralLoad
from load.reddit.reddit_load import RedditLoad
from load.html.html_load import HtmlLoad
from load.web.web_load import WebLoad
from load.youtube.youtube_load import YoutubeLoad
from load.mongo.mongo_load import MongoLoad

sample_size = 50
kg_llm = LangchainLLMWrapper(ChatOpenAI(model_name="gpt-4o-mini"))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
latest_kg_path = "evals/output/kg_output_LATEST.json"
latest_kg = KnowledgeGraph.load(latest_kg_path)


def estimate_ragas_eval_cost(
    documents: list[Document], num_test_cases: int = 200
) -> None:
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


def get_dataset_df(sample: bool = False) -> pd.DataFrame:
    dfs = [
        HtmlLoad.get_artifact(select_merged=True),
        MongoLoad.get_artifact(select_merged=True),
        RedditLoad.get_artifact(select_merged=True),
        RedditGeneralLoad.get_artifact(select_merged=True),
        YoutubeLoad.get_artifact(select_merged=True),
    ]

    if sample:
        dfs = [df.sample(sample_size) for df in dfs]

    return pd.concat(dfs, ignore_index=True)


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


def CREATE_KG(
    df: pd.DataFrame, transforms: list[BaseGraphTransformation], label: str
) -> None:
    docs = df_to_documents(df)
    num_docs = len(docs)
    print(f"Constructing KG with {num_docs} docs")
    kg = construct_kg_nodes(docs)
    apply_transforms(kg, transforms)
    print(f"Transformed KG has {len(kg.nodes)} nodes")
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    kg.save(f"evals/output/kg_output__n_{num_docs}__{label}__{timestamp}.json")
    kg.save(latest_kg_path)


dataset_df = get_dataset_df(sample=True)

# %% #####################################  ANALYZE DF #########################

print(f"Total number of rows in dataset: {len(dataset_df)}")
# Check for duplicate content and IDs
duplicate_content = dataset_df[
    dataset_df.duplicated(subset=["page_content"], keep=False)
]
duplicate_ids = dataset_df[dataset_df.duplicated(subset=["id"], keep=False)]

print(f"\nDuplicate content rows: {len(duplicate_content)}")
print(f"Duplicate ID rows: {len(duplicate_ids)}")

# %% #####################################  CREATE KG #########################

docs = df_to_documents(dataset_df)
# CREATE_KG(
#     dataset_df,
#     default_transforms(documents=docs, llm=kg_llm, embedding_model=embeddings),
#     "default_transforms",
# )

# %% #####################################  GET INDIVIDUAL DFS #########################

mongo_df = MongoLoad.get_artifact(select_merged=True)
youtube_df = YoutubeLoad.get_artifact(select_merged=True)
reddit_df = RedditLoad.get_artifact(select_merged=True)
reddit_general_df = RedditGeneralLoad.get_artifact(select_merged=True)
html_df = HtmlLoad.get_artifact(select_merged=True)

# Count rows with >500 words in page_content
long_mongo_docs = mongo_df[mongo_df["page_content"].str.split().str.len() > 500]
print(f"Number of documents with >500 words: {len(long_mongo_docs)}")

# %%
total_tokens = (
    youtube_df["page_content"].apply(lambda x: num_tokens_from_string(x)).sum()
)
print(f"Total tokens: {total_tokens}")
