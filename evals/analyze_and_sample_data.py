# %%
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

kg_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
latest_kg_path = "evals/output/kg_output_LATEST.json"


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


def get_dataset_df(sample_individual: bool = False) -> pd.DataFrame:
    dfs = [
        HtmlLoad.get_artifact(select_merged=True),
        MongoLoad.get_artifact(select_merged=True).sample(frac=0.2),
        RedditLoad.get_artifact(select_merged=True),
        RedditGeneralLoad.get_artifact(select_merged=True),
        YoutubeLoad.get_artifact(select_merged=True),
    ]

    if sample_individual:
        dfs = [df.sample(frac=0.2) for df in dfs]

    return pd.concat(dfs, ignore_index=True)


def get_slim_dataset_df() -> pd.DataFrame:
    dfs = [
        HtmlLoad.get_artifact(select_merged=True).sample(3),
        MongoLoad.get_artifact(select_merged=True).sample(3),
        RedditLoad.get_artifact(select_merged=True).sample(3),
        YoutubeLoad.get_artifact(select_merged=True).sample(3),
    ]

    return pd.concat(dfs, ignore_index=True)


# %% #####################################  ANALYZE DF #########################

dataset_df = get_dataset_df()

print(f"Total number of rows in dataset: {len(dataset_df)}")
# Check for duplicate content and IDs
duplicate_content = dataset_df[
    dataset_df.duplicated(subset=["page_content"], keep=False)
]
duplicate_ids = dataset_df[dataset_df.duplicated(subset=["id"], keep=False)]

print(f"\nDuplicate content rows: {len(duplicate_content)}")
print(f"Duplicate ID rows: {len(duplicate_ids)}")

# %% #####################################  GET INDIVIDUAL DFS #########################

mongo_df = MongoLoad.get_artifact(select_merged=True)
youtube_df = YoutubeLoad.get_artifact(select_merged=True)
reddit_df = RedditLoad.get_artifact(select_merged=True)
reddit_general_df = RedditGeneralLoad.get_artifact(select_merged=True)
html_df = HtmlLoad.get_artifact(select_merged=True)


# %% #####################################  YOUTUBE TOKENS #####################################
total_youtube_tokens = (
    youtube_df["page_content"].apply(lambda x: num_tokens_from_string(x)).sum()
)
print(f"Total youtube tokens: {total_youtube_tokens}")

# %% ##################################### FILTER MONGO #####################################

# Count rows where page_content has significantly more words than topic_content
print(f"Total mongo docs: {len(mongo_df)}")
long_mongo_docs = mongo_df[
    (
        mongo_df["page_content"].str.split().str.len()
        - mongo_df["topic_content"].str.split().str.len()
    )
    > 130
]
print(f"Number of documents with >min comment/reply words: {len(long_mongo_docs)}")
print(long_mongo_docs["topic_category_name"].value_counts())
print(f"\n\nFinal docs: {len(long_mongo_docs)}")

# %% ##################################### SAVE SAMPLE DATAFRAME #####################################

dataset_df_sample = get_slim_dataset_df()
# Calculate token count of all page_content in the dataset
total_tokens = 0
for content in dataset_df_sample["page_content"]:
    tokens = num_tokens_from_string(content)
    total_tokens += tokens

print(f"Total tokens in dataset page_content: {total_tokens}")
print(f"Type value counts: {dataset_df_sample['type'].value_counts()}")


# Save the sample dataframe to a parquet file
dataset_df_sample.to_parquet("evals/sample_df.parquet", index=False)
