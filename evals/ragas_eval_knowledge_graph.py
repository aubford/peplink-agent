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

from load.reddit_general.reddit_general_load import RedditGeneralLoad
from load.reddit.reddit_load import RedditLoad
from load.html.html_load import HtmlLoad
from load.web.web_load import WebLoad
from load.youtube.youtube_load import YoutubeLoad
from load.mongo.mongo_load import MongoLoad

LLM = "gpt-4o-mini"
sample_size = 10


def count_words() -> None:
    dfs = YoutubeLoad.get_artifacts()
    dfs.extend(WebLoad.get_artifacts())
    dfs.extend(HtmlLoad.get_artifacts())
    dfs.extend(RedditGeneralLoad.get_artifacts())
    dfs.extend(RedditLoad.get_artifacts())
    dfs.extend(MongoLoad.get_artifacts())
    combined_df = pd.concat(dfs, ignore_index=True)
    words = combined_df["page_content"].str.split().str.len().sum()
    print(words)


count_words()

# %%


def get_prototyping_dataset_as_documents() -> list[Document]:
    y_df = YoutubeLoad.get_artifact(select_merged=True).sample(sample_size)
    w_df = WebLoad.get_artifact(select_merged=True).sample(sample_size)
    h_df = HtmlLoad.get_artifact(select_merged=True).sample(sample_size)
    rg_df = RedditGeneralLoad.get_artifact(select_merged=True).sample(sample_size)
    r_df = RedditLoad.get_artifact(select_merged=True).sample(sample_size)
    m_df = MongoLoad.get_artifact(select_merged=True).sample(sample_size)
    combined_df = pd.concat([y_df, w_df, h_df, rg_df, r_df, m_df], ignore_index=True)
    return df_to_documents(combined_df)


def get_dataset_as_documents() -> list[Document]:
    dfs = YoutubeLoad.get_artifacts()
    dfs.extend(WebLoad.get_artifacts())
    dfs.extend(HtmlLoad.get_artifacts())
    dfs.extend(RedditGeneralLoad.get_artifacts())
    dfs.extend(RedditLoad.get_artifacts())
    dfs.extend(MongoLoad.get_artifacts())
    combined_df = pd.concat(dfs, ignore_index=True)
    return df_to_documents(combined_df)

# %%
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


knowledge_graph = KnowledgeGraph()
docs = get_prototyping_dataset_as_documents()

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

transform(knowledge_graph, docs)

print(knowledge_graph.nodes)


# Save knowledge graph to staged file
# knowledge_graph.save("evals/staged_knowledge_graph.json")
