# %%
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
import pandas as pd
from transform.youtube.youtube_transform import YouTubeTransform
from transform.web.web_transform import WebTransform
from transform.html.html_transform import HTMLTransform
from util.document_utils import df_to_documents
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from datetime import datetime

LLM = "gpt-4o"


sample_size = 1


def get_prototyping_dataset() -> list[Document]:
    y_df = YouTubeTransform.get_artifacts()[0].sample(sample_size)
    w_df = WebTransform.get_artifacts()[0].sample(sample_size)
    h_df = HTMLTransform.get_artifacts()[0].sample(sample_size)
    combined_df = pd.concat([y_df, w_df, h_df], ignore_index=True)
    return df_to_documents(combined_df)


def get_dataset_as_documents() -> list[Document]:
    dfs = YouTubeTransform.get_artifacts()
    dfs.extend(WebTransform.get_artifacts())
    dfs.extend(HTMLTransform.get_artifacts())
    combined_df = pd.concat(dfs, ignore_index=True)
    return df_to_documents(combined_df)


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
docs = get_prototyping_dataset()

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

# Save knowledge graph to staged file
# knowledge_graph.save("evals/staged_knowledge_graph.json")
