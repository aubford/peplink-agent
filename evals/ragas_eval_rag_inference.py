# %%
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from ragas import evaluate
import pandas as pd
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)


def generate_testset(testset_size: int = 10, llm_model: str = "gpt-4o") -> pd.DataFrame:
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model_name=llm_model))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    kg = KnowledgeGraph.load("staged_knowledge_graph.json")
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg)

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
    ]

    dataset = generator.generate(
        testset_size=testset_size,
        query_distribution=query_distribution,
    )

    # upload to ragas app
    dataset.upload()
    return dataset.to_pandas()


def evaluate_rag(dataset: list[Document], llm_model: str = "gpt-4o") -> pd.DataFrame:
    from inference.rag_inference import retrieval_chain  # Import your actual RAG chain

    # Run inference on all test samples
    for test_row in dataset:
        result = retrieval_chain.invoke(
            {"input": test_row.eval_sample.user_input, "chat_history": []}  # Add history if testing multi-turn
        )
        test_row.eval_sample.response = result["answer"]
        test_row.eval_sample.retrieved_contexts = [doc.page_content for doc in result["context"]]

    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
            ResponseRelevancy(),
            ContextEntityRecall(),
            NoiseSensitivity(),
        ],
        llm=LangchainLLMWrapper(ChatOpenAI(model_name=llm_model)),
    )
    return result

test_set = generate_testset(2)
evaluation_result = evaluate_rag(test_set)
evaluation_result.to_pandas().to_csv("rag_evaluation_results.csv")
print(evaluation_result)
