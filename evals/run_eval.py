# %%
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from ragas import evaluate
from ragas.evaluation import EvaluationResult
import pandas as pd
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)


def evaluate_rag(
    dataset: list[Document], llm_model: str = "gpt-4o"
) -> EvaluationResult:
    from inference.rag_inference import retrieval_chain

    # Run inference on all test samples
    for test_row in dataset:
        result = retrieval_chain.invoke(
            {"input": test_row.eval_sample.user_input, "chat_history": []}
        )
        test_row.eval_sample.response = result["answer"]
        test_row.eval_sample.retrieved_contexts = [
            doc.page_content for doc in result["context"]
        ]

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


evaluation_result = evaluate_rag(test_set)
evaluation_result.to_pandas().to_csv("evals/rag_evaluation_results.csv")
print(evaluation_result)
