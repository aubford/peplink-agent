# %%
from datetime import datetime
from pathlib import Path
from config.config import global_config
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.evaluation import EvaluationResult
import pandas as pd
from typing import Any
from ragas import EvaluationDataset
from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
)
from inference.rag_inference import RagInference


class RagasEval:
    def __init__(
        self,
        evals_dir: Path,
        testset_name: str,
        llm_model: str = "gpt-4o-mini",
    ):
        generated_testset_df = pd.read_parquet(
            evals_dir / "testsets" / testset_name / "generated_testset.parquet"
        )[0:1]
        nodes_df = pd.read_parquet(
            evals_dir / "testsets" / testset_name / "__nodes.parquet"
        )
        self.testset_name = testset_name
        self.test_set: Testset = self.create_testset(generated_testset_df, nodes_df)
        self.llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
        self.output_dir = evals_dir / "runs"
        self.rag_inference = RagInference(llm_model=llm_model)

    def create_testset(
        self, generated_testset_df: pd.DataFrame, nodes_df: pd.DataFrame
    ) -> Testset:
        # Rename columns to match the expected schema
        df = generated_testset_df.copy()
        df.rename(columns={"query": "user_input"}, inplace=True)
        df.rename(columns={"answer": "reference"}, inplace=True)
        df["reference_contexts"] = df["node_ids"].apply(
            lambda x: nodes_df[nodes_df["node_id"].isin(x)]["page_content"].tolist()
        )
        df.drop(columns=["document_ids", "documents_text", "cluster_id"], inplace=True)
        df["synthesizer_name"] = "multi_hop_abstract_query_synthesizer"

        return Testset.from_list(df.to_dict(orient="records"))

    def evaluate_rag(self) -> None:
        for test_row in self.test_set:
            user_query: Any = test_row.eval_sample.user_input
            result = self.rag_inference.query(user_query)
            test_row.eval_sample.response = result["answer"]
            test_row.eval_sample.retrieved_contexts = [
                doc.page_content for doc in result["context"]
            ]

        evaluation_dataset = self.test_set.to_evaluation_dataset()
        eval_result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                LLMContextRecall(),
                Faithfulness(),
                FactualCorrectness(),
                ResponseRelevancy(),
                ContextEntityRecall(),
            ],
            llm=self.llm,
        )
        eval_result.to_pandas().to_parquet(
            self.output_dir
            / f"result__{self.testset_name}___{datetime.now().strftime('%Y-%m-%d_%H_%M')}.parquet"
        )


if __name__ == "__main__":
    evals_dir = Path(__file__).parent
    testset_name = "testset_10__25-04-08-16_58"
    eval = RagasEval(evals_dir, testset_name, llm_model="gpt-4o-mini")
    eval.evaluate_rag()
