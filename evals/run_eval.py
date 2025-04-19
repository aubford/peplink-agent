from datetime import datetime
from pathlib import Path
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.testset.synthesizers.testset_schema import Testset
from ragas.metrics import (
    NonLLMContextRecall,
    Faithfulness,
    ResponseRelevancy,
    ResponseRelevancyDiverse,
    LLMContextPrecisionWithReference,
    FactualCorrectness,
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
)
from inference.rag_inference import RagInference
import pandas as pd
import asyncio
import os


class RagasEval:
    def __init__(
        self,
        evals_dir: Path,
        testset_name: str,
        eval_llm_model: str = "gpt-4o-mini",
        eval_boost_llm_model: str = "gpt-4o",
        inference_llm_model: str = "gpt-4o-mini",
        run_name: str = datetime.now().strftime("%Y-%m-%d_%H_%M"),
    ):
        generated_testset_df = pd.read_parquet(
            evals_dir / "testsets" / testset_name / "generated_testset.parquet"
        )
        nodes_df = pd.read_parquet(
            evals_dir / "testsets" / testset_name / "__nodes.parquet"
        )
        self.test_run_name = f"{testset_name}__{run_name}"
        self.test_set: Testset = self.create_testset(generated_testset_df, nodes_df)

        # Initialize the RAG inference component once
        self.rag_inference = RagInference(llm_model=inference_llm_model)

        self.eval_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=eval_llm_model,
                temperature=0.1,
            )
        )
        self.boost_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=eval_boost_llm_model,
                temperature=0.1,
            )
        )
        self.output_dir = evals_dir / "runs"

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
        df["synthesizer_name"] = "custom"

        return Testset.from_list(df.to_dict(orient="records"))

    async def process_all_rows(self):
        """Process all test rows in parallel using LangChain's native parallelization."""
        # Create a dictionary of tasks for each test row
        async_tasks = {}

        for i, test_row in enumerate(self.test_set):
            async_tasks[f"query_{i}"] = test_row.eval_sample.user_input

        # Use the batch query method from RagInference
        results = await self.rag_inference.batch_query_for_eval(async_tasks)

        # Process results and update test rows
        for i, test_row in enumerate(self.test_set):
            try:
                result = results[f"query_{i}"]
                eval_sample = test_row.eval_sample

                if "answer" not in result or "context" not in result:
                    raise ValueError(f"Unexpected result format: {result.keys()}")

                eval_sample.response = result["answer"]
                eval_sample.retrieved_contexts = [
                    doc.page_content for doc in result["context"]
                ]
            except Exception as e:
                print(f"Error processing result for query {i}: {e}")
                raise

    def save_metrics_summary(self, eval_result_df: pd.DataFrame) -> None:
        """
        Calculate mean for each metric column and append results to test_runs.parquet.

        Args:
            eval_result_df: DataFrame containing the evaluation results
        """
        # Extract only float columns (metrics)
        float_cols = eval_result_df.select_dtypes(include=['float64']).columns

        # Calculate mean for each metric
        metrics_summary = {}
        for col in float_cols:
            metrics_summary[col] = eval_result_df[col].mean()

        # Add testset name
        metrics_summary['testset_name'] = self.test_run_name

        # Create a DataFrame with the summary
        summary_df = pd.DataFrame([metrics_summary])

        # Check if file exists
        test_runs_summary_path = self.output_dir / "test_runs_summary.parquet"
        if os.path.exists(test_runs_summary_path):
            # Read existing file and append
            existing_df = pd.read_parquet(test_runs_summary_path)
            updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
        else:
            updated_df = summary_df

        # Save to parquet
        updated_df.to_parquet(test_runs_summary_path, index=False)

    async def evaluate_rag(self) -> None:
        """Run the evaluation on the test set."""
        # Run the async processing using an event loop
        await self.process_all_rows()

        evaluation_dataset = self.test_set.to_evaluation_dataset()
        eval_result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                Faithfulness(),
                ResponseGroundedness(llm=self.boost_llm),  # NVIDIA
                ResponseRelevancyDiverse(llm=self.boost_llm),
                ResponseRelevancy(llm=self.eval_llm),
                LLMContextPrecisionWithReference(),
                NonLLMContextRecall(),
                ContextRelevance(llm=self.boost_llm),  # NVIDIA
                FactualCorrectness(),
                AnswerAccuracy(llm=self.boost_llm),  # NVIDIA
            ],
            llm=self.eval_llm,
        )

        eval_result_df = eval_result.to_pandas()

        # Save the full results
        eval_result_df.to_parquet(
            self.output_dir / f"result__{self.test_run_name}.parquet"
        )

        # Save metrics summary
        self.save_metrics_summary(eval_result_df)


if __name__ == "__main__":
    evals_dir = Path(__file__).parent
    testset_name = "testset_100__25-04-08-16_44"
    eval = RagasEval(evals_dir, testset_name)
    asyncio.run(eval.evaluate_rag())
