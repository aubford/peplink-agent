from datetime import datetime
from pathlib import Path
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.testset.synthesizers.testset_schema import Testset
from load.batch_manager import BatchManager
from batch_llm import BatchChatOpenAI
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
import os
import json
from typing import cast


class RagasEval:

    def __init__(
        self,
        evals_dir: Path,
        testset_name: str,
        inference_llm_model: str = "gpt-4.1-mini",
        eval_llm_model: str = "gpt-4.1-nano",
        eval_boost_llm_model: str = "gpt-4.1",
        run_name: str | None = None,
    ):
        generated_testset_df = pd.read_parquet(
            evals_dir / "testsets" / testset_name / "generated_testset.parquet"
        )
        nodes_df = pd.read_parquet(
            evals_dir / "testsets" / testset_name / "__nodes.parquet"
        )
        self.test_run_name = (
            f"{testset_name}__{run_name or datetime.now().strftime("%Y-%m-%d_%H_%M")}"
        )
        self.test_set: Testset = self.init_testset(generated_testset_df, nodes_df)

        self.batch_manager = BatchManager(
            base_path=evals_dir / "batches",
            endpoint="/v1/chat/completions",
            batch_name=f"{testset_name}__{run_name or 'default_batch'}",
        )
        self.batch_contexts_path = (
            self.batch_manager.batch_path / "batch_contexts.parquet"
        )

        self.rag_inference = RagInference(
            llm_model=inference_llm_model,
            eval_llm=BatchChatOpenAI(
                model=inference_llm_model,
                temperature=0,
                batch_manager=self.batch_manager,
            ),
        )

        self.eval_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=eval_llm_model,
                temperature=0,
            )
        )
        self.boost_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=eval_boost_llm_model,
                temperature=0,
            )
        )
        self.output_dir = evals_dir / "runs"

    def init_testset(
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

    def create_batch_contexts_file(self, results: dict[str, dict[str, str]]) -> None:
        contexts_data = {
            # result["answer"] contains the custom_id when using the eval LLM
            result["answer"]: [result["context"], query_idx]
            for query_idx, result in results.items()
        }
        df = pd.DataFrame.from_dict(
            contexts_data, orient="index", columns=["context", "query_idx"]
        )
        # Convert 'context' column from list to JSON string for storage
        df["context"] = df["context"].apply(
            lambda x: json.dumps([doc.page_content for doc in x])
        )
        df.index.name = "custom_id"
        df.to_parquet(self.batch_contexts_path)

    def get_batch_results(self) -> None:
        batch_results = self.batch_manager.get_content_if_ready()
        batch_results_df = pd.DataFrame.from_dict(
            batch_results, orient="index", columns=["result"]
        )
        batch_contexts = pd.read_parquet(self.batch_contexts_path)
        batch_contexts["context"] = batch_contexts["context"].apply(json.loads)
        merged_df = batch_contexts.merge(
            batch_results_df, left_index=True, right_index=True
        )
        self.apply_results_to_testset(merged_df.set_index("query_idx"))

    def apply_results_to_testset(self, results: pd.DataFrame):
        # Process results and update testset w/ the answer and contexts
        for i, test_row in enumerate(self.test_set):
            try:
                # Get the row where 'query_idx' equals f"query_{i}"
                result_row = results.loc[f"query_{i}"]
                eval_sample = test_row.eval_sample

                eval_sample.response = cast(str, result_row["result"])
                eval_sample.retrieved_contexts = cast(list[str], result_row["context"])
            except Exception as e:
                print(f"Error processing result for query {i}: {e}")
                raise

    async def generate_batchfile(self):
        """Generate a batchfile.jsonl to be uploaded to OpenAI's Batch API."""
        assert self.test_set is not None, "Test set is not set"

        async_tasks = {}
        for i, test_row in enumerate(self.test_set):
            async_tasks[f"query_{i}"] = test_row.eval_sample.user_input

        self.batch_manager.clear_batch_files()
        results = await self.rag_inference.batch_query_for_eval(async_tasks)
        self.create_batch_contexts_file(results)

        self.batch_manager.create_batch_job()

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
        self.get_batch_results()
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
                FactualCorrectness(llm=self.eval_llm),
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
