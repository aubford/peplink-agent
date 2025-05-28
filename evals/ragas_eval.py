from pathlib import Path
from textwrap import dedent
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.testset.synthesizers.testset_schema import Testset
from ragas.metrics import (
    NonLLMContextRecall,
    NonLLMContextPrecisionWithReference,
    Faithfulness,
    ResponseRelevancy,
    ResponseRelevancyDiverse,
    FactualCorrectness,
    AnswerAccuracy,
    # BleuScore,
    # RougeScore,
    # FaithfulnesswithHHEM,
    EmbeddingContextPrecision,
    EmbeddingContextRecall,
)

from evals.eval_batch_inference_manager import EvalBatchInferenceManager
from inference.rag_inference import (
    default_conversation_template,
    InferenceBase,
)
from evals.mock_exam import MockExam
import pandas as pd
import os
import json
from datetime import datetime
from typing import Callable, cast, Any, Literal

from dotenv import load_dotenv
from load.document_index import DocumentIndex
from util.util_main import handle_file_exists
from langsmith import tracing_context

load_dotenv()

MAIN_TESTSET_NAME = "testset-200_main_testset_25-04-23"
evals_dir = Path(__file__).parent


class RagasEval:

    def __init__(
        self,
        run_name: str,
        eval_llm: str,
        inference_fact: Callable[..., InferenceBase],
        query_column: str = "query",
        testset_name: str = MAIN_TESTSET_NAME,
        sample: tuple | Literal[False] = False,
        test_run: bool = False,
        should_create_batch_job: bool = True,
        # Faithfulness: response -> context (do the claims in answer come from context); this metric is expensive
        with_faithfulness: bool = False,
    ):
        self.with_faithfulness = with_faithfulness
        self.query_column = query_column

        generated_testset_df = pd.read_json(
            evals_dir / "testsets" / testset_name / "generated_testset.json"
        )
        if sample:
            generated_testset_df = generated_testset_df[sample[0] : sample[1]]
        nodes_df = pd.read_parquet(
            evals_dir / "testsets" / testset_name / "__nodes.parquet"
        )
        # Merge nodes_df with kg_input_data.parquet on 'id'
        document_index = DocumentIndex.get_document_index()
        nodes_df = pd.merge(
            nodes_df, document_index, on="id", how="inner", suffixes=("", "_di")
        )
        self.runs_dir = evals_dir / "experiments"
        self.output_dir = self.runs_dir / run_name
        self.output_file_path = self.output_dir / f"{run_name}.parquet"
        # If knowledge_graph file already exists, rename it with a timestamp
        handle_file_exists(self.output_file_path, should_raise=False)

        self.test_set: Testset = self.init_testset(generated_testset_df, nodes_df)

        self.test_run = test_run
        self.should_create_batch_job = should_create_batch_job

        inference = inference_fact()
        self.inference_manager = EvalBatchInferenceManager(
            run_name=run_name,
            batch_name=f"{testset_name}_batch",
            conversation_template=default_conversation_template,
            inference=inference,
        )

        self.batch_contexts_path = (
            self.inference_manager.batch_path / "batch_contexts.parquet"
        )

        self.mock_exam = MockExam(
            run_name=run_name,
            inference=inference_fact(),
            output_dir=self.output_dir,
            should_create_batch_job=should_create_batch_job,
            sample=sample,
        )

        self.eval_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=eval_llm,
                temperature=0,
            )
        )

        self.metrics_summary: dict[str, Any] = {
            "run_name": run_name,
            "metadata": dedent(
                f"""
                Testset name: {testset_name}
                Inference LLM: {inference.llm_model}
                Eval LLM: {eval_llm}
                Pinecone index: {inference.pinecone_index_name}
                Sample: {str(sample[0]) + ", " + str(sample[1]) if sample else "full"}
                Datetime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                """
            ).strip(),
        }

    def init_testset(
        self, generated_testset_df: pd.DataFrame, nodes_df: pd.DataFrame
    ) -> Testset:
        # Rename columns to match the expected schema
        df = generated_testset_df.copy()
        df.rename(columns={self.query_column: "user_input"}, inplace=True)
        df.rename(columns={"answer": "reference"}, inplace=True)
        df.rename(columns={"cluster_id": "id"}, inplace=True)
        df["reference_contexts"] = df["node_ids"].apply(
            lambda cluster_node_ids: [
                (row["page_content"], row["technical_summary"])
                for _, row in nodes_df[
                    nodes_df["node_id"].isin(cluster_node_ids)
                ].iterrows()
            ]
        )

        def build_embeddings_tuple(row: pd.Series) -> tuple:
            if row["technical_summary_embedding"] is not None:
                return (
                    row["page_content_embedding"],
                    row["technical_summary_embedding"],
                    row["primary_content_embedding"],
                )
            return (
                row["page_content_embedding"],
                row["primary_content_embedding"],
            )

        df["reference_contexts_embeddings"] = df["node_ids"].apply(
            lambda cluster_node_ids: [
                build_embeddings_tuple(row)
                for _, row in nodes_df[
                    nodes_df["node_id"].isin(cluster_node_ids)
                ].iterrows()
            ]
        )
        df.drop(columns=["document_ids", "documents_text"], inplace=True)
        df["synthesizer_name"] = "custom"
        df["id"] = df["id"].astype(str)

        return Testset.from_list(df.to_dict(orient="records"))

    def create_batch_contexts_file(self, results: dict[str, dict[str, str]]) -> None:
        df = pd.DataFrame.from_dict(
            {
                result["custom_id"]: [
                    cluster_id,
                    result["context"],
                    result["query"],
                    result["retrieval_query"],
                ]
                for cluster_id, result in results.items()
            },
            orient="index",
            columns=["cluster_id", "context", "query", "retrieval_query"],
        )
        # Convert 'context' column from list to JSON string for storage
        df["context"] = df["context"].apply(
            lambda x: json.dumps([doc.page_content for doc in x])
        )
        df.index.name = "custom_id"
        df.to_parquet(self.batch_contexts_path)

    def get_batch_results(self) -> None:
        batch_results = self.inference_manager.get_content_if_ready()
        batch_contexts = pd.read_parquet(self.batch_contexts_path)
        batch_contexts["context"] = batch_contexts["context"].apply(json.loads)

        batch_results_df = pd.DataFrame.from_dict(
            batch_results, orient="index", columns=["result"]
        )
        merged_df = batch_contexts.merge(
            batch_results_df, left_index=True, right_index=True
        )

        self.apply_results_to_testset(merged_df.set_index("cluster_id"))

    def apply_results_to_testset(self, results: pd.DataFrame):
        # Process results and update testset w/ the answer and contexts
        for test_row in self.test_set:
            eval_sample = test_row.eval_sample
            try:
                # testset cluster_id is now the index of results
                result_row = results.loc[eval_sample.id]

                eval_sample.response = cast(str, result_row["result"])
                eval_sample.retrieved_contexts = cast(list[str], result_row["context"])
            except Exception as e:
                print(f"Error processing result for query {eval_sample.id}: {e} ")
                raise

    async def generate_batchfiles(self):
        """Generate a batchfile.jsonl to be uploaded to OpenAI's Batch API."""
        # to trace inference, just run the app
        with tracing_context(enabled=False):
            assert self.test_set is not None, "Test set is not set"
            async_tasks = {}
            for test_row in self.test_set:
                async_tasks[test_row.eval_sample.id] = test_row.eval_sample.user_input

            # dont clear existing batch files when testing
            if self.should_create_batch_job:
                self.inference_manager.clear_batch_files()
            results = await self.inference_manager.run_queries(async_tasks)
            self.create_batch_contexts_file(results)

            await self.mock_exam.generate_batchfile()

            # create a new batch job if not testing
            if self.should_create_batch_job:
                self.inference_manager.create_batch_job()

    def save_metrics_summary(self, eval_result_df: pd.DataFrame) -> None:
        """
        Calculate mean for each metric column and append results to test_runs.parquet.

        Args:
            eval_result_df: DataFrame containing the evaluation results
        """
        if self.test_run:
            return

        # Extract only float columns (metrics)
        float_cols = eval_result_df.select_dtypes(include=["float64"]).columns

        # Add metrics averages
        for col in float_cols:
            self.metrics_summary[col] = eval_result_df[col].mean()

        # Add mock exam score and missed questions
        score, missed_questions = self.mock_exam.get_results()
        self.metrics_summary["mock_exam_score"] = score
        self.metrics_summary["mock_exam_missed_questions"] = [
            q["question_id"] for q in missed_questions
        ]

        # Calculate correlation between nv_accuracy and factual_correctness
        corr = eval_result_df["nv_accuracy"].corr(
            eval_result_df["factual_correctness(mode=recall)"]
        )
        self.metrics_summary["nv_fc_corr"] = corr

        # Calculate correlation between answer_relevancy and answer_relevancy_diverse
        corr = eval_result_df["answer_relevancy"].corr(
            eval_result_df["answer_relevancy_diverse"]
        )
        self.metrics_summary["ar_ar_diverse_corr"] = corr

        # Create a DataFrame with the summary
        summary_df = pd.DataFrame([self.metrics_summary])

        # Check if file exists
        test_runs_summary_path = self.runs_dir / "experiments_summary.parquet"
        if os.path.exists(test_runs_summary_path):
            # Read existing file and append
            existing_df = pd.read_parquet(test_runs_summary_path)
            updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
        else:
            updated_df = summary_df

        # Move metadata to the end
        cols = [col for col in updated_df.columns if col != "metadata"] + ["metadata"]
        updated_df = updated_df[cols]
        # Round all float columns before saving
        float_cols = updated_df.select_dtypes(include=["float64"]).columns
        updated_df[float_cols] = updated_df[float_cols].round(5)
        # Save to parquet
        updated_df.to_parquet(test_runs_summary_path, index=False)

    async def evaluate_rag(self) -> None:
        """Run the evaluation on the test set."""
        self.get_batch_results()
        evaluation_dataset = self.test_set.to_evaluation_dataset()
        metrics = [
            # use HHEM-Open hallucination detection classifier model
            # FaithfulnesswithHHEM(batch_size=2),
            # response -> question (does the answer address the entire question)
            # todo: turn ResponseRelevancy off once we have confirmed ResponseRelevancyDiverse works better
            ResponseRelevancy(),
            ResponseRelevancyDiverse(),
            # context -> reference contexts (do the contexts match the reference contexts)
            # note that you can't compare this across context types! (e.g. summaries vs. full docs)
            # we care much more about recall since there are plenty of relevant docs that may not have been
            # part of the KG cluster used to create this sample.
            EmbeddingContextPrecision(),
            NonLLMContextPrecisionWithReference(),
            EmbeddingContextRecall(),
            NonLLMContextRecall(),
            # response -> reference answer (ground truth)
            FactualCorrectness(mode="recall"),
            AnswerAccuracy(),  # NVIDIA
            # less accurate metrics
            # BleuScore(),
            # RougeScore(),
        ]

        if self.with_faithfulness:
            # response -> context (do the claims in answer come from context)
            metrics.append(Faithfulness())

        eval_result = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=self.eval_llm,
        )

        doc_separator = "\n\n" + "-" * 20 + "\n\n"
        eval_result_df = eval_result.to_pandas()
        try:
            eval_result_df["reference_contexts"] = eval_result_df[
                "reference_contexts"
            ].apply(lambda x: doc_separator.join([doc[0] for doc in x]))
        except Exception as e:
            print(f"Error processing 'reference_contexts': {e}")
            eval_result_df["reference_contexts"] = ""

        try:
            eval_result_df["retrieved_contexts"] = eval_result_df[
                "retrieved_contexts"
            ].apply(lambda x: doc_separator.join(x if isinstance(x, list) else []))
        except Exception as e:
            print(f"Error processing 'retrieved_contexts': {e}")
            eval_result_df["retrieved_contexts"] = ""

        try:
            eval_result_df["reference_answer_statements_recall"] = eval_result_df[
                "reference_answer_statements_recall"
            ].apply(
                lambda x: (
                    doc_separator.join(
                        [
                            f"Statement: {doc['statement']} - {'✅' if doc['verdict'] else '❌'}\n\nReason: {doc['reason']}"
                            for doc in x
                        ]
                    )
                    if isinstance(x, list)
                    else ""
                )
            )
        except Exception as e:
            print(f"Error processing 'reference_answer_statements_recall': {e}")
            eval_result_df["reference_answer_statements_recall"] = ""

        float_cols = eval_result_df.select_dtypes(include=["float64"]).columns
        eval_result_df[float_cols] = eval_result_df[float_cols].round(5)
        eval_result_df.to_parquet(self.output_file_path)
        # Save the full results
        eval_result_df.drop(columns=["reference_contexts_embeddings"], inplace=True)
        self.save_metrics_summary(eval_result_df)
