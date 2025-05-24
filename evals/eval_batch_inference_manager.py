from typing import Any

from langchain_core.prompts import BasePromptTemplate
from pydantic import BaseModel

from evals.evals_utils import runs_dir
from inference.rag_inference import InferenceBase
from load.batch_manager import BatchManager


class EvalBatchInferenceManager(BatchManager):
    def __init__(
        self,
        run_name: str,
        batch_name: str,
        conversation_template: BasePromptTemplate,
        inference: InferenceBase,
        schema: type[BaseModel] | None = None,
    ):
        super().__init__(
            base_path=runs_dir / run_name / "batches",
            batch_name=batch_name,
            schema=schema,
        )

        # ensure temp is 0 for eval operations
        inference.set_temperature(0)
        self.compiled_inference = inference.compile(
            conversation_template=conversation_template,
            batch_manager=self,
        )

    async def run_queries(self, queries: dict[str, str]) -> dict[str, dict[str, Any]]:
        """
        Process multiple queries in parallel for evaluation purposes using LangChain's
        native batch processing capabilities.

        Args:
            queries: Dictionary of query identifiers to queries

        Returns:
            dict: Mapping of query identifiers to results
        """

        # Get just the inputs for processing
        batch_inputs = [
            {"thread_id": thread_id, "query": query, "chat_history": []}
            for thread_id, query in queries.items()
        ]

        # Use native batch processing with proper rate limiting
        results = await self.compiled_inference.abatch(
            batch_inputs,
            config=[
                {"max_concurrency": 20, "configurable": {"thread_id": i["thread_id"]}}
                for i in batch_inputs
            ],
        )

        for result in results:
            result["custom_id"] = result["answer"]
        return {result["thread_id"]: result for result in results}
