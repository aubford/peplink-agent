# %%
from load.base_load import BaseLoad
from langchain.docstore.document import Document
import json
from openai import OpenAI
from typing import List, Dict, Any, Optional, Literal, Union
import time
import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

load_dotenv()

class RedditLoad(BaseLoad):
    folder_name = "reddit"
    system_prompt = "Complete this!!"

    def __init__(self):
        super().__init__()  # Call BaseLoad constructor with no arguments
        # Create the BatchManager after BaseLoad initialization so staging_folder is available
        self.batch_manager = BatchManager(Path(self.staging_folder), self.system_prompt)

    def create_merged_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets into a single dataframe and perform operations like deduplication."""
        return pd.concat(dfs)


ValidEndpoints = Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"]


class BatchManager:
    """
    Handles batch processing functionality using OpenAI's Batch API.

    This class provides methods to create, monitor, and retrieve results from batch jobs.
    Batch jobs are processed asynchronously by OpenAI and completed within 24 hours.
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        system_prompt: str,
        endpoint: ValidEndpoints = "/v1/chat/completions",
    ):
        """
        Initialize the BatchManager with OpenAI client and a base path for file operations.

        Args:
            base_path: Directory path to use for storing batch files and results
            endpoint: API endpoint to use for the batch (must be one of '/v1/chat/completions', '/v1/embeddings', '/v1/completions')
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

        # Ensure base_path is a Path object and exists
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.file_name = self.base_path / "batchfile.jsonl"
        self.output_file_name = self.base_path / "batch_results.jsonl"
        self.status_file_name = self.base_path / "batch_status.json"
        self.system_prompt = system_prompt
        self.endpoint: ValidEndpoints = endpoint
        self.batch_id = None

    def create_batch_tasks(
        self,
        items: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> List[Dict]:
        """
        Create a list of batch tasks from item dictionaries.

        Args:
            items: List of dictionaries with 'id' and 'prompt' properties to process
            model: OpenAI model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate

        Returns:
            List of task dictionaries ready for batch processing
        """
        tasks = []

        for item in items:
            task = {
                "custom_id": item["id"],
                "method": "POST",
                "url": self.endpoint,
                "body": {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": item["prompt"]},
                    ],
                },
            }
            tasks.append(task)

        return tasks

    def create_batch_file(self, tasks: List[Dict[str, Any]]) -> Path:
        """
        Create a JSONL file from a list of task dictionaries.

        Args:
            tasks: List of task dictionaries to be processed in batch

        Returns:
            Path to the created file
        """
        with open(self.file_name, "w") as file:
            for task in tasks:
                file.write(json.dumps(task) + "\n")
        return self.file_name

    def write_status_file(self, batch_job: Any) -> None:
        """
        Write batch job status to a JSON file.

        Args:
            batch_job: The batch job object to write to file
        """
        print(f"Batch status: {batch_job.status}")
        with open(self.status_file_name, "w") as f:
            json.dump(batch_job.model_dump(), f, indent=2)

    def create_batch_job(self) -> Any:
        """
        Create a batch job using a JSONL file.

        Returns:
            Batch job object
        """
        # Upload the file
        batch_file = self.client.files.create(
            file=open(self.file_name, "rb"), purpose="batch"
        )

        # Create the batch job using the instance's endpoint
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint=self.endpoint,
            completion_window="24h",
        )
        self.write_status_file(batch_job)

        self.batch_id = batch_job.id

        return batch_job

    def get_batch_status(self) -> Any:
        """
        Retrieve the status of a batch job.

        Args:
            batch_id: ID of the batch job

        Returns:
            Batch job status object
        """
        if not self.batch_id:
            raise ValueError("Batch ID is not set")
        try:
            batch = self.client.batches.retrieve(self.batch_id)
            self.write_status_file(batch)
            return batch
        except Exception as e:
            print(f"Error retrieving batch status: {str(e)}")
            return None

    def check_batch_and_get_results(self) -> Dict[str, Any]:
        """
        Check the status of a batch job and retrieve results if complete.
        This is an on-demand method with no waiting or polling.

        Args:
            batch_id: ID of the batch job to check

        Returns:
            Dictionary containing status information and results if complete
        """
        batch_job = self.get_batch_status()

        if not batch_job:
            return {
                "status": "error",
                "message": f"Could not retrieve batch job {self.batch_id}",
            }

        # If job completed, fetch and save results
        if batch_job.status == "completed":
            try:
                results = self.get_results(batch_job)
                return {
                    "status": "completed",
                    "batch_id": self.batch_id,
                    "results": results,
                    "output_file": str(self.output_file_name),
                }
            except Exception as e:
                return {
                    "status": "error",
                    "batch_id": self.batch_id,
                    "message": f"Error retrieving results: {str(e)}",
                }
        elif batch_job.status == "failed":
            return {
                "status": "failed",
                "batch_id": self.batch_id,
                "message": f"Batch job failed",
            }
        else:
            # Job is still in progress
            return {
                "status": batch_job.status,
                "batch_id": self.batch_id,
                "message": f"Job is still {batch_job.status}. Check again later.",
            }

    def get_results(self, batch_job: Any) -> List[Dict]:
        """
        Retrieve and parse results from a completed batch job.

        Args:
            batch_job: Completed batch job object
            output_file_name: Optional file name to save the results (without path)

        Returns:
            List of result dictionaries
        """
        # Get the result file content
        result_file_id = batch_job.output_file_id
        result_content = self.client.files.content(result_file_id).content

        # Save the result file if name is provided
        if self.output_file_name:
            with open(self.output_file_name, "wb") as file:
                file.write(result_content)

        # Parse the results
        results = []
        for line in result_content.decode("utf-8").strip().split("\n"):
            if line:
                results.append(json.loads(line))

        return results

    def run_complete_batch(
        self,
        items: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 300,
    ) -> Dict[str, Any]:
        """
        Run a batch process without waiting for completion.

        Args:
            items: List of dictionaries with 'id' and 'prompt' properties to process
            model: OpenAI model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with batch job information for later status checking
        """
        # Create tasks
        tasks = self.create_batch_tasks(
            items=items,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Create batch file using the instance's file_name
        self.create_batch_file(tasks)

        # Create batch job
        batch_job = self.create_batch_job()

        # Return batch job information
        return {
            "status": batch_job.status,
            "batch_id": batch_job.id,
            "message": "Batch job created. Use check_batch_and_get_results method later to check status and retrieve results.",
            "created_at": batch_job.created_at,
            "expected_completion": "within 24 hours",
        }

    def test_batchfile(self, limit: int = 2) -> Dict[str, Any]:
        """
        Test the first n tasks in the batch file against the regular API endpoint.

        This method is useful for verifying that your batch configuration works correctly
        before submitting the full batch job.

        Args:
            limit: Number of tasks from the batch file to test (default: 2)

        Returns:
            Dictionary with test results and comparison information
        """
        results = []

        try:
            # Check if batch file exists
            if not self.file_name.exists():
                return {
                    "status": "error",
                    "message": "Batch file does not exist. Create a batch file first.",
                }

            # Read tasks from the batch file
            tasks = []
            with open(self.file_name, "r") as file:
                for idx, line in enumerate(file):
                    if idx >= limit:
                        break
                    tasks.append(json.loads(line))

            if not tasks:
                return {
                    "status": "error",
                    "message": "No tasks found in the batch file.",
                }

            # Run each task against the regular API
            for task in tasks:
                body = task.get("body", {})
                messages = body.get("messages", [])
                model = body.get("model", "gpt-4o-mini")
                temperature = body.get("temperature", 0.2)
                max_tokens = body.get("max_tokens", 300)

                # Call the OpenAI API directly
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Extract usage data safely
                usage_data = {}
                if hasattr(response, "usage") and response.usage:
                    usage_data = {
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            response.usage, "completion_tokens", 0
                        ),
                    }

                results.append(
                    {
                        "task_id": task.get("custom_id"),
                        "prompt": messages[-1].get("content") if messages else "",
                        "response": response.choices[0].message.content,
                        "model": model,
                        "finish_reason": response.choices[0].finish_reason,
                        "usage": usage_data,
                    }
                )

                # Add a small delay to avoid rate limits
                time.sleep(0.5)

            return {
                "status": "success",
                "message": f"Successfully tested {len(results)} tasks",
                "results": results,
                "total_tasks_in_file": sum(1 for _ in open(self.file_name, "r")),
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error testing batch file: {str(e)}",
                "results": results,
            }
