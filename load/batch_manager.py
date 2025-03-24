import os
import json
import time
from typing import List, Dict, Any, Literal
from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI

# noinspection PyProtectedMember
from openai.lib._parsing._completions import type_to_response_format_param

# Define ValidEndpoints type
ValidEndpoints = Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"]


class BatchManager:
    """
    Handles batch processing functionality using OpenAI's Batch API.

    This class provides methods to create, monitor, and retrieve results from a single batch job.
    """

    def __init__(
        self,
        base_path: Path,
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

        # Ensure base_path and batch subfolder exist
        base_path.mkdir(parents=True, exist_ok=True)
        batch_path = base_path / "batch"
        batch_path.mkdir(parents=True, exist_ok=True)

        self.file_name = batch_path / "batchfile.jsonl"
        self.output_file_name = batch_path / "batch_results.jsonl"
        self.status_file_name = batch_path / "batch_status.json"
        self.endpoint: ValidEndpoints = endpoint

        # Load batch_id from status file if it exists
        if self.status_file_name.exists():
            with open(self.status_file_name, "r") as status_file:
                status_data = json.load(status_file)
                self.batch_id = status_data.get("id")
        else:
            self.batch_id = None

    @property
    def current_batch_id(self) -> str | None:
        """
        Get the current batch ID if it is set, otherwise try to load it from the status file.
        """
        if self.status_file_name.exists() and not self.batch_id:
            with open(self.status_file_name, "r") as status_file:
                status_data = json.load(status_file)
                return status_data.get("id")
        else:
            return self.batch_id

    def create_batch_tasks(
        self,
        items: List[Dict[str, str]],
        system_prompt: str,
        schema: type[BaseModel],
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> List[Dict]:
        """
        Create a list of batch tasks from item dictionaries.

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
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item["prompt"]},
                    ],
                    "response_format": type_to_response_format_param(schema),
                },
            }
            tasks.append(task)

        with open(self.file_name, "w") as file:
            for task in tasks:
                file.write(json.dumps(task) + "\n")
        return tasks

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
        print(f"Batch job created: {batch_job.id}")
        self.write_status_file(batch_job)

        self.batch_id = batch_job.id

        return batch_job

    def get_batch_status(self) -> Any:
        """
        Retrieve the status of a batch job.

        Returns:
            Batch job status object
        """
        batch_id = self.current_batch_id
        if not batch_id:
            raise ValueError("No batch ID or status file found.")
        try:
            batch_job = self.client.batches.retrieve(batch_id)
            self.write_status_file(batch_job)
            return batch_job
        except Exception as e:
            print(f"Error retrieving batch status: {str(e)}")
            return None

    def check_batch_and_get_results(self) -> Dict[str, Any]:
        """
        Check the status of a batch job and retrieve results if complete.
        This is an on-demand method, no waiting or polling.

        Returns:
            Dictionary containing status information and results if complete
        """
        batch_job = self.get_batch_status()
        batch_id = batch_job.id

        if not batch_job:
            return {
                "status": "error",
                "message": f"Could not retrieve batch job",
            }

        # If job completed, fetch and save results
        if batch_job.status == "completed":
            try:
                results = self.get_results(batch_job)
                return {
                    "status": "completed",
                    "batch_id": batch_id,
                    "results": results,
                    "output_file": str(self.output_file_name),
                }
            except Exception as e:
                return {
                    "status": "error",
                    "batch_id": batch_id,
                    "message": f"Error retrieving results: {str(e)}",
                }
        elif batch_job.status == "failed":
            return {
                "status": "failed",
                "batch_id": batch_id,
                "message": f"Batch job failed",
            }
        else:
            # Job is still in progress
            return {
                "status": batch_job.status,
                "batch_id": batch_id,
                "message": f"Job is still {batch_job.status}. Check again later.",
            }

    def get_results(self, batch_job: Any) -> List[Dict]:
        """
        Retrieve and parse results from a completed batch job.

        Args:
            batch_job: Completed batch job object

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

    def run_batch_and_job(
        self,
        items: List[Dict[str, str]],
        system_prompt: str,
        schema: type[BaseModel],
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
            schema: Schema for the response
            system_prompt: System prompt for the batch
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with batch job information for later status checking
        """
        # Create tasks and batchfile
        self.create_batch_tasks(
            items=items,
            system_prompt=system_prompt,
            schema=schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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

    def get_batchfile(self) -> tuple[list[dict], list[list[dict]]]:
        # Check if batch file exists
        if not self.file_name.exists():
            raise ValueError("Batch file does not exist.")

        # Read tasks from the batch file
        tasks = []
        with open(self.file_name, "r") as file:
            for line in file:
                tasks.append(json.loads(line))

        if not tasks:
            raise ValueError("No tasks found in the batch file.")

        messages = [
            [
                message["content"].replace("\\n", "\n")
                for message in call["body"]["messages"]
            ]
            for call in tasks
        ]

        return (tasks, messages)

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

            result = {
                "status": "success",
                "message": f"Successfully tested {len(results)} tasks",
                "results": results,
                "total_tasks_in_file": sum(1 for _ in open(self.file_name, "r")),
            }
            print("Test Batchfile Success!!!!!!!!!!")
            print(result)
            return result

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error testing batch file: {str(e)}",
                "results": results,
            }
