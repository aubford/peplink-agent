import os
import json
import time
import shutil
from typing import Any, Literal
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

    Step 1, Create Batchfile: Either self.create_batch_task() or self.create_batch_tasks_to_batchfile()
    Step 2, Start the batch job: self.create_batch_job()
    Step 3: Wait until the batch job is completed.
    Step 4, Get results: self.check_batch_and_get_results() or self.get_content_if_ready()
    """

    def __init__(
        self,
        base_path: Path,
        endpoint: ValidEndpoints = "/v1/chat/completions",
        batch_name: str = "batch",
        schema: type[BaseModel] | None = None,
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
        batch_path = base_path / batch_name
        batch_path.mkdir(parents=True, exist_ok=True)

        self.batch_path = batch_path
        self.file_name = batch_path / "batchfile.jsonl"
        self.output_file_name = batch_path / "batch_results.json"
        self.status_file_name = batch_path / "batch_status.json"
        self.endpoint: ValidEndpoints = endpoint
        self.schema = schema

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

    def clear_batch_files(self) -> None:
        """
        Clear all batch files in the batch directory.
        """
        if self.batch_path.exists():
            shutil.rmtree(self.batch_path)
        self.batch_path.mkdir(parents=True, exist_ok=True)

    def create_batch_task(
        self,
        custom_id: str,
        messages: list[dict],  # List of non-system messages
        system_prompt: str,
        schema: type[BaseModel] | None = None,
        model: str = "gpt-4.1-nano",
        temperature: float = 0.2,
        max_tokens: int = 5000,
        **kwargs,
    ) -> dict:
        """
        Create a single batch task dictionary.

        Args:
            custom_id: Unique identifier for the task
            messages: List of non-system message dicts (e.g., user/assistant)
            system_prompt: Content for the system message
            schema: Pydantic schema for response_format
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens for response
            **kwargs: Additional keyword arguments for the body
        """
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": self.endpoint,
            "body": {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *messages,
                ],
                **kwargs,
            },
        }
        schema = schema or self.schema
        if schema:
            task["body"]["response_format"] = type_to_response_format_param(schema)  # type: ignore
        return task

    def create_batch_tasks_to_batchfile(
        self,
        items: list[dict[str, str]],
        system_prompt: str,
        schema: type[BaseModel] | None = None,
        model: str = "gpt-4.1-nano",
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> list[dict]:
        """
        Create a list of batch tasks from item dictionaries.

        Returns:
            List of task dictionaries ready for batch processing
        """
        tasks = [
            self.create_batch_task(
                custom_id=item["id"],
                messages=[{"role": "user", "content": item["prompt"]}],
                system_prompt=system_prompt,
                schema=schema or self.schema,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for item in items
        ]

        with open(self.file_name, "w") as file:
            for task in tasks:
                file.write(json.dumps(task) + "\n")
        return tasks

    def _write_status_file(self, batch_job: Any) -> None:
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
        self._write_status_file(batch_job)

        self.batch_id = batch_job.id

        return batch_job

    def _get_batch_status(self) -> Any:
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
            self._write_status_file(batch_job)
            return batch_job
        except Exception as e:
            print(f"Error retrieving batch status: {str(e)}")
            return None

    def check_batch_and_get_results(self) -> dict[str, Any]:
        """
        Check the status of a batch job and retrieve results if complete.
        This is an on-demand method, no waiting or polling.

        Returns:
            Dictionary containing status information and results if complete
        """
        batch_job = self._get_batch_status()
        batch_id = batch_job.id

        if not batch_job:
            return {
                "status": "error",
                "message": f"Could not retrieve batch job",
            }

        # If job completed, fetch and save results
        if batch_job.status == "completed":
            try:
                results = self._get_results(batch_job)
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

    def get_content_if_ready(self) -> dict[str, str]:
        # If results file exists, load and return its contents
        if self.output_file_name.exists():
            with open(self.output_file_name, "r") as f:
                results = json.load(f)
            return self._map_custom_id_to_content(results)
        # Otherwise, check status and fetch from API if completed
        batch_job = self._get_batch_status()
        if batch_job.status == "completed":
            results = self._get_results(batch_job)
            return self._map_custom_id_to_content(results)
        print(f"**Batch job status: {batch_job.status}")
        raise ValueError("Batch job is not completed")

    def _map_custom_id_to_content(self, results: list[dict]) -> dict[str, str]:
        """
        Map a list of result dicts to a {custom_id: content} dictionary.
        """
        return {
            item["custom_id"]: item["response"]["body"]["choices"][0]["message"][
                "content"
            ]
            for item in results
        }

    def _get_results(self, batch_job: Any) -> list[dict[str, Any]]:
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

        # Parse the results
        results = []
        for line in result_content.decode("utf-8").strip().split("\n"):
            if line:
                results.append(json.loads(line))

        # Save as JSON file
        with open(self.output_file_name, "w") as file:
            json.dump(results, file, indent=2)

        return results

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

        return tasks, messages

    def test_batchfile(self, limit: int = 1) -> dict[str, Any]:
        """
        Test the first n tasks in the batch file against the regular API endpoint.
        Verify that your batch configuration works correctly before starting the job.

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
                model = body.get("model", "gpt-4.1-nano")
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
