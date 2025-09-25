import os
import json
import time
import shutil
import logging
from typing import Any, Literal
from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI

# noinspection PyProtectedMember
from openai.lib._parsing._completions import type_to_response_format_param

# Define ValidEndpoints type
ValidEndpoints = Literal[
    "/v1/chat/completions", "/v1/embeddings", "/v1/completions", "/v1/responses"
]


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
        use_responses_api: bool = False,
    ):
        """Initialize BatchManager with configurable API endpoint.

        Args:
            base_path: Base directory for batch files
            endpoint: API endpoint to use
            batch_name: Name for the batch
            schema: Optional schema for structured output
            use_responses_api: Whether to use Responses API format instead of Chat Completions
        """
        self.base_path = Path(base_path)
        self.batch_name = batch_name
        self.use_responses_api = use_responses_api

        # Set endpoint based on API choice
        if use_responses_api:
            self.endpoint = "/v1/responses"
        else:
            self.endpoint = endpoint

        self.schema = schema
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.file_name = self.base_path / f"{batch_name}.jsonl"
        self.status_file = self.base_path / f"{batch_name}_status.json"
        self.results_file = self.base_path / f"{batch_name}_results.jsonl"

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

        # Load batch_id from status file if it exists
        if self.status_file.exists():
            with open(self.status_file, "r") as status_file:
                status_data = json.load(status_file)
                self.batch_id = status_data.get("id")
        else:
            self.batch_id = None

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    @property
    def current_batch_id(self) -> str | None:
        """
        Get the current batch ID if it is set, otherwise try to load it from the status file.
        """
        if self.status_file.exists() and not self.batch_id:
            with open(self.status_file, "r") as status_file:
                status_data = json.load(status_file)
                return status_data.get("id")
        else:
            return self.batch_id

    def clear_batch_files(self) -> None:
        """
        Clear all batch files in the batch directory.
        """
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

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
        if self.use_responses_api:
            # Responses API format
            # Convert messages to Responses API input format
            input_messages = [
                {"role": "system", "content": system_prompt},
                *messages,
            ]

            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": self.endpoint,
                "body": {
                    "model": model,
                    "input": input_messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    **kwargs,
                },
            }

            # Note: Responses API may handle structured output differently
            # For now, we'll include the response_format if schema is provided
            schema = schema or self.schema
            if schema:
                task["body"]["response_format"] = type_to_response_format_param(schema)  # type: ignore
        else:
            # Chat Completions API format (original)
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
        with open(self.status_file, "w") as f:
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
                    "output_file": str(self.results_file),
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
        if self.results_file.exists():
            with open(self.results_file, "r") as f:
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
        Handles both Chat Completions API and Responses API formats.
        """
        content_map = {}

        for item in results:
            custom_id = item["custom_id"]
            response_body = item["response"]["body"]

            try:
                if self.use_responses_api:
                    # Responses API format: extract from output[0].content[0].text
                    output = response_body.get("output", [])
                    if output and len(output) > 0:
                        content_blocks = output[0].get("content", [])
                        content = ""
                        for content_block in content_blocks:
                            if content_block.get("type") == "output_text":
                                content += content_block.get("text", "")
                        content_map[custom_id] = content
                    else:
                        content_map[custom_id] = ""
                else:
                    # Chat Completions API format: extract from choices[0].message.content
                    content_map[custom_id] = response_body["choices"][0]["message"]["content"]

            except (KeyError, IndexError) as e:
                self.logger.error(f"Error extracting content for {custom_id}: {e}")
                content_map[custom_id] = ""

        return content_map

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
        with open(self.results_file, "w") as file:
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
                model = body.get("model", "gpt-4.1-nano")
                temperature = body.get("temperature", 0.2)

                if self.use_responses_api:
                    # Responses API format
                    input_data = body.get("input", [])
                    max_completion_tokens = body.get("max_completion_tokens", 300)

                    # Call the OpenAI Responses API directly
                    response = self.client.responses.create(
                        model=model,
                        input=input_data,
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                    )

                    # Extract content from Responses API response
                    content = ""
                    if response.output and len(response.output) > 0:
                        for output_item in response.output:
                            if hasattr(output_item, 'content') and output_item.content:
                                for content_block in output_item.content:
                                    if hasattr(content_block, 'text'):
                                        content += content_block.text
                else:
                    # Chat Completions API format (original)
                    messages = body.get("messages", [])
                    max_tokens = body.get("max_tokens", 300)

                    # Call the OpenAI Chat Completions API directly
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    content = response.choices[0].message.content

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

                # Extract prompt data based on API format
                if self.use_responses_api:
                    input_data = body.get("input", [])
                    prompt = input_data[-1].get("content", "") if input_data else ""
                    finish_reason = "completed"  # Responses API may use different finish reasons
                else:
                    messages = body.get("messages", [])
                    prompt = messages[-1].get("content", "") if messages else ""
                    finish_reason = response.choices[0].finish_reason

                results.append(
                    {
                        "task_id": task.get("custom_id"),
                        "prompt": prompt,
                        "response": content,
                        "model": model,
                        "finish_reason": finish_reason,
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
