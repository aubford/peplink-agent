from pydantic import BaseModel
from load.base_load import BaseLoad
import json
from openai import OpenAI

# noinspection PyProtectedMember
from openai.lib._parsing._completions import type_to_response_format_param
from typing import List, Dict, Any, Literal
import time
import os
import pandas as pd
from pathlib import Path


class RedditData(BaseModel):
    id: str
    type: str
    primary_content: str
    title: str
    lead_content: str
    author_id: str
    author_name: str
    page_content: str
    score: int

    post_title: str
    post_content: str
    comment_content: str

    source_file: str
    subject_matter: str
    subreddit: str
    category: str
    url: str
    post_score: int
    comment_date: int
    post_id: str
    post_author_name: str
    post_author_id: str
    post_author_is_mod: bool
    post_author_is_gold: bool
    post_author_is_blocked: bool
    post_author_total_karma: str
    post_author_comment_karma: str
    post_author_link_karma: str
    post_author_verified: str
    post_author_has_verified_email: str
    post_author_has_subscribed: str
    post_author_is_employee: str
    comment_author_name: str
    comment_author_id: str
    comment_author_is_mod: bool
    comment_author_is_gold: bool
    comment_author_is_blocked: bool
    comment_author_total_karma: int
    comment_author_comment_karma: int
    comment_author_link_karma: int
    comment_author_verified: bool
    comment_author_has_verified_email: bool
    comment_author_has_subscribed: bool
    comment_author_is_employee: bool


class ModelResponse(BaseModel):
    summary: str  # summary of post information
    is_informative: bool  # if false we will filter it out
    themes: list[str]  # themes of the post


class ForumSyntheticDataPrompt:
    """
    Class to generate prompts for OpenAI API to extract structured data from forum posts.
    The prompt facilitates transformation of a document's primary_content and lead_content
    into a structured format for analysis.
    """

    @staticmethod
    def create_system_prompt() -> str:
        """Creates the system prompt that instructs the model on its task."""
        return """You are an expert technical content analyzer specializing in IT networking and Pepwave products.
Your task is to analyze forum conversations and extract key information.

You will be provided with a forum conversation consisting of:
1. The original forum post/question.
2. A response to the original post.

Together, these form a single conversation turn between two forum users.

Analyze this conversation and provide a structured output containing:
1. A list of technical themes discussed.
2. A summary of the technical information that can be learned from the conversation. Provide this in the form of sentences, not lists or other special formatting.
3. An assessment of whether there is useful technical information related to Pepwave products or IT networking.

Important guidelines:
- Focus only on technical content and information in your analysis.
- "Useful technical information" means factual statements, not questions asking for information.
- Be specific and precise in identifying themes.
- Provide a concise, factual summary that captures the key technical points.
- Base your analysis only on the provided content, do not make assumptions.
"""

    @staticmethod
    def create_prompt(lead_content: str, primary_content: str) -> str:
        """Creates a prompt that includes the forum post content."""

        return f"""# Forum Post (Original Post/Question)
{lead_content}

# Response to the Original Post
{primary_content}

Analyze this conversation according to the guidelines provided.
"""

    @staticmethod
    def get_examples() -> list[dict]:
        """
        Provides examples of conversations and ideal responses.

        Returns:
            List of example dictionaries.
        """
        return [
            {
                "lead_content": "I just installed a Pepwave MAX Transit Duo-CAT12 in my RV, but I'm having trouble with the cellular connection. The signal strength is showing only 2 bars even though my phone gets 4 bars in the same location. Does anyone know why this might be happening?",
                "primary_content": "Check your antenna connections first. The MAX Transit Duo requires proper external antennas to get the best signal. Make sure you're using the right cellular antennas and they're properly connected to the correct ports (they're labeled CELL on the router). Also, try changing the SIM priority in the admin panel - go to Network > Mobile > Settings, and you can change which SIM card is used or enable band locking for better performance on specific carriers. If you're in a fringe area, enabling band locking to the lower frequencies (like Band 12, 13, or 71 depending on your carrier) might help with penetration and range.",
                "expected_output": {
                    "themes": [
                        "Cellular signal strength",
                        "Antenna configuration",
                        "Pepwave MAX Transit Duo",
                        "SIM card settings",
                        "Band locking",
                        "RV networking",
                    ],
                    "summary": "The conversation discusses troubleshooting poor cellular signal on a Pepwave MAX Transit Duo-CAT12 installed in an RV. The response suggests checking antenna connections to the correct ports, adjusting SIM priority in the admin panel, and enabling band locking for specific carriers, particularly lower frequency bands for better range in fringe areas.",
                    "is_useful": True,
                },
            },
            {
                "lead_content": "Anyone have recommendations for a good backup internet solution? I work from home and need something reliable when my main fiber connection goes down.",
                "primary_content": "I've been there! After trying several options, I settled on a Peplink Balance 20X with a 5G capable modem. The SpeedFusion technology in the Peplink devices is amazing for combining connections. I use it with both my fixed connection and a cellular backup, and the handover between them is completely seamless. I can be on a Zoom call and if my main connection fails, the call doesn't drop at all because of the Hot Failover feature. It's not cheap but worth every penny for reliability.",
                "expected_output": {
                    "themes": [
                        "Backup internet solutions",
                        "Peplink Balance 20X",
                        "SpeedFusion technology",
                        "Connection bonding",
                        "Hot Failover",
                        "Work from home setup",
                    ],
                    "summary": "The conversation covers backup internet solutions for remote work. The response recommends the Peplink Balance 20X with a 5G modem, highlighting its SpeedFusion technology for combining connections and Hot Failover feature that provides seamless transition between primary and backup connections, maintaining continuity for applications like video calls.",
                    "is_useful": True,
                },
            },
            {
                "lead_content": "Does anyone know if Peplink routers work with AT&T FirstNet? I'm looking to set up a mobile command center for our emergency response team.",
                "primary_content": "No idea, I've never used FirstNet. Have you tried contacting Peplink support directly? They might have better information about carrier compatibility.",
                "expected_output": {
                    "themes": [
                        "FirstNet compatibility",
                        "Peplink routers",
                        "Emergency response equipment",
                    ],
                    "summary": "The conversation inquires about compatibility between Peplink routers and AT&T FirstNet for an emergency response mobile command center. The response does not provide any specific technical information about compatibility, only suggesting to contact Peplink support directly.",
                    "is_useful": False,
                },
            },
        ]

    @staticmethod
    def create_system_prompt_with_examples(num_examples: int = 2) -> str:
        """
        Creates a system prompt that includes both instructions and examples.

        Args:
            num_examples: Number of examples to include (default: 2)

        Returns:
            Complete system prompt with instructions and examples
        """
        base_prompt = ForumSyntheticDataPrompt.create_system_prompt()
        examples = ForumSyntheticDataPrompt.get_examples()

        # Limit to the requested number of examples
        examples = examples[: min(num_examples, len(examples))]

        examples_text = (
            "\n\nHere are some examples of conversations and expected analyses:\n\n"
        )

        for i, example in enumerate(examples, 1):
            # Format the example conversation
            example_prompt = ForumSyntheticDataPrompt.create_prompt(
                lead_content=example["lead_content"],
                primary_content=example["primary_content"],
            )

            # Format the expected output
            output = example["expected_output"]
            themes_str = ", ".join([f'"{theme}"' for theme in output["themes"]])

            expected_output = (
                f'Expected output:\n'
                f'{{\n'
                f'  "themes": [{themes_str}],\n'
                f'  "summary": "{output["summary"]}",\n'
                f'  "is_useful": {output["is_useful"]}\n'
                f'}}\n'
            )

            examples_text += (
                f"EXAMPLE {i}:\n\n{example_prompt}\n\n{expected_output}\n{'=' * 40}\n\n"
            )

        return base_prompt + examples_text


class RedditLoad(BaseLoad):
    folder_name = "reddit"

    def __init__(self):
        super().__init__()
        self.batch_manager = BatchManager(self.staging_folder)
        self.nlp = self._init_spacy()

    def create_merged_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets into a single dataframe and perform operations."""
        df = pd.concat(dfs)
        df = self.ner(df)
        return df

    def extract_entities(self, text: str) -> str:
        """Use SpaCy NER to extract entities from text, including custom PEPWAVE_SETTINGS_ENTITY."""
        if not text or pd.isna(text):
            return ""

        # Process the text with the pipeline that includes our EntityRuler
        doc = self.nlp(text)
        entities: set[str] = {
            ent.text
            for ent in doc.ents
            if ent.label_
            in {
                "URL",
                "FAC",
                "ORG",
                "GPE",
                "PRODUCT",
                "LOC",
                "WORK_OF_ART",
                "EVENT",
                "PEPWAVE_SETTINGS_ENTITY",  # Our custom entity type
            }
            and any(c.isalpha() for c in ent.text)
        }

        return ", ".join(entities) if entities else ""

    def ner(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use SpaCy NER feature to extract an "entities" column from the text in the primary_content and lead_content columns.
        The "entities" column contains a string that is a list of entities separated by commas.
        """

        # Process both columns and combine the entities
        df["primary_entities"] = df["comment_content"].apply(self.extract_entities)
        df["lead_entities"] = df["post_content"].apply(self.extract_entities)

        # Combine entities from both columns, removing duplicates
        def combine_entities(row):
            all_entities = []
            if row["primary_entities"]:
                all_entities.extend(row["primary_entities"].split(", "))
            if row["lead_entities"]:
                all_entities.extend(row["lead_entities"].split(", "))

            unique_entities = set(entity for entity in all_entities if entity)
            return ", ".join(unique_entities)

        df["entities"] = df.apply(combine_entities, axis=1)

        # Drop intermediate columns
        df = df.drop(columns=["primary_entities", "lead_entities"])

        return df


ValidEndpoints = Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"]


class BatchManager:
    """
    Handles batch processing functionality using OpenAI's Batch API.

    This class provides methods to create, monitor, and retrieve results from batch jobs.
    Batch jobs are processed asynchronously by OpenAI and completed within 24 hours.
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

        # Ensure base_path is a Path object and exists
        base_path.mkdir(parents=True, exist_ok=True)
        self.file_name = base_path / "batchfile.jsonl"
        self.output_file_name = base_path / "batch_results.jsonl"
        self.status_file_name = base_path / "batch_status.json"
        self.endpoint: ValidEndpoints = endpoint
        self.batch_id = None

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

        self.create_batch_file(tasks)
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
        schema: type[BaseModel],
        system_prompt: str,
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
