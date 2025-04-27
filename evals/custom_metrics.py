from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric
from pydantic import BaseModel, Field
from ragas.dataset_schema import SingleTurnSample
from openai import AsyncOpenAI
from evals.prompts.prompts import load_prompts
from config.logger import RotatingFileLogWriter
from typing import Any

oa_client = AsyncOpenAI()
PROMPTS = load_prompts()


logger = RotatingFileLogWriter(name="web_validation", backup_count=5)

"""not working, too expensive and unreliable"""


class WebValidationInput(BaseModel):
    some_claims_incorrect: bool = Field(
        description="Whether any of the claims made in the response are incorrect"
    )
    incorrect_claims: list[str] = Field(
        description="List of any incorrect statements in the response"
    )


class WebValidation(MetricWithLLM, SingleTurnMetric):
    name = "web_validation"

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Any
    ) -> int:
        api_response = await oa_client.responses.parse(
            model="gpt-4.1",
            temperature=0,
            tools=[{"type": "web_search_preview"}],
            tool_choice="required",
            input=PROMPTS["ragas_eval/web_validation"].format(
                question=sample.user_input, answer=sample.response
            ),
            text_format=WebValidationInput,
        )

        parsed: WebValidationInput = api_response.output_parsed  # type: ignore
        some_claims_incorrect = parsed.some_claims_incorrect
        incorrect_claims = parsed.incorrect_claims

        # ensure the model is behaving consistently with itself
        if some_claims_incorrect:
            assert len(incorrect_claims) > 0
            logger.log_and_print_header("Web Validator - Incorrect Claims")
            logger.log_and_print(f"Question:\n {sample.user_input}")
            logger.log_and_print(f"Incorrect_claims:\n {incorrect_claims}")
            logger.log_and_print("-" * 10)

        if not some_claims_incorrect:
            assert len(incorrect_claims) == 0

        return 0 if some_claims_incorrect else 1
