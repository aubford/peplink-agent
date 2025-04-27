from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric
from pydantic import BaseModel, Field
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import Callbacks
from openai import AsyncOpenAI
from evals.prompts.prompts import load_prompts

oa_client = AsyncOpenAI()
PROMPTS = load_prompts()


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
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        api_response = await oa_client.responses.parse(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=PROMPTS["ragas_eval/web_validation"].format(
                question=sample.user_input, answer=sample.response
            ),
            text_format=WebValidationInput,
        )

        parsed: WebValidationInput = api_response.output_parsed
        some_claims_incorrect = parsed.some_claims_incorrect
        incorrect_claims = parsed.incorrect_claims

        # ensure the model is behaving consistently with itself
        if some_claims_incorrect:
            assert len(incorrect_claims) > 0

        if not some_claims_incorrect:
            assert len(incorrect_claims) == 0

        return incorrect_claims
