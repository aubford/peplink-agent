import json
from pathlib import Path
from pydantic import BaseModel, Field
from load.batch_manager import BatchManager
from batch_llm import BatchChatOpenAI
from inference.rag_inference import RagInference
from evals.prompts.prompts import load_prompts


class MockExamOutput(BaseModel):
    correct_choices: list[str] = Field(
        description="An array containing the exact text for each correct choice"
    )


PROMPTS = load_prompts()


class MockExam:
    def __init__(
        self,
        evals_dir: Path,
        testset_name: str,
        llm_model: str,
        run_name: str | None = None,
        should_create_batch_job: bool = True,
        sample: bool | int = False,
    ):
        self.should_create_batch_job = should_create_batch_job
        self.batch_manager = BatchManager(
            base_path=evals_dir / "batches",
            endpoint="/v1/chat/completions",
            batch_name=f"{testset_name}__mock_exam__{run_name or 'default_batch'}",
            schema=MockExamOutput,
        )
        self.rag_inference = RagInference(
            llm_model=llm_model,
            eval_llm=BatchChatOpenAI(
                model=llm_model,
                temperature=0,
                batch_manager=self.batch_manager,
            ),
        )
        filename = (
            "certified_engineer_exam_sample.json"
            if sample
            else "certified_engineer_exam.json"
        )
        self.mock_exam_path = evals_dir / "testsets" / filename
        with open(self.mock_exam_path, "r") as f:
            self.mock_exam_questions = json.load(f)

    async def generate_batchfile(self) -> None:
        """
        Run PCE mock exam questions through the RAG system and evaluate results.

        Returns:
            DataFrame with columns: question, answer, correct?, retrieved_contexts
        """
        # Prepare queries (duplicate each question)
        queries = {}
        for q in self.mock_exam_questions:
            choices_text = "\n".join(
                [f"{j+1}. {choice}" for j, choice in enumerate(q["choices"])]
            )
            query = PROMPTS["ragas_eval/mock_exam_query"].format(
                question=q["question"], choices=choices_text
            )
            queries[q["question_id"]] = query

        if self.should_create_batch_job:
            self.batch_manager.clear_batch_files()
        results = await self.rag_inference.batch_query_for_eval(queries)
        for q in self.mock_exam_questions:
            q["custom_id"] = results[q["question_id"]]["answer"]

        with open(self.mock_exam_path, "w") as f:
            json.dump(self.mock_exam_questions, f)

        if self.should_create_batch_job:
            self.batch_manager.create_batch_job()

    def get_results(self) -> tuple[str, list[dict[str, str]]]:
        batch_results = self.batch_manager.get_content_if_ready()

        for q in self.mock_exam_questions:
            custom_id = q["custom_id"]
            q["given_answer"] = json.loads(batch_results[custom_id])["correct_choices"]

        correct_count = 0.0
        missed_questions = []
        for q in self.mock_exam_questions:
            given = set(q["given_answer"])
            correct = set(q["correct_answers"])
            if given == correct:
                correct_count += 1
            else:
                if given.issubset(correct) and len(given) > len(correct) / 2:
                    correct_count += 0.5
                missed_questions.append(q)
        # save missed questions
        missed_path = (
            self.mock_exam_path.parent / f"{self.mock_exam_path.stem}_missed.json"
        )
        with open(missed_path, "w") as f:
            json.dump(missed_questions, f)
        return f"{correct_count}/{len(self.mock_exam_questions)}", missed_questions
