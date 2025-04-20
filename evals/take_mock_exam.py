import pandas as pd
from pathlib import Path
import json
import re
import dotenv
from inference.rag_inference import RagInference


dotenv.load_dotenv()


class MockExam:
    def __init__(self):
        self.rag_inference = RagInference()
        mock_exam_path = Path(__file__).parent / "certified_engineer_mock_exam.json"
        with open(mock_exam_path, 'r') as f:
            self.mock_exam_questions = json.load(f)

    async def take_mock_exam(self) -> pd.DataFrame:
        """
        Run PCE mock exam questions through the RAG system and evaluate results.

        Args:
            questions: List of question dictionaries with fields:
                - question: The question text
                - choices: List of choice texts
                - correct_answers: List of indices of correct choices

        Returns:
            DataFrame with columns: question, answer, correct?, retrieved_contexts
        """
        # Create questions
        queries = {}
        for i, q in enumerate(self.mock_exam_questions):
            choices_text = "\n".join(
                [f"{i+1}. {choice}" for i, choice in enumerate(q["choices"])]
            )
            query = f"{q['question']}\n\nHere are your choices, respond with the number(s) of the correct choice(s):\n{choices_text}"
            queries[i] = query

        results = await self.rag_inference.batch_query_for_eval(queries)

        # Process results
        processed_results = []
        for i, (query, result) in enumerate(zip(queries, results)):
            # Extract answer numbers from the response
            answer_text = result.get("answer", "")
            # Look for numbers in the answer that match choices
            selected_choices = set()
            for match in re.finditer(r'\b([1-9])\b', answer_text):
                try:
                    selected = int(match.group(1))
                    if 1 <= selected <= len(q["choices"]):
                        selected_choices.add(selected)
                except ValueError:
                    continue

            # Check if answers are correct
            correct_answers = set(q["correct_answers"])
            is_correct = selected_choices == correct_answers

            processed_results.append(
                {
                    "question": q["question"],
                    "answer": answer_text,
                    "correct?": is_correct,
                    "retrieved_contexts": result.get("source_documents", []),
                }
            )

        return pd.DataFrame(processed_results)


def main():
    # Create exam taker
    exam_taker = MockExam()
    results_df = exam_taker.take_mock_exam()

    # Save results
    output_path = Path(__file__).parent / "pce_mock_exam_results.csv"
    results_df.to_csv(output_path, index=False)

    # Print summary
    correct_count = results_df["correct?"].sum()
    total_count = len(results_df)
    print(f"Exam Results: {correct_count}/{total_count} correct answers")
    print(f"Score: {(correct_count/total_count)*100:.2f}%")
    print(f"Detailed results saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    main()
