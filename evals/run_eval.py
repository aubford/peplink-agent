from pathlib import Path
import asyncio
from evals.ragas_eval import RagasEval

from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    evals_dir = Path(__file__).parent
    testset_name = "testset_100__april"
    ragas_eval = RagasEval(
        evals_dir, testset_name, test_run=True, sample=1, create_batch_job=False
    )
    # asyncio.run(eval.generate_batchfile())
    asyncio.run(ragas_eval.evaluate_rag())
