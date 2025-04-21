from pathlib import Path
import asyncio
from evals.ragas_eval import RagasEval


if __name__ == "__main__":
    evals_dir = Path(__file__).parent
    testset_name = "testset_100__april"
    eval = RagasEval(evals_dir, testset_name)
    # asyncio.run(eval.generate_batchfile())
    asyncio.run(eval.evaluate_rag())
