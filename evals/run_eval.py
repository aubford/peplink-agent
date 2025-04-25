from datetime import datetime
from pathlib import Path
import asyncio
from evals.ragas_eval import RagasEval

from dotenv import load_dotenv

load_dotenv()

GPT_4_1_MINI = "gpt-4.1-mini"
GPT_4_1_NANO = "gpt-4.1-nano"

date = datetime.now().strftime("%m-%d")

if __name__ == "__main__":
    evals_dir = Path(__file__).parent
    testset_name = ""
    ragas_eval = RagasEval(
        evals_dir,
        testset_name,
        inference_llm_model=GPT_4_1_MINI,
        eval_llm_model=GPT_4_1_NANO,
        eval_boost_llm_model=GPT_4_1_MINI,
        # sample=1,
        test_run=False,
        should_create_batch_job=True,
        run_name=f"init_base_run__4-21",
    )
    asyncio.run(ragas_eval.generate_batchfiles())
    # asyncio.run(ragas_eval.evaluate_rag())
