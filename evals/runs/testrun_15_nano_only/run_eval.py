from pathlib import Path
import asyncio
from evals.ragas_eval import RagasEval


if __name__ == "__main__":
    ragas_eval = RagasEval(
        run_name=Path(__file__).parent.name,
        inference_llm="nano",
        eval_llm="nano",
        eval_boost_llm="nano",
        query_column="query_original",
        sample=15,
    )
    # asyncio.run(ragas_eval.generate_batchfiles())
    asyncio.run(ragas_eval.evaluate_rag())
