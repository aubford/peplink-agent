from pathlib import Path
import asyncio
from evals.ragas_eval import RagasEval


if __name__ == "__main__":
    ragas_eval = RagasEval(
        run_name=Path(__file__).parent.name,
        inference_llm="mini",
        eval_llm="mini",
        eval_boost_llm="mini",
        sample=(1, 2),
    )
    # asyncio.run(ragas_eval.generate_batchfiles())
    asyncio.run(ragas_eval.evaluate_rag())
