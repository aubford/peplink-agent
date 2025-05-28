from pathlib import Path
import asyncio
from evals.ragas_eval import RagasEval

# from langchain.globals import set_debug
# set_debug(True)

if __name__ == "__main__":
    ragas_eval = RagasEval(
        run_name=Path(__file__).parent.name,
        inference_llm="mini",
        eval_llm="mini",
        pinecone_index_name="pepwave-early-april-technical-summary-embeddi",
    )
    # asyncio.run(ragas_eval.generate_batchfiles())
    asyncio.run(ragas_eval.evaluate_rag())
