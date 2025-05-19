from pathlib import Path
import asyncio
from evals.ragas_eval import RagasEval
from inference.rag_inference import RagInference
from util.util_main import models

# from langchain.globals import set_debug
# set_debug(True)

if __name__ == "__main__":
    inference = RagInference(
        llm_model=models['mini'],
        pinecone_index_name="pepwave-early-april-technical-summary-embeddi",
    )
    ragas_eval = RagasEval(
        inference=inference,
        run_name=Path(__file__).parent.name,
        eval_llm=models['mini'],
        sample=(0, 2),
        should_create_batch_job=True,
    )
    # asyncio.run(ragas_eval.generate_batchfiles())
    asyncio.run(ragas_eval.evaluate_rag())
