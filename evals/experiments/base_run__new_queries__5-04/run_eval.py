from pathlib import Path
import asyncio
from evals.ragas_eval import RagasEval

"""
Run evals on the same queries as "base_run_old_queries" except they have been transformed
 using the following LLM prompt: "Generate a semantically equivalent reformulation of the 
 provided query, explicitly removing its lexical overlap with the original source documents 
 while preserving all named entities exactly as they appear. Do not answer the query itself; 
 only produce the reformulated version."
"""


if __name__ == "__main__":
    ragas_eval = RagasEval(
        run_name=Path(__file__).parent.name,
        inference_llm="mini",
        eval_llm="mini",
        pinecone_index_name="pepwave-early-april-page-content-embedding",
    )
    # asyncio.run(ragas_eval.generate_batchfiles())
    asyncio.run(ragas_eval.evaluate_rag())
