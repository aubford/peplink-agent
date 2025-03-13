from langsmith import tracing_context
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.graph import KnowledgeGraph
from ragas.local_utils import MockLLM, MockEmbeddings
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.persona import Persona
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


LLM_MODEL = "gpt-4o"
TESTSET_SIZE = 1
evals_dir = Path(__file__).parent
latest_kg_path = Path.joinpath(
    evals_dir,
    "output/kg_output__n_55__sample_with_custom_transforms__03_05_14_56.json",
)

kg = KnowledgeGraph.load(Path(latest_kg_path))


generator_llm = LangchainLLMWrapper(ChatOpenAI(model_name=LLM_MODEL))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

persona = Persona(
    name="Technical Network Engineer",
    role_description="A professional specializing in network configurations and troubleshooting, particularly with experience in WAN setups and integrating satellite internet systems like Starlink with networking hardware such as Peplink Balance and Juniper SRX. This individual actively participates in technical forums to share knowledge, solve problems, and seek advice on complex network-related issues.",
)

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=kg,
    persona_list=[persona],
)

query_distribution = [
    # (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1),
    (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 1),
    # (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 1),
]

# Generate the dataset
dataset = generator.generate(
    testset_size=TESTSET_SIZE,
    query_distribution=query_distribution,
    with_debugging_logs=True,
    num_personas=1,
)

# Clean the dataset by ensuring all text fields are ASCII-compatible
# This is a simple approach that replaces problematic characters
for sample in dataset.samples:
    # Clean the eval_sample fields that might contain problematic Unicode
    if hasattr(sample.eval_sample, "question"):
        sample.eval_sample.question = sample.eval_sample.question.encode(
            "ascii", "replace"
        ).decode("ascii")
    if hasattr(sample.eval_sample, "ground_truth"):
        sample.eval_sample.ground_truth = sample.eval_sample.ground_truth.encode(
            "ascii", "replace"
        ).decode("ascii")
    if hasattr(sample.eval_sample, "contexts") and sample.eval_sample.contexts:
        sample.eval_sample.contexts = [
            ctx.encode("ascii", "replace").decode("ascii")
            for ctx in sample.eval_sample.contexts
        ]

# upload to ragas app
dataset.upload()
df = dataset.to_pandas()
df.to_parquet(Path.joinpath(evals_dir, "ragas_testset.parquet"))
