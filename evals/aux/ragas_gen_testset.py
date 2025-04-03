from datetime import datetime
from langsmith import tracing_context
from ragas.testset import TestsetGenerator, Testset
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
from ragas.testset.synthesizers.base import BaseSynthesizer
from evals.ragas_mocks import MockMultiHopAbstractQuerySynthesizer
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from util.util_main import to_serialized_parquet

load_dotenv()


LLM_MODEL = "gpt-4o"
TESTSET_SIZE = 10
evals_dir = Path(__file__).parent
kg_path = Path.joinpath(
    evals_dir,
    "output/kg_output__n_55__sample_with_custom_transforms__03_05_14_56.json",
)

kg = KnowledgeGraph.load(Path(kg_path))

# generator_llm = LangchainLLMWrapper(MockLLM()) # type: ignore
# generator_embeddings = LangchainEmbeddingsWrapper(MockEmbeddings())
# MockSynth = MockMultiHopAbstractQuerySynthesizer(llm=generator_llm)

generator_llm = LangchainLLMWrapper(ChatOpenAI(model=LLM_MODEL))
generator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-large")
)

network_persona = Persona(
    name="Technical Network Engineer",
    role_description="A professional specializing in network configurations and troubleshooting, particularly with experience in WAN setups and integrating satellite internet systems like Starlink with networking hardware such as Peplink Balance and Juniper SRX. This individual actively participates in technical forums to share knowledge, solve problems, and seek advice on complex network-related issues.",
)
full_time_rv_persona = Persona(
    name="Full-Time RV Owner",
    role_description="A full-time RV owner who is interested in using Pepwave devices to connect to the internet while traveling.",
)

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=kg,
    persona_list=[network_persona, full_time_rv_persona],
)

query_distribution: list[tuple[BaseSynthesizer, float]] = [
    # (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
    (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 1),
    # (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
]

# query_distribution: list[tuple[BaseSynthesizer, float]] = [
#     # (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1),
#     (MockSynth, 1),
#     # (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 1),
# ]

# Generate the dataset
dataset = generator.generate(
    testset_size=TESTSET_SIZE,
    query_distribution=query_distribution,
    with_debugging_logs=True,
    num_personas=1,
)

df = dataset.to_pandas()
current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
to_serialized_parquet(
    df, Path.joinpath(evals_dir, f"ragas_testset_{current_date}.parquet")
)
dataset.upload()
