# %%
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)

LLM_MODEL = "gpt-4o"
TESTSET_SIZE = 2

kg = KnowledgeGraph.load("evals/staged_knowledge_graph.json")

generator_llm = LangchainLLMWrapper(ChatOpenAI(model_name=LLM_MODEL))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

generator = TestsetGenerator(
    llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg
)

query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
]

dataset = generator.generate(
    testset_size=TESTSET_SIZE,
    query_distribution=query_distribution,
    with_debugging_logs=True,
)

# upload to ragas app
dataset.upload()
df = dataset.to_pandas()
df.to_csv("evals/ragas_testset.csv", index=False)
