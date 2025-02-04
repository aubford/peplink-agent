# %% [markdown]
# # Using Ragas to Evaluate a RAG Application built with LangChain and LangGraph
#
# In the following notebook, we'll be looking at how [Ragas](https://github.com/explodinggradients/ragas) can be helpful in a number of ways when looking to evaluate your RAG applications!
#
# While this example is rooted in LangChain/LangGraph - Ragas is framework agnostic (you don't even need to be using a framework!).
#
# We'll:
#
# - Collect our data
# - Create a synthetic test set
# - Create a simple RAG application
# - Evaluate our RAG application
#
# But first! Let's set some dependencies!

# %% [markdown]
# ## Dependencies and API Keys:

# %%
!pip install -qU ragas==0.2.10


# %%
!pip install -qU langchain-community==0.3.14 langchain-openai==0.2.14 unstructured==0.16.12 langgraph==0.2.61 langchain-qdrant==0.2.0


# %% [markdown]
# We'll also need to provide our API keys.
#
# First, OpenAI's for our LLM/embedding model combination!

# %%
import os
from getpass import getpass
os.environ["OPENAI_API_KEY"] = getpass("Please enter your OpenAI API key!")


# %% [markdown]
# OPTIONALLY:
#
# We can also provide a Ragas API key - which you can sign-up for [here](https://app.ragas.io/).

# %%
os.environ["RAGAS_APP_TOKEN"] = getpass("Please enter your Ragas API key!")


# %% [markdown]
# ## Generating Synthetic Test Data
#
# We wil be using Ragas to build out a set of synthetic test questions, references, and reference contexts. This is useful because it will allow us to find out how our system is performing.
#
# > NOTE: Ragas is best suited for finding *directional* changes in your LLM-based systems. The absolute scores aren't comparable in a vacuum.

# %% [markdown]
# ### Data Preparation
#
# We'll prepare our data - and download our webpages which we'll be using for our data today.
#
# These webpages are from [Simon Willison's](https://simonwillison.net/) yearly "AI learnings".
#
# - [2023 Blog](https://simonwillison.net/2023/Dec/31/ai-in-2023/)
# - [2024 Blog](https://simonwillison.net/2024/Dec/31/llms-in-2024/)
#
# Let's start by collecting our data into a useful pile!

# %%
!mkdir data


# %%
!curl https://simonwillison.net/2023/Dec/31/ai-in-2023/ -o data/2023_llms.html


# %%
!curl https://simonwillison.net/2024/Dec/31/llms-in-2024/ -o data/2024_llms.html


# %% [markdown]
# Next, let's load our data into a familiar LangChain format using the `DirectoryLoader`.

# %%
from langchain_community.document_loaders import DirectoryLoader

path = "data/"
loader = DirectoryLoader(path, glob="*.html")
docs = loader.load()


# %% [markdown]
# ### Knowledge Graph Based Synthetic Generation
#
# Ragas uses a knowledge graph based approach to create data. This is extremely useful as it allows us to create complex queries rather simply. The additional testset complexity allows us to evaluate larger problems more effectively, as systems tend to be very strong on simple evaluation tasks.
#
# Let's start by defining our `generator_llm` (which will generate our questions, summaries, and more), and our `generator_embeddings` which will be useful in building our graph.

# %% [markdown]
# ### Unrolled SDG

# %%
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


# %% [markdown]
# Next, we're going to instantiate our Knowledge Graph.
#
# This graph will contain N number of nodes that have M number of relationships. These nodes and relationships (AKA "edges") will define our knowledge graph and be used later to construct relevant questions and responses.

# %%
from ragas.testset.graph import KnowledgeGraph

kg = KnowledgeGraph()
kg


# %% [markdown]
# The first step we're going to take is to simply insert each of our full documents into the graph. This will provide a base that we can apply transformations to.

# %%
from ragas.testset.graph import Node, NodeType

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )
kg


# %% [markdown]
# Now, we'll apply the *default* transformations to our knowledge graph. This will take the nodes currently on the graph and transform them based on a set of [default transformations](https://docs.ragas.io/en/latest/references/transforms/#ragas.testset.transforms.default_transforms).
#
# These default transformations are dependent on the corpus length, in our case:
#
# - Producing Summaries -> produces summaries of the documents
# - Extracting Headlines -> finding the overall headline for the document
# - Theme Extractor -> extracts broad themes about the documents
#
# It then uses cosine-similarity and heuristics between the embeddings of the above transformations to construct relationships between the nodes.

# %%
from ragas.testset.transforms import default_transforms, apply_transforms

transformer_llm = generator_llm
embedding_model = generator_embeddings

default_transforms = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)
apply_transforms(kg, default_transforms)
kg


# %% [markdown]
# We can save and load our knowledge graphs as follows.

# %%
kg.save("ai_across_years_kg.json")
ai_across_years_kg = KnowledgeGraph.load("ai_across_years_kg.json")
ai_across_years_kg


# %% [markdown]
# Using our knowledge graph, we can construct a "test set generator" - which will allow us to create queries.

# %%
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model, knowledge_graph=ai_across_years_kg)


# %% [markdown]
# However, we'd like to be able to define the kinds of queries we're generating - which is made simple by Ragas having pre-created a number of different "QuerySynthesizer"s.
#
# Each of these Synthetsizers is going to tackle a separate kind of query which will be generated from a scenario and a persona.
#
# In essence, Ragas will use an LLM to generate a persona of someone who would interact with the data - and then use a scenario to construct a question from that data and persona.

# %%
from ragas.testset.synthesizers import default_query_distribution, SingleHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer

query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
]


# %% [markdown]
# Finally, we can use our `TestSetGenerator` to generate our testset!

# %%
testset = generator.generate(testset_size=10, query_distribution=query_distribution)
testset.to_pandas()


# %% [markdown]
# ### Abstracted SDG
#
# The above method is the full process - but we can shortcut that using the provided abstractions!
#
# This will generate our knowledge graph under the hood, and will - from there - generate our personas and scenarios to construct our queries.
#
#

# %%
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)


# %%
dataset.to_pandas()


# %% [markdown]
# #### OPTIONAL:
#
# If you've provided your Ragas API key - you can use this web interface to look at the created data!

# %%
dataset.upload()


# %% [markdown]
# ## LangChain RAG
#
# Now we'll construct our LangChain RAG, which we will be evaluating using the above created test data!

# %% [markdown]
# ### R - Retrieval
#
# Let's start with building our retrieval pipeline, which will involve loading the same data we used to create our synthetic test set above.
#
# > NOTE: We need to use the same data - as our test set is specifically designed for this data.

# %%
path = "data/"
loader = DirectoryLoader(path, glob="*.html")
docs = loader.load()


# %% [markdown]
# Now that we have our data loaded, let's split it into chunks!

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(docs)
len(split_documents)


# %% [markdown]
# Next up, we'll need to provide an embedding model that we can use to construct our vector store.

# %%
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# %% [markdown]
# Now we can build our in memory QDrant vector store.

# %%
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="ai_across_years",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="ai_across_years",
    embedding=embeddings,
)


# %% [markdown]
# We can now add our documents to our vector store.

# %%
vector_store.add_documents(documents=split_documents)


# %% [markdown]
# Let's define our retriever.

# %%
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# %% [markdown]
# Now we can produce a node for retrieval!

# %%
def retrieve(state):
  retrieved_docs = retriever.invoke(state["question"])
  return {"context" : retrieved_docs}


# %% [markdown]
# ### Augmented
#
# Let's create a simple RAG prompt!

# %%
from langchain.prompts import ChatPromptTemplate

RAG_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)


# %% [markdown]
# ### Generation
#
# We'll also need an LLM to generate responses - we'll use `gpt-4o-mini` to avoid using the same model as our judge model.

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")


# %% [markdown]
# Then we can create a `generate` node!

# %%
def generate(state):
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
  response = llm.invoke(messages)
  return {"response" : response.content}


# %% [markdown]
# ### Building RAG Graph with LangGraph
#
# Let's create some state for our LangGraph RAG graph!

# %%
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

class State(TypedDict):
  question: str
  context: List[Document]
  response: str


# %% [markdown]
# Now we can build our simple graph!
#
# > NOTE: We're using `add_sequence` since we will always move from retrieval to generation. This is essentially building a chain in LangGraph.

# %%
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# %% [markdown]
# Let's do a test to make sure it's doing what we'd expect.

# %%
response = graph.invoke({"question" : "How are LLM agents useful?"})


# %%
response["response"]


# %% [markdown]
# ## Evaluating the App with Ragas
#
# Now we can finally do our evaluation!
#
# We'll start by running the queries we generated usign SDG above through our application to get context and responses.

# %%
for test_row in dataset:
  response = graph.invoke({"question" : test_row.eval_sample.user_input})
  test_row.eval_sample.response = response["response"]
  test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]


# %%
dataset.to_pandas()


# %% [markdown]
# Then we can convert that table into a `EvaluationDataset` which will make the process of evaluation smoother.

# %%
from ragas import EvaluationDataset

evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())


# %% [markdown]
# We'll need to select a judge model - in this case we're using the same model that was used to generate our Synthetic Data.

# %%
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))


# %% [markdown]
# Next up - we simply evaluate on our desired metrics!

# %%
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
    llm=evaluator_llm
)
result


# %% [markdown]
# ## Making Adjustments and Re-Evaluating
#
# Now that we've got our baseline - let's make a change and see how the model improves or doesn't improve!
#
# > NOTE: This will be using Cohere's Rerank model - please be sure to sign-up for an API key!

# %%
os.environ["COHERE_API_KEY"] = getpass("Please enter your Cohere API key!")


# %%
!pip install -qU cohere langchain_cohere


# %%
retriever = vector_store.as_retriever(search_kwargs={"k": 20})


# %%
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

def retrieve_adjusted(state):
  compressor = CohereRerank(model="rerank-english-v3.0")
  compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever, search_kwargs={"k": 5}
  )
  retrieved_docs = compression_retriever.invoke(state["question"])
  return {"context" : retrieved_docs}


# %%
class State(TypedDict):
  question: str
  context: List[Document]
  response: str

graph_builder = StateGraph(State).add_sequence([retrieve_adjusted, generate])
graph_builder.add_edge(START, "retrieve_adjusted")
graph = graph_builder.compile()


# %%
response = graph.invoke({"question" : "How are LLM agents useful?"})
response["response"]


# %%
for test_row in dataset:
  response = graph.invoke({"question" : test_row.eval_sample.user_input})
  test_row.eval_sample.response = response["response"]
  test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]


# %%
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
    llm=evaluator_llm
)
result
