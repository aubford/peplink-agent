#!/usr/bin/env python
# coding: utf-8

# # Google Cloud Vertex AI Reranker
# 
# > The [Vertex Search Ranking API](https://cloud.google.com/generative-ai-app-builder/docs/ranking) is one of the standalone APIs in [Vertex AI Agent Builder](https://cloud.google.com/generative-ai-app-builder/docs/builder-apis). It takes a list of documents and reranks those documents based on how relevant the documents are to a query. Compared to embeddings, which look only at the semantic similarity of a document and a query, the ranking API can give you precise scores for how well a document answers a given query. The ranking API can be used to improve the quality of search results after retrieving an initial set of candidate documents.
# 
# >The ranking API is stateless so there's no need to index documents before calling the API. All you need to do is pass in the query and documents. This makes the API well suited for reranking documents from any document retrievers.
# 
# >For more information, see [Rank and rerank documents](https://cloud.google.com/generative-ai-app-builder/docs/ranking).

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain langchain-community langchain-google-community langchain-google-community[vertexaisearch] langchain-google-vertexai langchain-chroma langchain-text-splitters beautifulsoup4')


# ### Setup

# In[5]:


PROJECT_ID = ""
REGION = ""
RANKING_LOCATION_ID = "global"  # @param {type:"string"}

# Initialize GCP project for Vertex AI
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)


# ### Load and Prepare data
# 
# For this example, we will be using the [Google Wiki page](https://en.wikipedia.org/wiki/Google)to demonstrate how the Vertex Ranking API works.
# 
# We use a standard pipeline of `load -> split -> embed data`.
# 
# The embeddings are created using the [Vertex Embeddings API](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#supported_models) model - `textembedding-gecko@003`

# In[6]:


from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

vectordb = None

# Load wiki page
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Google")
data = loader.load()

# Split doc into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=5)
splits = text_splitter.split_documents(data)

print(f"Your {len(data)} documents have been split into {len(splits)} chunks")

if vectordb is not None:  # delete existing vectordb if it already exists
    vectordb.delete_collection()

embedding = VertexAIEmbeddings(model_name="textembedding-gecko@003")
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)


# In[7]:


import pandas as pd
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_google_community.vertex_rank import VertexAIRank

# Instantiate the VertexAIReranker with the SDK manager
reranker = VertexAIRank(
    project_id=PROJECT_ID,
    location_id=RANKING_LOCATION_ID,
    ranking_config="default_ranking_config",
    title_field="source",
    top_n=5,
)

basic_retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # fetch top 5 documents

# Create the ContextualCompressionRetriever with the VertexAIRanker as a Reranker
retriever_with_reranker = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=basic_retriever
)


# ### Testing out the Vertex Ranking API
# 
# Let's query both the `basic_retriever` and `retriever_with_reranker` with the same query and compare the retrieved documents.
# 
# The Ranking API takes in the input from the `basic_retriever` and passes it to the Ranking API.
# 
# The ranking API is used to improve the quality of the ranking and determine a score that indicates the relevance of each record to the query.
# 
# You can see the difference between the Unranked and the Ranked Documents. The Ranking API moves the most semantically relevant documents to the top of the context window of the LLM thus helping it form a better answer with reasoning.

# In[8]:


import pandas as pd

# Use the basic_retriever and the retriever_with_reranker to get relevant documents
query = "how did the name google originate?"
retrieved_docs = basic_retriever.invoke(query)
reranked_docs = retriever_with_reranker.invoke(query)

# Create two lists of results for unranked and ranked docs
unranked_docs_content = [docs.page_content for docs in retrieved_docs]
ranked_docs_content = [docs.page_content for docs in reranked_docs]

# Create a comparison DataFrame using the padded lists
comparison_df = pd.DataFrame(
    {
        "Unranked Documents": unranked_docs_content,
        "Ranked Documents": ranked_docs_content,
    }
)

comparison_df


# Let's inspect a couple of reranked documents. We observe that the retriever still returns the relevant Langchain type [documents](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) but as part of the metadata field, we also recieve the `relevance_score` from the Ranking API.

# In[10]:


for i in range(2):
    print(f"Document {i}")
    print(reranked_docs[i])
    print("----------------------------------------------------------\n")


# ### Putting it all together
# 
# This shows an example of a complete RAG chain with a simple prompt template on how you can perform reranking using the Vertex Ranking API.
# 
# 

# In[15]:


from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_vertexai import VertexAI

llm = VertexAI(model_name="gemini-1.0-pro-002")

# Instantiate the VertexAIReranker with the SDK manager
reranker = VertexAIRank(
    project_id=PROJECT_ID,
    location_id=RANKING_LOCATION_ID,
    ranking_config="default_ranking_config",
    title_field="source",  # metadata field key from your existing documents
    top_n=5,
)

# value of k can be set to a higher value as well for tweaking performance
# eg: # of docs: basic_retriever(100) -> reranker(5)
basic_retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # fetch top 5 documents

# Create the ContextualCompressionRetriever with the VertexAIRanker as a Reranker
retriever_with_reranker = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=basic_retriever
)

template = """
<context>
{context}
</context>

Question:
{query}

Don't give information outside the context or repeat your findings.
Answer:
"""
prompt = PromptTemplate.from_template(template)

reranker_setup_and_retrieval = RunnableParallel(
    {"context": retriever_with_reranker, "query": RunnablePassthrough()}
)

chain = reranker_setup_and_retrieval | prompt | llm


# In[12]:


query = "how did the name google originate?"


# In[24]:


chain.invoke(query)

