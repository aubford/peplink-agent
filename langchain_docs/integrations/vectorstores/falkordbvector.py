#!/usr/bin/env python
# coding: utf-8

# # FalkorDBVectorStore
# <a href="https://docs.falkordb.com/" target="_blank">FalkorDB</a> is an open-source graph database with integrated support for vector similarity search
#
# it supports:
# - approximate nearest neighbor search
# - Euclidean similarity & Cosine Similarity
# - Hybrid search combining vector and keyword searches
#
# This notebook shows how to use the FalkorDB vector index (`FalkorDB`)
#
# See the <a href="https://docs.falkordb.com/" target="_blank">installation instruction</a>
#
#

# ## Setup

# In[ ]:


# Pip install necessary package
get_ipython().run_line_magic("pip", "install --upgrade  falkordb")
get_ipython().run_line_magic("pip", "install --upgrade  tiktoken")
get_ipython().run_line_magic(
    "pip", "install --upgrade  langchain langchain_huggingface"
)


# ### Credentials
# We want to use `HuggingFace` so we have to get the HuggingFace API Key

# In[1]:


import getpass
import os

if "HUGGINGFACE_API_KEY" not in os.environ:
    os.environ["HUGGINGFACE_API_KEY"] = getpass.getpass("HuggingFace API Key:")


# If you want to get automated tracing of your model calls you can also set your LangSmith API key by uncommenting below:

# In[2]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ## Initialization

# In[3]:


from langchain_community.vectorstores.falkordb_vector import FalkorDBVector
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# You can use FalkorDBVector locally with docker. See <a href="https://docs.falkordb.com/" target="_blank">installation instruction</a>

# In[4]:


host = "localhost"
port = 6379


# Or you can use FalkorDBVector with <a href="https://app.falkordb.cloud">FalkorDB Cloud</a>

# In[5]:


# E.g
# host = "r-6jissuruar.instance-zwb082gpf.hc-v8noonp0c.europe-west1.gcp.f2e0a955bb84.cloud"
# port = 62471
# username = "falkordb" # SET ON FALKORDB CLOUD
# password = "password" # SET ON FALKORDB CLOUD


# In[6]:


vector_store = FalkorDBVector(host=host, port=port, embedding=HuggingFaceEmbeddings())


# ## Manage vector store

# ### Add items to vector store

# In[7]:


from langchain_core.documents import Document

document_1 = Document(page_content="foo", metadata={"source": "https://example.com"})

document_2 = Document(page_content="bar", metadata={"source": "https://example.com"})

document_3 = Document(page_content="baz", metadata={"source": "https://example.com"})

documents = [document_1, document_2, document_3]

vector_store.add_documents(documents=documents, ids=["1", "2", "3"])


# ### Update items in vector store

# In[8]:


updated_document = Document(
    page_content="qux", metadata={"source": "https://another-example.com"}
)

vector_store.update_documents(document_id="1", document=updated_document)


# ### Delete items from vector store

# In[9]:


vector_store.delete(ids=["3"])


# ## Query vector store
#
# Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.

# ### Query directly
#
# Performing a simple similarity search can be done as follows:

# In[10]:


results = vector_store.similarity_search(
    query="thud", k=1, filter={"source": "https://another-example.com"}
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")


# If you want to execute a similarity search and receive the corresponding scores you can run:

# In[11]:


results = vector_store.similarity_search_with_score(query="bar")
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")


# ### Query by turning into retriever
# You can also transform the vector store into a retriever for easier usage in your chains.

# In[12]:


retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
retriever.invoke("thud")


# ## Usage for retrieval-augmented generation
# For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:
# - <a href="https://python.langchain.com/v0.2/docs/tutorials/#working-with-external-knowledge" target="_blank">Tutorials: working with external knowledge</a>
# - <a href="https://python.langchain.com/v0.2/docs/how_to/#qa-with-rag" target="_blank">How-to: Question and answer with RAG</a>
# - <a href="Retrieval conceptual docs" target="_blank">Retrieval conceptual docs</a>
#

# ## API reference
# For detailed documentation of all `FalkorDBVector` features and configurations head to the API reference: https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.falkordb_vector.FalkorDBVector.html
