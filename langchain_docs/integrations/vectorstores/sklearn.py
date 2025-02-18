#!/usr/bin/env python
# coding: utf-8

# # scikit-learn
#
# >[scikit-learn](https://scikit-learn.org/stable/) is an open-source collection of machine learning algorithms, including some implementations of the [k nearest neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). `SKLearnVectorStore` wraps this implementation and adds the possibility to persist the vector store in json, bson (binary json) or Apache Parquet format.
#
# This notebook shows how to use the `SKLearnVectorStore` vector database.
#
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

# In[1]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  scikit-learn")

# # if you plan to use bson serialization, install also:
get_ipython().run_line_magic("pip", "install --upgrade --quiet  bson")

# # if you plan to use parquet serialization, install also:
get_ipython().run_line_magic("pip", "install --upgrade --quiet  pandas pyarrow")


# To use OpenAI embeddings, you will need an OpenAI key. You can get one at https://platform.openai.com/account/api-keys or feel free to use any other embeddings.

# In[2]:


import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI key:")


# ## Basic usage
#
# ### Load a sample document corpus

# In[3]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()


# ### Create the SKLearnVectorStore, index the document corpus and run a sample query

# In[4]:


import tempfile

persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")

vector_store = SKLearnVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_path=persist_path,  # persist_path and serializer are optional
    serializer="parquet",
)

query = "What did the president say about Ketanji Brown Jackson"
docs = vector_store.similarity_search(query)
print(docs[0].page_content)


# ## Saving and loading a vector store

# In[5]:


vector_store.persist()
print("Vector store was persisted to", persist_path)


# In[6]:


vector_store2 = SKLearnVectorStore(
    embedding=embeddings, persist_path=persist_path, serializer="parquet"
)
print("A new instance of vector store was loaded from", persist_path)


# In[7]:


docs = vector_store2.similarity_search(query)
print(docs[0].page_content)


# ## Clean-up

# In[8]:


os.remove(persist_path)
