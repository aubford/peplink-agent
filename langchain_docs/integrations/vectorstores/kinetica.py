#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Kinetica
---
# # Kinetica Vectorstore API
# 
# >[Kinetica](https://www.kinetica.com/) is a database with integrated support for vector similarity search
# 
# It supports:
# - exact and approximate nearest neighbor search
# - L2 distance, inner product, and cosine distance
# 
# This notebook shows how to use the Kinetica vector store (`Kinetica`).

# This needs an instance of Kinetica which can easily be setup using the instructions given here - [installation instruction](https://www.kinetica.com/developer-edition/).

# In[ ]:


# Pip install necessary package
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-openai langchain-community')
get_ipython().run_line_magic('pip', 'install gpudb>=7.2.2.0')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  tiktoken')


# We want to use `OpenAIEmbeddings` so we have to get the OpenAI API Key.

# In[2]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[3]:


## Loading Environment Variables
from dotenv import load_dotenv

load_dotenv()


# In[ ]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import (
    Kinetica,
    KineticaSettings,
)
from langchain_openai import OpenAIEmbeddings


# In[5]:


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# In[ ]:


# Kinetica needs the connection to the database.
# This is how to set it up.
HOST = os.getenv("KINETICA_HOST", "http://127.0.0.1:9191")
USERNAME = os.getenv("KINETICA_USERNAME", "")
PASSWORD = os.getenv("KINETICA_PASSWORD", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def create_config() -> KineticaSettings:
    return KineticaSettings(host=HOST, username=USERNAME, password=PASSWORD)


# In[7]:


from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]


# ## Similarity Search with Euclidean Distance (Default)

# In[8]:


# The Kinetica Module will try to create a table with the name of the collection.
# So, make sure that the collection name is unique and the user has the permission to create a table.

COLLECTION_NAME = "langchain_example"
connection = create_config()

db = Kinetica(
    connection,
    embeddings,
    collection_name=COLLECTION_NAME,
)

db.add_documents(documents=documents, ids=uuids)


# In[9]:


# query = "What did the president say about Ketanji Brown Jackson"
# docs_with_score = db.similarity_search_with_score(query)


# In[10]:


print()
print("Similarity Search")
results = db.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

print()
print("Similarity search with score")
results = db.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")


# ## Working with vectorstore
# 
# Above, we created a vectorstore from scratch. However, often times we want to work with an existing vectorstore.
# In order to do that, we can initialize it directly.

# In[11]:


store = Kinetica(
    collection_name=COLLECTION_NAME,
    config=connection,
    embedding_function=embeddings,
)


# ### Add documents
# We can add documents to the existing vectorstore.

# In[12]:


store.add_documents([Document(page_content="foo")])


# In[13]:


docs_with_score = db.similarity_search_with_score("foo")


# In[14]:


docs_with_score[0]


# In[15]:


docs_with_score[1]


# ### Overriding a vectorstore
# 
# If you have an existing collection, you override it by doing `from_documents` and setting `pre_delete_collection` = True

# In[16]:


db = Kinetica.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    config=connection,
    pre_delete_collection=True,
)


# In[17]:


docs_with_score = db.similarity_search_with_score("foo")


# In[18]:


docs_with_score[0]


# ### Using a VectorStore as a Retriever

# In[19]:


retriever = store.as_retriever()


# In[20]:


print(retriever)

