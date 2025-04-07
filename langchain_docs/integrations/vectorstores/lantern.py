#!/usr/bin/env python
# coding: utf-8

# # Lantern
# 
# >[Lantern](https://github.com/lanterndata/lantern) is an open-source vector similarity search for `Postgres`
# 
# It supports:
# - Exact and approximate nearest neighbor search
# - L2 squared distance, hamming distance, and cosine distance
# 
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
# 
# This notebook shows how to use the Postgres vector database (`Lantern`).

# See the [installation instruction](https://github.com/lanterndata/lantern#-quick-install).

# We want to use `OpenAIEmbeddings` so we have to get the OpenAI API Key.

# # Pip install necessary package
# !pip install openai
# !pip install psycopg2-binary
# !pip install tiktoken

# In[1]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[2]:


## Loading Environment Variables
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()


# In[5]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Lantern
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# In[6]:


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# In[8]:


# Lantern needs the connection string to the database.
# Example postgresql://postgres:postgres@localhost:5432/postgres
CONNECTION_STRING = getpass.getpass("DB Connection String:")

# # Alternatively, you can create it from environment variables.
# import os

# CONNECTION_STRING = Lantern.connection_string_from_db_params(
#     driver=os.environ.get("LANTERN_DRIVER", "psycopg2"),
#     host=os.environ.get("LANTERN_HOST", "localhost"),
#     port=int(os.environ.get("LANTERN_PORT", "5432")),
#     database=os.environ.get("LANTERN_DATABASE", "postgres"),
#     user=os.environ.get("LANTERN_USER", "postgres"),
#     password=os.environ.get("LANTERN_PASSWORD", "postgres"),
# )

# or you can pass it via `LANTERN_CONNECTION_STRING` env variable


# ## Similarity Search with Cosine Distance (Default)

# In[10]:


# The Lantern Module will try to create a table with the name of the collection.
# So, make sure that the collection name is unique and the user has the permission to create a table.

COLLECTION_NAME = "state_of_the_union_test"

db = Lantern.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True,
)


# In[11]:


query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)


# In[12]:


for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)


# ## Maximal Marginal Relevance Search (MMR)
# Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.

# In[13]:


docs_with_score = db.max_marginal_relevance_search_with_score(query)


# In[14]:


for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)


# ## Working with vectorstore
# 
# Above, we created a vectorstore from scratch. However, often times we want to work with an existing vectorstore.
# In order to do that, we can initialize it directly.

# In[15]:


store = Lantern(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)


# ### Add documents
# We can add documents to the existing vectorstore.

# In[16]:


store.add_documents([Document(page_content="foo")])


# In[17]:


docs_with_score = db.similarity_search_with_score("foo")


# In[18]:


docs_with_score[0]


# In[19]:


docs_with_score[1]


# ### Overriding a vectorstore
# 
# If you have an existing collection, you override it by doing `from_documents` and setting `pre_delete_collection` = True 
# This will delete the collection before re-populating it

# In[20]:


db = Lantern.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True,
)


# In[21]:


docs_with_score = db.similarity_search_with_score("foo")


# In[22]:


docs_with_score[0]


# ### Using a VectorStore as a Retriever

# In[23]:


retriever = store.as_retriever()


# In[24]:


print(retriever)


# In[ ]:




