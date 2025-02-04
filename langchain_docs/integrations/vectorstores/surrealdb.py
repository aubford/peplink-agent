#!/usr/bin/env python
# coding: utf-8

# # SurrealDB
# 
# >[SurrealDB](https://surrealdb.com/) is an end-to-end cloud-native database designed for modern applications, including web, mobile, serverless, Jamstack, backend, and traditional applications. With SurrealDB, you can simplify your database and API infrastructure, reduce development time, and build secure, performant apps quickly and cost-effectively.
# >
# >**Key features of SurrealDB include:**
# >
# >* **Reduces development time:** SurrealDB simplifies your database and API stack by removing the need for most server-side components, allowing you to build secure, performant apps faster and cheaper.
# >* **Real-time collaborative API backend service:** SurrealDB functions as both a database and an API backend service, enabling real-time collaboration.
# >* **Support for multiple querying languages:** SurrealDB supports SQL querying from client devices, GraphQL, ACID transactions, WebSocket connections, structured and unstructured data, graph querying, full-text indexing, and geospatial querying.
# >* **Granular access control:** SurrealDB provides row-level permissions-based access control, giving you the ability to manage data access with precision.
# >
# >View the [features](https://surrealdb.com/features), the latest [releases](https://surrealdb.com/releases), and [documentation](https://surrealdb.com/docs).
# 
# This notebook shows how to use functionality related to the `SurrealDBStore`.

# ## Setup
# 
# Uncomment the below cells to install surrealdb.

# In[ ]:


# %pip install --upgrade --quiet  surrealdb langchain langchain-community


# ## Using SurrealDBStore

# In[1]:


# add this import for running in jupyter notebook
import nest_asyncio

nest_asyncio.apply()


# In[2]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SurrealDBStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# In[3]:


documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)


# ### Creating a SurrealDBStore object

# In[4]:


db = SurrealDBStore(
    dburl="ws://localhost:8000/rpc",  # url for the hosted SurrealDB database
    embedding_function=embeddings,
    db_user="root",  # SurrealDB credentials if needed: db username
    db_pass="root",  # SurrealDB credentials if needed: db password
    # ns="langchain", # namespace to use for vectorstore
    # db="database",  # database to use for vectorstore
    # collection="documents", #collection to use for vectorstore
)

# this is needed to initialize the underlying async library for SurrealDB
await db.initialize()

# delete all existing documents from the vectorstore collection
await db.adelete()

# add documents to the vectorstore
ids = await db.aadd_documents(docs)

# document ids of the added documents
ids[:5]


# ### (alternatively) Create a SurrealDBStore object and add documents

# In[5]:


await db.adelete()

db = await SurrealDBStore.afrom_documents(
    dburl="ws://localhost:8000/rpc",  # url for the hosted SurrealDB database
    embedding=embeddings,
    documents=docs,
    db_user="root",  # SurrealDB credentials if needed: db username
    db_pass="root",  # SurrealDB credentials if needed: db password
    # ns="langchain", # namespace to use for vectorstore
    # db="database",  # database to use for vectorstore
    # collection="documents", #collection to use for vectorstore
)


# ### Similarity search

# In[6]:


query = "What did the president say about Ketanji Brown Jackson"
docs = await db.asimilarity_search(query)


# In[7]:


print(docs[0].page_content)


# ### Similarity search with score

# The returned distance score is cosine distance. Therefore, a lower score is better.

# In[8]:


docs = await db.asimilarity_search_with_score(query)


# In[9]:


docs[0]

