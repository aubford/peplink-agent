#!/usr/bin/env python
# coding: utf-8

# # DocArray
# 
# >[DocArray](https://github.com/docarray/docarray) is a versatile, open-source tool for managing your multi-modal data. It lets you shape your data however you want, and offers the flexibility to store and search it using various document index backends. Plus, it gets even better - you can utilize your `DocArray` document index to create a `DocArrayRetriever`, and build awesome Langchain apps!
# 
# This notebook is split into two sections. The [first section](#document-index-backends) offers an introduction to all five supported document index backends. It provides guidance on setting up and indexing each backend and also instructs you on how to build a `DocArrayRetriever` for finding relevant documents. 
# In the [second section](#movie-retrieval-using-hnswdocumentindex), we'll select one of these backends and illustrate how to use it through a basic example.
# 

# ## Document Index Backends

# In[2]:


import random

from docarray import BaseDoc
from docarray.typing import NdArray
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import DocArrayRetriever

embeddings = FakeEmbeddings(size=32)


# Before you start building the index, it's important to define your document schema. This determines what fields your documents will have and what type of data each field will hold.
# 
# For this demonstration, we'll create a somewhat random schema containing 'title' (str), 'title_embedding' (numpy array), 'year' (int), and 'color' (str)

# In[2]:


class MyDoc(BaseDoc):
    title: str
    title_embedding: NdArray[32]
    year: int
    color: str


# ### InMemoryExactNNIndex
# 
# `InMemoryExactNNIndex` stores all Documents in memory. It is a great starting point for small datasets, where you may not want to launch a database server.
# 
# Learn more here: https://docs.docarray.org/user_guide/storing/index_in_memory/

# In[3]:


from docarray.index import InMemoryExactNNIndex

# initialize the index
db = InMemoryExactNNIndex[MyDoc]()
# index data
db.index(
    [
        MyDoc(
            title=f"My document {i}",
            title_embedding=embeddings.embed_query(f"query {i}"),
            year=i,
            color=random.choice(["red", "green", "blue"]),
        )
        for i in range(100)
    ]
)
# optionally, you can create a filter query
filter_query = {"year": {"$lte": 90}}


# In[4]:


# create a retriever
retriever = DocArrayRetriever(
    index=db,
    embeddings=embeddings,
    search_field="title_embedding",
    content_field="title",
    filters=filter_query,
)

# find the relevant document
doc = retriever.invoke("some query")
print(doc)


# ### HnswDocumentIndex
# 
# `HnswDocumentIndex` is a lightweight Document Index implementation that runs fully locally and is best suited for small- to medium-sized datasets. It stores vectors on disk in [hnswlib](https://github.com/nmslib/hnswlib), and stores all other data in [SQLite](https://www.sqlite.org/index.html).
# 
# Learn more here: https://docs.docarray.org/user_guide/storing/index_hnswlib/

# In[5]:


from docarray.index import HnswDocumentIndex

# initialize the index
db = HnswDocumentIndex[MyDoc](work_dir="hnsw_index")

# index data
db.index(
    [
        MyDoc(
            title=f"My document {i}",
            title_embedding=embeddings.embed_query(f"query {i}"),
            year=i,
            color=random.choice(["red", "green", "blue"]),
        )
        for i in range(100)
    ]
)
# optionally, you can create a filter query
filter_query = {"year": {"$lte": 90}}


# In[6]:


# create a retriever
retriever = DocArrayRetriever(
    index=db,
    embeddings=embeddings,
    search_field="title_embedding",
    content_field="title",
    filters=filter_query,
)

# find the relevant document
doc = retriever.invoke("some query")
print(doc)


# ### WeaviateDocumentIndex
# 
# `WeaviateDocumentIndex` is a document index that is built upon [Weaviate](https://weaviate.io/) vector database.
# 
# Learn more here: https://docs.docarray.org/user_guide/storing/index_weaviate/

# In[7]:


# There's a small difference with the Weaviate backend compared to the others.
# Here, you need to 'mark' the field used for vector search with 'is_embedding=True'.
# So, let's create a new schema for Weaviate that takes care of this requirement.

from pydantic import Field


class WeaviateDoc(BaseDoc):
    title: str
    title_embedding: NdArray[32] = Field(is_embedding=True)
    year: int
    color: str


# In[8]:


from docarray.index import WeaviateDocumentIndex

# initialize the index
dbconfig = WeaviateDocumentIndex.DBConfig(host="http://localhost:8080")
db = WeaviateDocumentIndex[WeaviateDoc](db_config=dbconfig)

# index data
db.index(
    [
        MyDoc(
            title=f"My document {i}",
            title_embedding=embeddings.embed_query(f"query {i}"),
            year=i,
            color=random.choice(["red", "green", "blue"]),
        )
        for i in range(100)
    ]
)
# optionally, you can create a filter query
filter_query = {"path": ["year"], "operator": "LessThanEqual", "valueInt": "90"}


# In[9]:


# create a retriever
retriever = DocArrayRetriever(
    index=db,
    embeddings=embeddings,
    search_field="title_embedding",
    content_field="title",
    filters=filter_query,
)

# find the relevant document
doc = retriever.invoke("some query")
print(doc)


# ### ElasticDocIndex
# 
# `ElasticDocIndex` is a document index that is built upon [ElasticSearch](https://github.com/elastic/elasticsearch)
# 
# Learn more [here](https://docs.docarray.org/user_guide/storing/index_elastic/)

# In[10]:


from docarray.index import ElasticDocIndex

# initialize the index
db = ElasticDocIndex[MyDoc](
    hosts="http://localhost:9200", index_name="docarray_retriever"
)

# index data
db.index(
    [
        MyDoc(
            title=f"My document {i}",
            title_embedding=embeddings.embed_query(f"query {i}"),
            year=i,
            color=random.choice(["red", "green", "blue"]),
        )
        for i in range(100)
    ]
)
# optionally, you can create a filter query
filter_query = {"range": {"year": {"lte": 90}}}


# In[11]:


# create a retriever
retriever = DocArrayRetriever(
    index=db,
    embeddings=embeddings,
    search_field="title_embedding",
    content_field="title",
    filters=filter_query,
)

# find the relevant document
doc = retriever.invoke("some query")
print(doc)


# ### QdrantDocumentIndex
# 
# `QdrantDocumentIndex` is a document index that is built upon [Qdrant](https://qdrant.tech/) vector database
# 
# Learn more [here](https://docs.docarray.org/user_guide/storing/index_qdrant/)

# In[12]:


from docarray.index import QdrantDocumentIndex
from qdrant_client.http import models as rest

# initialize the index
qdrant_config = QdrantDocumentIndex.DBConfig(path=":memory:")
db = QdrantDocumentIndex[MyDoc](qdrant_config)

# index data
db.index(
    [
        MyDoc(
            title=f"My document {i}",
            title_embedding=embeddings.embed_query(f"query {i}"),
            year=i,
            color=random.choice(["red", "green", "blue"]),
        )
        for i in range(100)
    ]
)
# optionally, you can create a filter query
filter_query = rest.Filter(
    must=[
        rest.FieldCondition(
            key="year",
            range=rest.Range(
                gte=10,
                lt=90,
            ),
        )
    ]
)


# In[13]:


# create a retriever
retriever = DocArrayRetriever(
    index=db,
    embeddings=embeddings,
    search_field="title_embedding",
    content_field="title",
    filters=filter_query,
)

# find the relevant document
doc = retriever.invoke("some query")
print(doc)


# ## Movie Retrieval using HnswDocumentIndex

# In[14]:


movies = [
    {
        "title": "Inception",
        "description": "A thief who steals corporate secrets through the use of dream-sharing technology is given the task of planting an idea into the mind of a CEO.",
        "director": "Christopher Nolan",
        "rating": 8.8,
    },
    {
        "title": "The Dark Knight",
        "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
        "director": "Christopher Nolan",
        "rating": 9.0,
    },
    {
        "title": "Interstellar",
        "description": "Interstellar explores the boundaries of human exploration as a group of astronauts venture through a wormhole in space. In their quest to ensure the survival of humanity, they confront the vastness of space-time and grapple with love and sacrifice.",
        "director": "Christopher Nolan",
        "rating": 8.6,
    },
    {
        "title": "Pulp Fiction",
        "description": "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
        "director": "Quentin Tarantino",
        "rating": 8.9,
    },
    {
        "title": "Reservoir Dogs",
        "description": "When a simple jewelry heist goes horribly wrong, the surviving criminals begin to suspect that one of them is a police informant.",
        "director": "Quentin Tarantino",
        "rating": 8.3,
    },
    {
        "title": "The Godfather",
        "description": "An aging patriarch of an organized crime dynasty transfers control of his empire to his reluctant son.",
        "director": "Francis Ford Coppola",
        "rating": 9.2,
    },
]


# In[15]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[16]:


from docarray import BaseDoc, DocList
from docarray.typing import NdArray
from langchain_openai import OpenAIEmbeddings


# define schema for your movie documents
class MyDoc(BaseDoc):
    title: str
    description: str
    description_embedding: NdArray[1536]
    rating: float
    director: str


embeddings = OpenAIEmbeddings()


# get "description" embeddings, and create documents
docs = DocList[MyDoc](
    [
        MyDoc(
            description_embedding=embeddings.embed_query(movie["description"]), **movie
        )
        for movie in movies
    ]
)


# In[17]:


from docarray.index import HnswDocumentIndex

# initialize the index
db = HnswDocumentIndex[MyDoc](work_dir="movie_search")

# add data
db.index(docs)


# ### Normal Retriever

# In[18]:


from langchain_community.retrievers import DocArrayRetriever

# create a retriever
retriever = DocArrayRetriever(
    index=db,
    embeddings=embeddings,
    search_field="description_embedding",
    content_field="description",
)

# find the relevant document
doc = retriever.invoke("movie about dreams")
print(doc)


# ### Retriever with Filters

# In[19]:


from langchain_community.retrievers import DocArrayRetriever

# create a retriever
retriever = DocArrayRetriever(
    index=db,
    embeddings=embeddings,
    search_field="description_embedding",
    content_field="description",
    filters={"director": {"$eq": "Christopher Nolan"}},
    top_k=2,
)

# find relevant documents
docs = retriever.invoke("space travel")
print(docs)


# ### Retriever with MMR search

# In[20]:


from langchain_community.retrievers import DocArrayRetriever

# create a retriever
retriever = DocArrayRetriever(
    index=db,
    embeddings=embeddings,
    search_field="description_embedding",
    content_field="description",
    filters={"rating": {"$gte": 8.7}},
    search_type="mmr",
    top_k=3,
)

# find relevant documents
docs = retriever.invoke("action movies")
print(docs)


# In[ ]:




