#!/usr/bin/env python
# coding: utf-8

# # Azure Cosmos DB Mongo vCore
#
# This notebook shows you how to leverage this integrated [vector database](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-database) to store documents in collections, create indicies and perform vector search queries using approximate nearest neighbor algorithms such as COS (cosine distance), L2 (Euclidean distance), and IP (inner product) to locate documents close to the query vectors.
#
# Azure Cosmos DB is the database that powers OpenAI's ChatGPT service. It offers single-digit millisecond response times, automatic and instant scalability, along with guaranteed speed at any scale.
#
# Azure Cosmos DB for MongoDB vCore(https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/) provides developers with a fully managed MongoDB-compatible database service for building modern applications with a familiar architecture. You can apply your MongoDB experience and continue to use your favorite MongoDB drivers, SDKs, and tools by pointing your application to the API for MongoDB vCore account's connection string.
#
# [Sign Up](https://azure.microsoft.com/en-us/free/) for lifetime free access to get started today.
#

#

# In[1]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  pymongo langchain-openai langchain-community"
)


# In[2]:


import os

CONNECTION_STRING = "YOUR_CONNECTION_STRING"
INDEX_NAME = "izzy-test-index"
NAMESPACE = "izzy_test_db.izzy_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")


# We want to use `AzureOpenAIEmbeddings` so we need to set up our Azure OpenAI API Key alongside other environment variables.

# In[3]:


# Set up the OpenAI Environment Variables

os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_EMBEDDINGS_MODEL_NAME"] = "text-embedding-ada-002"  # the model name


# Now, we need to load the documents into the collection, create the index and then run our queries against the index to retrieve matches.
#
# Please refer to the [documentation](https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/vector-search) if you have questions about certain parameters

# In[4]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

SOURCE_FILE_NAME = "../../how_to/state_of_the_union.txt"

loader = TextLoader(SOURCE_FILE_NAME)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# OpenAI Settings
model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")


openai_embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    model=model_name, chunk_size=1
)


# In[5]:


docs[0]


# In[6]:


from pymongo import MongoClient

# INDEX_NAME = "izzy-test-index-2"
# NAMESPACE = "izzy_test_db.izzy_test_collection"
# DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

client: MongoClient = MongoClient(CONNECTION_STRING)
collection = client[DB_NAME][COLLECTION_NAME]

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")

vectorstore = AzureCosmosDBVectorSearch.from_documents(
    docs,
    openai_embeddings,
    collection=collection,
    index_name=INDEX_NAME,
)

# Read more about these variables in detail here. https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/vector-search
num_lists = 100
dimensions = 1536
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_IVF
m = 16
ef_construction = 64
ef_search = 40
score_threshold = 0.1

vectorstore.create_index(
    num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
)

"""
# DiskANN vectorstore
maxDegree = 40
dimensions = 1536
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_DISKANN
lBuild = 20

vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=kind ,
            max_degree=maxDegree,
            l_build=lBuild,
        )

# -----------------------------------------------------------

# HNSW vectorstore
dimensions = 1536
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_HNSW
m = 16
ef_construction = 64

vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=kind ,
            m=m,
            ef_construction=ef_construction,
        )
"""


# In[7]:


# perform a similarity search between the embedding of the query and the embeddings of the documents
query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)


# In[8]:


print(docs[0].page_content)


# Once the documents have been loaded and the index has been created, you can now instantiate the vector store directly and run queries against the index

# In[9]:


vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
    CONNECTION_STRING, NAMESPACE, openai_embeddings, index_name=INDEX_NAME
)

# perform a similarity search between a query and the ingested documents
query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)

print(docs[0].page_content)


# In[10]:


vectorstore = AzureCosmosDBVectorSearch(
    collection, openai_embeddings, index_name=INDEX_NAME
)

# perform a similarity search between a query and the ingested documents
query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)

print(docs[0].page_content)


# ## Filtered vector search (Preview)
# Azure Cosmos DB for MongoDB supports pre-filtering with $lt, $lte, $eq, $neq, $gte, $gt, $in, $nin, and $regex. To use this feature, enable "filtering vector search" in the "Preview Features" tab of your Azure Subscription. Learn more about preview features [here](https://learn.microsoft.com/azure/cosmos-db/mongodb/vcore/vector-search#filtered-vector-search-preview).

# In[29]:


# create a filter index
vectorstore.create_filter_index(
    property_to_filter="metadata.source", index_name="filter_index"
)


# In[30]:


query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(
    query, pre_filter={"metadata.source": {"$ne": "filter content"}}
)


# In[31]:


len(docs)


# In[32]:


docs = vectorstore.similarity_search(
    query,
    pre_filter={"metadata.source": {"$ne": "../../how_to/state_of_the_union.txt"}},
)


# In[33]:


len(docs)


# In[ ]:
