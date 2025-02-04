#!/usr/bin/env python
# coding: utf-8

# # Amazon Document DB
# 
# >[Amazon DocumentDB (with MongoDB Compatibility)](https://docs.aws.amazon.com/documentdb/) makes it easy to set up, operate, and scale MongoDB-compatible databases in the cloud.
# > With Amazon DocumentDB, you can run the same application code and use the same drivers and tools that you use with MongoDB.
# > Vector search for Amazon DocumentDB combines the flexibility and rich querying capability of a JSON-based document database with the power of vector search.
# 
# 
# This notebook shows you how to use [Amazon Document DB Vector Search](https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html) to store documents in collections, create indicies and perform vector search queries using approximate nearest neighbor algorithms such "cosine", "euclidean", and "dotProduct". By default, DocumentDB creates Hierarchical Navigable Small World (HNSW) indexes. To learn about other supported vector index types, please refer to the document linked above.
# 
# To use DocumentDB, you must first deploy a cluster. Please refer to the [Developer Guide](https://docs.aws.amazon.com/documentdb/latest/developerguide/what-is.html) for more details.
# 
# [Sign Up](https://aws.amazon.com/free/) for free to get started today.
#         

# In[2]:


get_ipython().system('pip install pymongo')


# In[4]:


import getpass

# DocumentDB connection string
# i.e., "mongodb://{username}:{pass}@{cluster_endpoint}:{port}/?{params}"
CONNECTION_STRING = getpass.getpass("DocumentDB Cluster URI:")

INDEX_NAME = "izzy-test-index"
NAMESPACE = "izzy_test_db.izzy_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")


# We want to use `OpenAIEmbeddings` so we need to set up our OpenAI environment variables. 

# In[5]:


import getpass
import os

# Set up the OpenAI Environment Variables
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"] = (
    "smart-agent-embedding-ada"  # the deployment name for the embedding model
)
os.environ["OPENAI_EMBEDDINGS_MODEL_NAME"] = "text-embedding-ada-002"  # the model name


# Now, we will load the documents into the collection, create the index, and then perform queries against the index.
# 
# Please refer to the [documentation](https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html) if you have questions about certain parameters

# In[7]:


from langchain.vectorstores.documentdb import (
    DocumentDBSimilarityType,
    DocumentDBVectorSearch,
)
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
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


openai_embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
    deployment=model_deployment, model=model_name
)


# In[ ]:


from pymongo import MongoClient

INDEX_NAME = "izzy-test-index-2"
NAMESPACE = "izzy_test_db.izzy_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

client: MongoClient = MongoClient(CONNECTION_STRING)
collection = client[DB_NAME][COLLECTION_NAME]

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")

vectorstore = DocumentDBVectorSearch.from_documents(
    documents=docs,
    embedding=openai_embeddings,
    collection=collection,
    index_name=INDEX_NAME,
)

# number of dimensions used by model above
dimensions = 1536

# specify similarity algorithm, valid options are:
#   cosine (COS), euclidean (EUC), dotProduct (DOT)
similarity_algorithm = DocumentDBSimilarityType.COS

vectorstore.create_index(dimensions, similarity_algorithm)


# In[ ]:


# perform a similarity search between the embedding of the query and the embeddings of the documents
query = "What did the President say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)


# In[9]:


print(docs[0].page_content)


# Once the documents have been loaded and the index has been created, you can now instantiate the vector store directly and run queries against the index

# In[ ]:


vectorstore = DocumentDBVectorSearch.from_connection_string(
    connection_string=CONNECTION_STRING,
    namespace=NAMESPACE,
    embedding=openai_embeddings,
    index_name=INDEX_NAME,
)

# perform a similarity search between a query and the ingested documents
query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)


# In[ ]:


print(docs[0].page_content)


# In[ ]:


# perform a similarity search between a query and the ingested documents
query = "Which stats did the President share about the U.S. economy"
docs = vectorstore.similarity_search(query)


# In[ ]:


print(docs[0].page_content)


# ## Question Answering

# In[ ]:


qa_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 25},
)


# In[ ]:


from langchain_core.prompts import PromptTemplate

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# In[ ]:


from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

docs = qa({"query": "gpt-4 compute requirements"})

print(docs["result"])
print(docs["source_documents"])

