#!/usr/bin/env python
# coding: utf-8

# # Intel's Visual Data Management System (VDMS)
# 
# >Intel's [VDMS](https://github.com/IntelLabs/vdms) is a storage solution for efficient access of big-”visual”-data that aims to achieve cloud scale by searching for relevant visual data via visual metadata stored as a graph and enabling machine friendly enhancements to visual data for faster access. VDMS is licensed under MIT.
# 
# VDMS supports:
# * K nearest neighbor search
# * Euclidean distance (L2) and inner product (IP)
# * Libraries for indexing and computing distances: TileDBDense, TileDBSparse, FaissFlat (Default), FaissIVFFlat, Flinng
# * Embeddings for text, images, and video
# * Vector and metadata searches
# 
# VDMS has server and client components. To setup the server, see the [installation instructions](https://github.com/IntelLabs/vdms/blob/master/INSTALL.md) or use the [docker image](https://hub.docker.com/r/intellabs/vdms).
# 
# This notebook shows how to use VDMS as a vector store using the docker image.
# 
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
# 
# To begin, install the Python packages for the VDMS client and Sentence Transformers:

# In[1]:


# Pip install necessary package
get_ipython().run_line_magic('pip', 'install --upgrade --quiet pip vdms sentence-transformers langchain-huggingface > /dev/null')


# ## Start VDMS Server
# Here we start the VDMS server with port 55555.

# In[2]:


get_ipython().system('docker run --rm -d -p 55555:55555 --name vdms_vs_test_nb intellabs/vdms:latest')


# ## Basic Example (using the Docker Container)
# 
# In this basic example, we demonstrate adding documents into VDMS and using it as a vector database.
# 
# You can run the VDMS Server in a Docker container separately to use with LangChain which connects to the server via the VDMS Python Client. 
# 
# VDMS has the ability to handle multiple collections of documents, but the LangChain interface expects one, so we need to specify the name of the collection . The default collection name used by LangChain is "langchain".
# 

# In[3]:


import time
import warnings

warnings.filterwarnings("ignore")

from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import VDMS
from langchain_community.vectorstores.vdms import VDMS_Client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter

time.sleep(2)
DELIMITER = "-" * 50

# Connect to VDMS Vector Store
vdms_client = VDMS_Client(host="localhost", port=55555)


# Here are some helper functions for printing results.

# In[4]:


def print_document_details(doc):
    print(f"Content:\n\t{doc.page_content}\n")
    print("Metadata:")
    for key, value in doc.metadata.items():
        if value != "Missing property":
            print(f"\t{key}:\t{value}")


def print_results(similarity_results, score=True):
    print(f"{DELIMITER}\n")
    if score:
        for doc, score in similarity_results:
            print(f"Score:\t{score}\n")
            print_document_details(doc)
            print(f"{DELIMITER}\n")
    else:
        for doc in similarity_results:
            print_document_details(doc)
            print(f"{DELIMITER}\n")


def print_response(list_of_entities):
    for ent in list_of_entities:
        for key, value in ent.items():
            if value != "Missing property":
                print(f"\n{key}:\n\t{value}")
        print(f"{DELIMITER}\n")


# ### Load Document and Obtain Embedding Function
# Here we load the most recent State of the Union Address and split the document into chunks. 
# 
# LangChain vector stores use a string/keyword `id` for bookkeeping documents. By default, `id` is a uuid but here we're defining it as an integer cast as a string. Additional metadata is also provided with the documents and the HuggingFaceEmbeddings are used for this example as the embedding function.

# In[5]:


# load the document and split it into chunks
document_path = "../../how_to/state_of_the_union.txt"
raw_documents = TextLoader(document_path).load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(raw_documents)
ids = []
for doc_idx, doc in enumerate(docs):
    ids.append(str(doc_idx + 1))
    docs[doc_idx].metadata["id"] = str(doc_idx + 1)
    docs[doc_idx].metadata["page_number"] = int(doc_idx + 1)
    docs[doc_idx].metadata["president_included"] = (
        "president" in doc.page_content.lower()
    )
print(f"# Documents: {len(docs)}")


# create the open-source embedding function
model_name = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=model_name)
print(
    f"# Embedding Dimensions: {len(embedding.embed_query('This is a test document.'))}"
)


# ### Similarity Search using Faiss Flat and Euclidean Distance (Default)
# 
# In this section, we add the documents to VDMS using FAISS IndexFlat indexing (default) and Euclidena distance (default) as the distance metric for simiarity search. We search for three documents (`k=3`) related to the query `What did the president say about Ketanji Brown Jackson`.

# In[6]:


# add data
collection_name = "my_collection_faiss_L2"
db_FaissFlat = VDMS.from_documents(
    docs,
    client=vdms_client,
    ids=ids,
    collection_name=collection_name,
    embedding=embedding,
)

# Query (No metadata filtering)
k = 3
query = "What did the president say about Ketanji Brown Jackson"
returned_docs = db_FaissFlat.similarity_search(query, k=k, filter=None)
print_results(returned_docs, score=False)


# In[7]:


# Query (with filtering)
k = 3
constraints = {"page_number": [">", 30], "president_included": ["==", True]}
query = "What did the president say about Ketanji Brown Jackson"
returned_docs = db_FaissFlat.similarity_search(query, k=k, filter=constraints)
print_results(returned_docs, score=False)


# ### Similarity Search using Faiss IVFFlat and Inner Product (IP) Distance
# 
# In this section, we add the documents to VDMS using Faiss IndexIVFFlat indexing and IP as the distance metric for similarity search. We search for three documents (`k=3`) related to the query `What did the president say about Ketanji Brown Jackson` and also return the score along with the document.
# 

# In[8]:


db_FaissIVFFlat = VDMS.from_documents(
    docs,
    client=vdms_client,
    ids=ids,
    collection_name="my_collection_FaissIVFFlat_IP",
    embedding=embedding,
    engine="FaissIVFFlat",
    distance_strategy="IP",
)
# Query
k = 3
query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db_FaissIVFFlat.similarity_search_with_score(query, k=k, filter=None)
print_results(docs_with_score)


# ### Similarity Search using FLINNG and IP Distance
# 
# In this section, we add the documents to VDMS using Filters to Identify Near-Neighbor Groups (FLINNG) indexing and IP as the distance metric for similarity search. We search for three documents (`k=3`) related to the query `What did the president say about Ketanji Brown Jackson` and also return the score along with the document.

# In[9]:


db_Flinng = VDMS.from_documents(
    docs,
    client=vdms_client,
    ids=ids,
    collection_name="my_collection_Flinng_IP",
    embedding=embedding,
    engine="Flinng",
    distance_strategy="IP",
)
# Query
k = 3
query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db_Flinng.similarity_search_with_score(query, k=k, filter=None)
print_results(docs_with_score)


# ### Similarity Search using TileDBDense and Euclidean Distance
# 
# In this section, we add the documents to VDMS using TileDB Dense indexing and L2 as the distance metric for similarity search. We search for three documents (`k=3`) related to the query `What did the president say about Ketanji Brown Jackson` and also return the score along with the document.
# 
# 

# In[10]:


db_tiledbD = VDMS.from_documents(
    docs,
    client=vdms_client,
    ids=ids,
    collection_name="my_collection_tiledbD_L2",
    embedding=embedding,
    engine="TileDBDense",
    distance_strategy="L2",
)

k = 3
query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db_tiledbD.similarity_search_with_score(query, k=k, filter=None)
print_results(docs_with_score)


# ### Update and Delete
# 
# While building toward a real application, you want to go beyond adding data, and also update and delete data.
# 
# Here is a basic example showing how to do so.  First, we will update the metadata for the document most relevant to the query by adding a date. 

# In[11]:


from datetime import datetime

doc = db_FaissFlat.similarity_search(query)[0]
print(f"Original metadata: \n\t{doc.metadata}")

# Update the metadata for a document by adding last datetime document read
datetime_str = datetime(2024, 5, 1, 14, 30, 0).isoformat()
doc.metadata["last_date_read"] = {"_date": datetime_str}
print(f"new metadata: \n\t{doc.metadata}")
print(f"{DELIMITER}\n")

# Update document in VDMS
id_to_update = doc.metadata["id"]
db_FaissFlat.update_document(collection_name, id_to_update, doc)
response, response_array = db_FaissFlat.get(
    collection_name,
    constraints={
        "id": ["==", id_to_update],
        "last_date_read": [">=", {"_date": "2024-05-01T00:00:00"}],
    },
)

# Display Results
print(f"UPDATED ENTRY (id={id_to_update}):")
print_response([response[0]["FindDescriptor"]["entities"][0]])


# Next we will delete the last document by ID (id=42).

# In[12]:


print("Documents before deletion: ", db_FaissFlat.count(collection_name))

id_to_remove = ids[-1]
db_FaissFlat.delete(collection_name=collection_name, ids=[id_to_remove])
print(
    f"Documents after deletion (id={id_to_remove}): {db_FaissFlat.count(collection_name)}"
)


# ## Other Information
# VDMS supports various types of visual data and operations. Some of the capabilities are integrated in the LangChain interface but additional workflow improvements will be added as VDMS is under continuous development.
# 
# Addtional capabilities integrated into LangChain are below.
# 
# ### Similarity search by vector
# Instead of searching by string query, you can also search by embedding/vector.

# In[13]:


embedding_vector = embedding.embed_query(query)
returned_docs = db_FaissFlat.similarity_search_by_vector(embedding_vector)

# Print Results
print_document_details(returned_docs[0])


# ### Filtering on metadata
# 
# It can be helpful to narrow down the collection before working with it.
# 
# For example, collections can be filtered on metadata using the get method. A dictionary is used to filter metadata. Here we retrieve the document where `id = 2` and remove it from the vector store.

# In[14]:


response, response_array = db_FaissFlat.get(
    collection_name,
    limit=1,
    include=["metadata", "embeddings"],
    constraints={"id": ["==", "2"]},
)

# Delete id=2
db_FaissFlat.delete(collection_name=collection_name, ids=["2"])

print("Deleted entry:")
print_response([response[0]["FindDescriptor"]["entities"][0]])


# ### Retriever options
# 
# This section goes over different options for how to use VDMS as a retriever.
# 
# 
# #### Simiarity Search
# 
# Here we use similarity search in the retriever object.
# 

# In[15]:


retriever = db_FaissFlat.as_retriever()
relevant_docs = retriever.invoke(query)[0]

print_document_details(relevant_docs)


# #### Maximal Marginal Relevance Search (MMR)
# 
# In addition to using similarity search in the retriever object, you can also use `mmr`.

# In[16]:


retriever = db_FaissFlat.as_retriever(search_type="mmr")
relevant_docs = retriever.invoke(query)[0]

print_document_details(relevant_docs)


# We can also use MMR directly.

# In[17]:


mmr_resp = db_FaissFlat.max_marginal_relevance_search_with_score(query, k=2, fetch_k=10)
print_results(mmr_resp)


# ### Delete collection
# Previously, we removed documents based on its `id`. Here, all documents are removed since no ID is provided.

# In[18]:


print("Documents before deletion: ", db_FaissFlat.count(collection_name))

db_FaissFlat.delete(collection_name=collection_name)

print("Documents after deletion: ", db_FaissFlat.count(collection_name))


# ## Stop VDMS Server

# In[19]:


get_ipython().system('docker kill vdms_vs_test_nb')


# In[ ]:




