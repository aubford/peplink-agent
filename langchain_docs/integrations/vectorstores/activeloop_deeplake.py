#!/usr/bin/env python
# coding: utf-8

# #  Activeloop Deep Lake
# 
# >[Activeloop Deep Lake](https://docs.activeloop.ai/) as a Multi-Modal Vector Store that stores embeddings and their metadata including text, Jsons, images, audio, video, and more. It saves the data locally, in your cloud, or on Activeloop storage. It performs hybrid search including embeddings and their attributes.
# 
# This notebook showcases basic functionality related to `Activeloop Deep Lake`. While `Deep Lake` can store embeddings, it is capable of storing any type of data. It is a serverless data lake with version control, query engine and streaming dataloaders to deep learning frameworks.  
# 
# For more information, please see the Deep Lake [documentation](https://docs.activeloop.ai) or [api reference](https://docs.deeplake.ai)

# ## Setting up

# In[ ]:


get_ipython().run_line_magic('pip', "install --upgrade --quiet  langchain-openai langchain-community 'deeplake[enterprise]' tiktoken")


# ## Example provided by Activeloop
# 
# [Integration with LangChain](https://docs.activeloop.ai/tutorials/vector-store/deep-lake-vector-store-in-langchain).
# 

# ## Deep Lake locally

# In[2]:


from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# In[3]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
activeloop_token = getpass.getpass("activeloop token:")
embeddings = OpenAIEmbeddings()


# In[4]:


from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# ### Create a local dataset
# 
# Create a dataset locally at `./deeplake/`, then run similarity search. The Deeplake+LangChain integration uses Deep Lake datasets under the hood, so `dataset` and `vector store` are used interchangeably. To create a dataset in your own cloud, or in the Deep Lake storage, [adjust the path accordingly](https://docs.activeloop.ai/storage-and-credentials/storage-options).

# In[ ]:


db = DeepLake(dataset_path="./my_deeplake/", embedding=embeddings, overwrite=True)
db.add_documents(docs)
# or shorter
# db = DeepLake.from_documents(docs, dataset_path="./my_deeplake/", embedding=embeddings, overwrite=True)


# ### Query dataset

# In[5]:


query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)


# To disable dataset summary printings all the time, you can specify verbose=False during VectorStore initialization.

# In[6]:


print(docs[0].page_content)


# Later, you can reload the dataset without recomputing embeddings

# In[7]:


db = DeepLake(dataset_path="./my_deeplake/", embedding=embeddings, read_only=True)
docs = db.similarity_search(query)


# Deep Lake, for now, is single writer and multiple reader. Setting `read_only=True` helps to avoid acquiring the writer lock.

# ### Retrieval Question/Answering

# In[8]:


from langchain.chains import RetrievalQA
from langchain_openai import OpenAIChat

qa = RetrievalQA.from_chain_type(
    llm=OpenAIChat(model="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=db.as_retriever(),
)


# In[9]:


query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)


# ### Attribute based filtering in metadata

# Let's create another vector store containing metadata with the year the documents were created.

# In[10]:


import random

for d in docs:
    d.metadata["year"] = random.randint(2012, 2014)

db = DeepLake.from_documents(
    docs, embeddings, dataset_path="./my_deeplake/", overwrite=True
)


# In[11]:


db.similarity_search(
    "What did the president say about Ketanji Brown Jackson",
    filter={"metadata": {"year": 2013}},
)


# ### Choosing distance function
# Distance function `L2` for Euclidean, `L1` for Nuclear, `Max` l-infinity distance, `cos` for cosine similarity, `dot` for dot product 

# In[12]:


db.similarity_search(
    "What did the president say about Ketanji Brown Jackson?", distance_metric="cos"
)


# ### Maximal Marginal relevance
# Using maximal marginal relevance

# In[13]:


db.max_marginal_relevance_search(
    "What did the president say about Ketanji Brown Jackson?"
)


# ### Delete dataset

# In[14]:


db.delete_dataset()


# and if delete fails you can also force delete

# In[15]:


DeepLake.force_delete_by_path("./my_deeplake")


# ## Deep Lake datasets on cloud (Activeloop, AWS, GCS, etc.) or in memory
# By default, Deep Lake datasets are stored locally. To store them in memory, in the Deep Lake Managed DB, or in any object storage, you can provide the [corresponding path and credentials when creating the vector store](https://docs.activeloop.ai/storage-and-credentials/storage-options). Some paths require registration with Activeloop and creation of an API token that can be [retrieved here](https://app.activeloop.ai/)

# In[16]:


os.environ["ACTIVELOOP_TOKEN"] = activeloop_token


# In[26]:


# Embed and store the texts
username = "<USERNAME_OR_ORG>"  # your username on app.activeloop.ai
dataset_path = f"hub://{username}/langchain_testing_python"  # could be also ./local/path (much faster locally), s3://bucket/path/to/dataset, gcs://path/to/dataset, etc.

docs = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
db = DeepLake(dataset_path=dataset_path, embedding=embeddings, overwrite=True)
ids = db.add_documents(docs)


# In[18]:


query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)


# #### `tensor_db` execution option 

# In order to utilize Deep Lake's Managed Tensor Database, it is necessary to specify the runtime parameter as `{'tensor_db': True}` during the creation of the vector store. This configuration enables the execution of queries on the Managed Tensor Database, rather than on the client side. It should be noted that this functionality is not applicable to datasets stored locally or in-memory. In the event that a vector store has already been created outside of the Managed Tensor Database, it is possible to transfer it to the Managed Tensor Database by following the prescribed steps.

# In[27]:


# Embed and store the texts
username = "<USERNAME_OR_ORG>"  # your username on app.activeloop.ai
dataset_path = f"hub://{username}/langchain_testing"

docs = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
db = DeepLake(
    dataset_path=dataset_path,
    embedding=embeddings,
    overwrite=True,
    runtime={"tensor_db": True},
)
ids = db.add_documents(docs)


# ### TQL Search

# Furthermore, the execution of queries is also supported within the similarity_search method, whereby the query can be specified utilizing Deep Lake's Tensor Query Language (TQL).

# In[21]:


search_id = db.vectorstore.dataset.id[0].numpy()


# In[22]:


search_id[0]


# In[23]:


docs = db.similarity_search(
    query=None,
    tql=f"SELECT * WHERE id == '{search_id[0]}'",
)


# In[25]:


db.vectorstore.summary()


# ### Creating vector stores on AWS S3

# In[82]:


dataset_path = "s3://BUCKET/langchain_test"  # could be also ./local/path (much faster locally), hub://bucket/path/to/dataset, gcs://path/to/dataset, etc.

embedding = OpenAIEmbeddings()
db = DeepLake.from_documents(
    docs,
    dataset_path=dataset_path,
    embedding=embeddings,
    overwrite=True,
    creds={
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "aws_session_token": os.environ["AWS_SESSION_TOKEN"],  # Optional
    },
)


# ## Deep Lake API
# you can access the Deep Lake  dataset at `db.vectorstore`

# In[26]:


# get structure of the dataset
db.vectorstore.summary()


# In[27]:


# get embeddings numpy array
embeds = db.vectorstore.dataset.embedding.numpy()


# ### Transfer local dataset to cloud
# Copy already created dataset to the cloud. You can also transfer from cloud to local.

# In[73]:


import deeplake

username = "davitbun"  # your username on app.activeloop.ai
source = f"hub://{username}/langchain_testing"  # could be local, s3, gcs, etc.
destination = f"hub://{username}/langchain_test_copy"  # could be local, s3, gcs, etc.

deeplake.deepcopy(src=source, dest=destination, overwrite=True)


# In[76]:


db = DeepLake(dataset_path=destination, embedding=embeddings)
db.add_documents(docs)

