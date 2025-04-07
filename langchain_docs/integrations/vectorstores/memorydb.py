#!/usr/bin/env python
# coding: utf-8

# # Amazon MemoryDB
# 
# >[Vector Search](https://docs.aws.amazon.com/memorydb/latest/devguide/vector-search.html/) introduction and langchain integration guide.
# 
# ## What is Amazon MemoryDB?
# 
# MemoryDB is compatible with Redis OSS, a popular open source data store, enabling you to quickly build applications using the same flexible and friendly Redis OSS data structures, APIs, and commands that they already use today. With MemoryDB, all of your data is stored in memory, which enables you to achieve microsecond read and single-digit millisecond write latency and high throughput. MemoryDB also stores data durably across multiple Availability Zones (AZs) using a Multi-AZ transactional log to enable fast failover, database recovery, and node restarts.
# 
# 
# ## Vector search for MemoryDB 
# 
# Vector search for MemoryDB extends the functionality of MemoryDB. Vector search can be used in conjunction with existing MemoryDB functionality. Applications that do not use vector search are unaffected by its presence. Vector search is available in all Regions that MemoryDB is available. You can use your existing MemoryDB data or Redis OSS API to build machine learning and generative AI use cases, such as retrieval-augmented generation, anomaly detection, document retrieval, and real-time recommendations.
# 
# * Indexing of multiple fields in Redis hashes and `JSON`
# * Vector similarity search (with `HNSW` (ANN) or `FLAT` (KNN))
# * Vector Range Search (e.g. find all vectors within a radius of a query vector)
# * Incremental indexing without performance loss
# 

# ## Setting up
# 
# 
# ### Install Redis Python client
# 
# `Redis-py` is a python  client that can be used to connect to MemoryDB

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  redis langchain-aws')


# In[3]:


from langchain_aws.embeddings import BedrockEmbeddings

embeddings = BedrockEmbeddings()


# ### MemoryDB Connection
# 
# Valid Redis Url schemas are:
# 1. `redis://`  - Connection to Redis cluster, unencrypted
# 2. `rediss://` - Connection to Redis cluster, with TLS encryption
# 
# More information about additional connection parameters can be found in the [redis-py documentation](https://redis-py.readthedocs.io/en/stable/connections.html).

# ### Sample data
# 
# First we will describe some sample data so that the various attributes of the Redis vector store can be demonstrated.

# In[4]:


metadata = [
    {
        "user": "john",
        "age": 18,
        "job": "engineer",
        "credit_score": "high",
    },
    {
        "user": "derrick",
        "age": 45,
        "job": "doctor",
        "credit_score": "low",
    },
    {
        "user": "nancy",
        "age": 94,
        "job": "doctor",
        "credit_score": "high",
    },
    {
        "user": "tyler",
        "age": 100,
        "job": "engineer",
        "credit_score": "high",
    },
    {
        "user": "joe",
        "age": 35,
        "job": "dentist",
        "credit_score": "medium",
    },
]
texts = ["foo", "foo", "foo", "bar", "bar"]
index_name = "users"


# ### Create MemoryDB vector store
# 
# The InMemoryVectorStore instance can be initialized using the below methods 
# - ``InMemoryVectorStore.__init__`` - Initialize directly
# - ``InMemoryVectorStore.from_documents`` - Initialize from a list of ``Langchain.docstore.Document`` objects
# - ``InMemoryVectorStore.from_texts`` - Initialize from a list of texts (optionally with metadata)
# - ``InMemoryVectorStore.from_existing_index`` - Initialize from an existing MemoryDB index
# 

# In[7]:


from langchain_aws.vectorstores.inmemorydb import InMemoryVectorStore

vds = InMemoryVectorStore.from_texts(
    embeddings,
    redis_url="rediss://cluster_endpoint:6379/ssl=True ssl_cert_reqs=none",
)


# In[6]:


vds.index_name


# ## Querying
# 
# There are multiple ways to query the ``InMemoryVectorStore``  implementation based on what use case you have:
# 
# - ``similarity_search``: Find the most similar vectors to a given vector.
# - ``similarity_search_with_score``: Find the most similar vectors to a given vector and return the vector distance
# - ``similarity_search_limit_score``: Find the most similar vectors to a given vector and limit the number of results to the ``score_threshold``
# - ``similarity_search_with_relevance_scores``: Find the most similar vectors to a given vector and return the vector similarities
# - ``max_marginal_relevance_search``: Find the most similar vectors to a given vector while also optimizing for diversity

# In[10]:


results = vds.similarity_search("foo")
print(results[0].page_content)


# In[12]:


# with scores (distances)
results = vds.similarity_search_with_score("foo", k=5)
for result in results:
    print(f"Content: {result[0].page_content} --- Score: {result[1]}")


# In[13]:


# limit the vector distance that can be returned
results = vds.similarity_search_with_score("foo", k=5, distance_threshold=0.1)
for result in results:
    print(f"Content: {result[0].page_content} --- Score: {result[1]}")


# In[14]:


# with scores
results = vds.similarity_search_with_relevance_scores("foo", k=5)
for result in results:
    print(f"Content: {result[0].page_content} --- Similiarity: {result[1]}")


# In[16]:


# you can also add new documents as follows
new_document = ["baz"]
new_metadata = [{"user": "sam", "age": 50, "job": "janitor", "credit_score": "high"}]
# both the document and metadata must be lists
vds.add_texts(new_document, new_metadata)


# ## MemoryDB as Retriever
# 
# Here we go over different options for using the vector store as a retriever.
# 
# There are three different search methods we can use to do retrieval. By default, it will use semantic similarity.

# In[26]:


query = "foo"
results = vds.similarity_search_with_score(query, k=3, return_metadata=True)

for result in results:
    print("Content:", result[0].page_content, " --- Score: ", result[1])


# In[27]:


retriever = vds.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# In[28]:


docs = retriever.invoke(query)
docs


# There is also the `similarity_distance_threshold` retriever which allows the user to specify the vector distance

# In[29]:


retriever = vds.as_retriever(
    search_type="similarity_distance_threshold",
    search_kwargs={"k": 4, "distance_threshold": 0.1},
)


# In[30]:


docs = retriever.invoke(query)
docs


# Lastly, the ``similarity_score_threshold`` allows the user to define the minimum score for similar documents

# In[31]:


retriever = vds.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.9, "k": 10},
)


# In[32]:


retriever.invoke("foo")


# In[13]:


retriever.invoke("foo")


# ## Delete  index

# To delete your entries you have to address them by their keys.

# In[34]:


# delete the indices too
InMemoryVectorStore.drop_index(
    index_name="users", delete_documents=True, redis_url="redis://localhost:6379"
)
InMemoryVectorStore.drop_index(
    index_name="users_modified",
    delete_documents=True,
    redis_url="redis://localhost:6379",
)

