#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Redis
---
# # Redis Vector Store
# 
# This notebook covers how to get started with the Redis vector store.
# 
# >[Redis](https://redis.io/docs/stack/vectorsearch/) is a popular open-source, in-memory data structure store that can be used as a database, cache, message broker, and queue. It now includes vector similarity search capabilities, making it suitable for use as a vector store.

# ### What is Redis?
# 
# Most developers are familiar with `Redis`. At its core, `Redis` is a NoSQL Database in the key-value family that can used as a cache, message broker, stream processing and a primary database. Developers choose `Redis` because it is fast, has a large ecosystem of client libraries, and has been deployed by major enterprises for years.
# 
# On top of these traditional use cases, `Redis` provides additional capabilities like the Search and Query capability that allows users to create secondary index structures within `Redis`. This allows `Redis` to be a Vector Database, at the speed of a cache. 
# 
# 
# ### Redis as a Vector Database
# 
# `Redis` uses compressed, inverted indexes for fast indexing with a low memory footprint. It also supports a number of advanced features such as:
# 
# * Indexing of multiple fields in Redis hashes and `JSON`
# * Vector similarity search (with `HNSW` (ANN) or `FLAT` (KNN))
# * Vector Range Search (e.g. find all vectors within a radius of a query vector)
# * Incremental indexing without performance loss
# * Document ranking (using [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), with optional user-provided weights)
# * Field weighting
# * Complex boolean queries with `AND`, `OR`, and `NOT` operators
# * Prefix matching, fuzzy matching, and exact-phrase queries
# * Support for [double-metaphone phonetic matching](https://redis.io/docs/stack/search/reference/phonetic_matching/)
# * Auto-complete suggestions (with fuzzy prefix suggestions)
# * Stemming-based query expansion in [many languages](https://redis.io/docs/stack/search/reference/stemming/) (using [Snowball](http://snowballstem.org/))
# * Support for Chinese-language tokenization and querying (using [Friso](https://github.com/lionsoul2014/friso))
# * Numeric filters and ranges
# * Geospatial searches using Redis geospatial indexing
# * A powerful aggregations engine
# * Supports for all `utf-8` encoded text
# * Retrieve full documents, selected fields, or only the document IDs
# * Sorting results (for example, by creation date)
# 
# ### Clients
# 
# Since `Redis` is much more than just a vector database, there are often use cases that demand the usage of a `Redis` client besides just the `LangChain` integration. You can use any standard `Redis` client library to run Search and Query commands, but it's easiest to use a library that wraps the Search and Query API. Below are a few examples, but you can find more client libraries [here](https://redis.io/resources/clients/).
# 
# | Project | Language | License | Author | Stars |
# |----------|---------|--------|---------|-------|
# | [jedis][jedis-url] | Java | MIT | [Redis][redis-url] | ![Stars][jedis-stars] |
# | [redisvl][redisvl-url] | Python | MIT | [Redis][redis-url] | ![Stars][redisvl-stars] |
# | [redis-py][redis-py-url] | Python | MIT | [Redis][redis-url] | ![Stars][redis-py-stars] |
# | [node-redis][node-redis-url] | Node.js | MIT | [Redis][redis-url] | ![Stars][node-redis-stars] |
# | [nredisstack][nredisstack-url] | .NET | MIT | [Redis][redis-url] | ![Stars][nredisstack-stars] |
# 
# [redis-url]: https://redis.com
# 
# [redisvl-url]: https://github.com/redis/redis-vl-python
# [redisvl-stars]: https://img.shields.io/github/stars/redis/redisvl.svg?style=social&amp;label=Star&amp;maxAge=2592000
# [redisvl-package]: https://pypi.python.org/pypi/redisvl
# 
# [redis-py-url]: https://github.com/redis/redis-py
# [redis-py-stars]: https://img.shields.io/github/stars/redis/redis-py.svg?style=social&amp;label=Star&amp;maxAge=2592000
# [redis-py-package]: https://pypi.python.org/pypi/redis
# 
# [jedis-url]: https://github.com/redis/jedis
# [jedis-stars]: https://img.shields.io/github/stars/redis/jedis.svg?style=social&amp;label=Star&amp;maxAge=2592000
# [Jedis-package]: https://search.maven.org/artifact/redis.clients/jedis
# 
# [nredisstack-url]: https://github.com/redis/nredisstack
# [nredisstack-stars]: https://img.shields.io/github/stars/redis/nredisstack.svg?style=social&amp;label=Star&amp;maxAge=2592000
# [nredisstack-package]: https://www.nuget.org/packages/nredisstack/
# 
# [node-redis-url]: https://github.com/redis/node-redis
# [node-redis-stars]: https://img.shields.io/github/stars/redis/node-redis.svg?style=social&amp;label=Star&amp;maxAge=2592000
# [node-redis-package]: https://www.npmjs.com/package/redis
# 
# [redis-om-python-url]: https://github.com/redis/redis-om-python
# [redis-om-python-author]: https://redis.com
# [redis-om-python-stars]: https://img.shields.io/github/stars/redis/redis-om-python.svg?style=social&amp;label=Star&amp;maxAge=2592000
# 
# [redisearch-go-url]: https://github.com/RediSearch/redisearch-go
# [redisearch-go-author]: https://redis.com
# [redisearch-go-stars]: https://img.shields.io/github/stars/RediSearch/redisearch-go.svg?style=social&amp;label=Star&amp;maxAge=2592000
# 
# [redisearch-api-rs-url]: https://github.com/RediSearch/redisearch-api-rs
# [redisearch-api-rs-author]: https://redis.com
# [redisearch-api-rs-stars]: https://img.shields.io/github/stars/RediSearch/redisearch-api-rs.svg?style=social&amp;label=Star&amp;maxAge=2592000
# 
# 
# ### Deployment options
# 
# There are many ways to deploy Redis with RediSearch. The easiest way to get started is to use Docker, but there are are many potential options for deployment such as
# 
# - [Redis Cloud](https://redis.com/redis-enterprise-cloud/overview/)
# - [Docker (Redis Stack)](https://hub.docker.com/r/redis/redis-stack)
# - Cloud marketplaces: [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-e6y7ork67pjwg?sr=0-2&ref_=beagle&applicationId=AWSMPContessa), [Google Marketplace](https://console.cloud.google.com/marketplace/details/redislabs-public/redis-enterprise?pli=1), or [Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/garantiadata.redis_enterprise_1sp_public_preview?tab=Overview)
# - On-premise: [Redis Enterprise Software](https://redis.com/redis-enterprise-software/overview/)
# - Kubernetes: [Redis Enterprise Software on Kubernetes](https://docs.redis.com/latest/kubernetes/)
#   
# ### Redis connection Url schemas
# 
# Valid Redis Url schemas are:
# 1. `redis://`  - Connection to Redis standalone, unencrypted
# 2. `rediss://` - Connection to Redis standalone, with TLS encryption
# 3. `redis+sentinel://`  - Connection to Redis server via Redis Sentinel, unencrypted
# 4. `rediss+sentinel://` - Connection to Redis server via Redis Sentinel, booth connections with TLS encryption
# 
# More information about additional connection parameters can be found in the [redis-py documentation](https://redis-py.readthedocs.io/en/stable/connections.html).

# ## Setup
# 
# To use the RedisVectorStore, you'll need to install the `langchain-redis` partner package, as well as the other packages used throughout this notebook.

# In[1]:


get_ipython().run_line_magic('pip', 'install -qU langchain-redis langchain-huggingface sentence-transformers scikit-learn')


# ### Credentials
# 
# Redis connection credentials are passed as part of the Redis Connection URL. Redis Connection URLs are versatile and can accommodate various Redis server topologies and authentication methods. These URLs follow a specific format that includes the connection protocol, authentication details, host, port, and database information.
# The basic structure of a Redis Connection URL is:
# 
# ```
# [protocol]://[auth]@[host]:[port]/[database]
# ```
# 
# Where:
# 
# * protocol can be redis for standard connections, rediss for SSL/TLS connections, or redis+sentinel for Sentinel connections.
# * auth includes username and password (if applicable).
# * host is the Redis server hostname or IP address.
# * port is the Redis server port.
# * database is the Redis database number.
# 
# Redis Connection URLs support various configurations, including:
# 
# * Standalone Redis servers (with or without authentication)
# * Redis Sentinel setups
# * SSL/TLS encrypted connections
# * Different authentication methods (password-only or username-password)
# 
# Below are examples of Redis Connection URLs for different configurations:

# In[2]:


# connection to redis standalone at localhost, db 0, no password
redis_url = "redis://localhost:6379"
# connection to host "redis" port 7379 with db 2 and password "secret" (old style authentication scheme without username / pre 6.x)
redis_url = "redis://:secret@redis:7379/2"
# connection to host redis on default port with user "joe", pass "secret" using redis version 6+ ACLs
redis_url = "redis://joe:secret@redis/0"

# connection to sentinel at localhost with default group mymaster and db 0, no password
redis_url = "redis+sentinel://localhost:26379"
# connection to sentinel at host redis with default port 26379 and user "joe" with password "secret" with default group mymaster and db 0
redis_url = "redis+sentinel://joe:secret@redis"
# connection to sentinel, no auth with sentinel monitoring group "zone-1" and database 2
redis_url = "redis+sentinel://redis:26379/zone-1/2"

# connection to redis standalone at localhost, db 0, no password but with TLS support
redis_url = "rediss://localhost:6379"
# connection to redis sentinel at localhost and default port, db 0, no password
# but with TLS support for booth Sentinel and Redis server
redis_url = "rediss+sentinel://localhost"


# ### Launching a Redis Instance with Docker
# 
# To use Redis with LangChain, you need a running Redis instance. You can start one using Docker with:
# 
# ```bash
# docker run -d -p 6379:6379 redis/redis-stack:latest
# ```
# 
# For this example, we'll use a local Redis instance. If you're using a remote instance, you'll need to modify the Redis URL accordingly.

# In[3]:


import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[4]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# Let's check that Redis is up an running by pinging it:

# In[5]:


import redis

redis_client = redis.from_url(REDIS_URL)
redis_client.ping()


# ### Sample Data
# 
# The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics. We'll use a subset for this demonstration and focus on two categories: 'alt.atheism' and 'sci.space':

# In[6]:


from langchain.docstore.document import Document
from sklearn.datasets import fetch_20newsgroups

categories = ["alt.atheism", "sci.space"]
newsgroups = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True, random_state=42
)

# Use only the first 250 documents
texts = newsgroups.data[:250]
metadata = [
    {"category": newsgroups.target_names[target]} for target in newsgroups.target[:250]
]

len(texts)


# ## Initialization
# 
# The RedisVectorStore instance can be initialized in several ways:
# 
# - `RedisVectorStore.__init__` - Initialize directly
# - `RedisVectorStore.from_texts` - Initialize from a list of texts (optionally with metadata)
# - `RedisVectorStore.from_documents` - Initialize from a list of `langchain_core.documents.Document` objects
# - `RedisVectorStore.from_existing_index` - Initialize from an existing Redis index
# 
# Below we will use the `RedisVectorStore.__init__` method using a `RedisConfig` instance.
# 
# import EmbeddingTabs from "@theme/EmbeddingTabs";
# 
# <EmbeddingTabs/>
# 

# In[7]:


get_ipython().run_cell_magic('capture', '', '# | output: false\n# | echo: false\nos.environ["TOKENIZERS_PARALLELISM"] = "false"\nfrom langchain_huggingface import HuggingFaceEmbeddings\nfrom tqdm.auto import tqdm\n\nembeddings = HuggingFaceEmbeddings(model_name="msmarco-distilbert-base-v4")\n')


# We'll use the SentenceTransformer model to create embeddings. This model runs locally and doesn't require an API key.

# In[8]:


from langchain_redis import RedisConfig, RedisVectorStore

config = RedisConfig(
    index_name="newsgroups",
    redis_url=REDIS_URL,
    metadata_schema=[
        {"name": "category", "type": "tag"},
    ],
)

vector_store = RedisVectorStore(embeddings, config=config)


# ## Manage vector store
# 
# ### Add items to vector store

# In[9]:


ids = vector_store.add_texts(texts, metadata)

print(ids[0:10])


# Let's inspect the first document:

# In[10]:


texts[0], metadata[0]


# ### Delete items from vector store

# In[11]:


# Delete documents by passing one or more keys/ids
vector_store.index.drop_keys(ids[0])


# ### Inspecting the created Index
# 
# Once the ``Redis`` VectorStore object has been constructed, an index will have been created in Redis if it did not already exist. The index can be inspected with both the ``rvl``and the ``redis-cli`` command line tool. If you installed ``redisvl`` above, you can use the ``rvl`` command line tool to inspect the index.

# In[12]:


# assumes you're running Redis locally (use --host, --port, --password, --username, to change this)
get_ipython().system('rvl index listall --port 6379')


# The ``Redis`` VectorStore implementation will attempt to generate index schema (fields for filtering) for any metadata passed through the ``from_texts``, ``from_texts_return_keys``, and ``from_documents`` methods. This way, whatever metadata is passed will be indexed into the Redis search index allowing
# for filtering on those fields.
# 
# Below we show what fields were created from the metadata we defined above

# In[13]:


get_ipython().system('rvl index info -i newsgroups --port 6379')


# In[14]:


get_ipython().system('rvl stats -i newsgroups --port 6379')


# ## Query vector store
# 
# Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent. 
# 
# ### Query directly
# 
# Performing a simple similarity search can be done as follows:

# In[15]:


query = "Tell me about space exploration"
results = vector_store.similarity_search(query, k=2)

print("Simple Similarity Search Results:")
for doc in results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print()


# If you want to execute a similarity search and receive the corresponding scores you can run:

# In[16]:


# Similarity search with score and filter
scored_results = vector_store.similarity_search_with_score(query, k=2)

print("Similarity Search with Score Results:")
for doc, score in scored_results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {score}")
    print()


# ### Query by turning into retriever
# 
# You can also transform the vector store into a retriever for easier usage in your chains.

# In[17]:


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever.invoke("What planet in the solar system has the largest number of moons?")


# ## Usage for retrieval-augmented generation
# 
# For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:
# 
# - [Tutorials](/docs/tutorials/)
# - [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
# - [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

# ## Redis-specific functionality
# 
# Redis offers some unique features for vector search:

# ### Similarity search with metadata filtering
# We can filter our search results based on metadata:

# In[18]:


from redisvl.query.filter import Tag

query = "Tell me about space exploration"

# Create a RedisVL filter expression
filter_condition = Tag("category") == "sci.space"

filtered_results = vector_store.similarity_search(query, k=2, filter=filter_condition)

print("Filtered Similarity Search Results:")
for doc in filtered_results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print()


# ### Maximum marginal relevance search
# Maximum marginal relevance search helps in getting diverse results:

# In[19]:


# Maximum marginal relevance search with filter
mmr_results = vector_store.max_marginal_relevance_search(
    query, k=2, fetch_k=10, filter=filter_condition
)

print("Maximum Marginal Relevance Search Results:")
for doc in mmr_results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print()


# ## Chain usage
# The code below shows how to use the vector store as a retriever in a simple RAG chain:
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs customVarName="llm" />
# 

# In[20]:


# | output: false
# | echo: false
from getpass import getpass

from langchain_openai import ChatOpenAI

# Check if OPENAI_API_KEY is already set in the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("OpenAI API key not found in environment variables.")
    openai_api_key = getpass("Please enter your OpenAI API key: ")

    # Set the API key for the current session
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print("OpenAI API key has been set for this session.")
else:
    print("OpenAI API key found in environment variables.")

llm = ChatOpenAI(model="gpt-4o-mini")


# In[21]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:""",
        ),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Describe the Space Shuttle program?")


# ## Connect to an existing Index
# 
# In order to have the same metadata indexed when using the ``Redis`` VectorStore. You will need to have the same ``index_schema`` passed in either as a path to a yaml file or as a dictionary. The following shows how to obtain the schema from an index and connect to an existing index.

# In[24]:


# write the schema to a yaml file
vector_store.index.schema.to_yaml("redis_schema.yaml")


# In[33]:


# now we can connect to our existing index as follows

new_rdvs = RedisVectorStore(
    embeddings,
    redis_url=REDIS_URL,
    schema_path="redis_schema.yaml",
)

results = new_rdvs.similarity_search("Space Shuttle Propulsion System", k=3)
print(results[0])


# In[35]:


# compare the two schemas to verify they are the same
new_rdvs.index.schema == vector_store.index.schema


# ## Cleanup vector store

# In[22]:


# Clear vector store
vector_store.index.delete(drop=True)


# ## API reference
# 
# For detailed documentation of all RedisVectorStore features and configurations head to the API reference: https://python.langchain.com/api_reference/redis/vectorstores/langchain_redis.vectorstores.RedisVectorStore.html
