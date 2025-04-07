#!/usr/bin/env python
# coding: utf-8

# # Astra DB Vector Store
# 
# This page provides a quickstart for using [Astra DB](https://docs.datastax.com/en/astra/home/astra.html) as a Vector Store.
# 
# > DataStax [Astra DB](https://docs.datastax.com/en/astra/home/astra.html) is a serverless vector-capable database built on Apache Cassandra® and made conveniently available through an easy-to-use JSON API.
# 
# ## Setup

# Use of the integration requires the `langchain-astradb` partner package:

# In[ ]:


pip install -qU "langchain-astradb>=0.3.3"


# ### Credentials
# 
# In order to use the AstraDB vector store, you must first head to the [AstraDB website](https://astra.datastax.com), create an account, and then create a new database - the initialization might take a few minutes. 
# 
# Once the database has been initialized, you should [create an application token](https://docs.datastax.com/en/astra-db-serverless/administration/manage-application-tokens.html#generate-application-token) and save it for later use. 
# 
# You will also want to copy the `API Endpoint` from the `Database Details` and store that in the `ASTRA_DB_API_ENDPOINT` variable.
# 
# You may optionally provide a namespace, which you can manage from the `Data Explorer` tab of your database dashboard. If you don't wish to set a namespace, you can leave the `getpass` prompt for `ASTRA_DB_NAMESPACE` empty.

# In[7]:


import getpass

ASTRA_DB_API_ENDPOINT = getpass.getpass("ASTRA_DB_API_ENDPOINT = ")
ASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")

desired_namespace = getpass.getpass("ASTRA_DB_NAMESPACE = ")
if desired_namespace:
    ASTRA_DB_NAMESPACE = desired_namespace
else:
    ASTRA_DB_NAMESPACE = None


# If you want to get best in-class automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ## Initialization
# 
# There are two ways to create an Astra DB vector store, which differ in how the embeddings are computed.
# 
# #### Method 1: Explicit embeddings
# 
# You can separately instantiate a `langchain_core.embeddings.Embeddings` class and pass it to the `AstraDBVectorStore` constructor, just like with most other LangChain vector stores.
# 
# #### Method 2: Integrated embedding computation
# 
# Alternatively, you can use the [Vectorize](https://www.datastax.com/blog/simplifying-vector-embedding-generation-with-astra-vectorize) feature of Astra DB and simply specify the name of a supported embedding model when creating the store. The embedding computations are entirely handled within the database. (To proceed with this method, you must have enabled the desired embedding integration for your database, as described [in the docs](https://docs.datastax.com/en/astra-db-serverless/databases/embedding-generation.html).)
# 
# ### Explicit Embedding Initialization
# 
# Below, we instantiate our vector store using the explicit embedding class:
# 
# import EmbeddingTabs from "@theme/EmbeddingTabs";
# 
# <EmbeddingTabs/>
# 

# In[11]:


# | output: false
# | echo: false
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# In[22]:


from langchain_astradb import AstraDBVectorStore

vector_store = AstraDBVectorStore(
    collection_name="astra_vector_langchain",
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)


# ### Integrated Embedding Initialization
# 
# Here it is assumed that you have
# 
# - Enabled the OpenAI integration in your Astra DB organization,
# - Added an API Key named `"OPENAI_API_KEY"` to the integration, and scoped it to the database you are using.
# 
# For more details on how to do this, please consult the [documentation](https://docs.datastax.com/en/astra-db-serverless/integrations/embedding-providers/openai.html).

# In[ ]:


from astrapy.info import CollectionVectorServiceOptions

openai_vectorize_options = CollectionVectorServiceOptions(
    provider="openai",
    model_name="text-embedding-3-small",
    authentication={
        "providerKey": "OPENAI_API_KEY",
    },
)

vector_store_integrated = AstraDBVectorStore(
    collection_name="astra_vector_langchain_integrated",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
    collection_vector_service_options=openai_vectorize_options,
)


# ## Manage vector store
# 
# Once you have created your vector store, we can interact with it by adding and deleting different items.
# 
# ### Add items to vector store
# 
# We can add items to our vector store by using the `add_documents` function.

# In[23]:


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

vector_store.add_documents(documents=documents, ids=uuids)


# ### Delete items from vector store
# 
# We can delete items from our vector store by ID by using the `delete` function.

# In[24]:


vector_store.delete(ids=uuids[-1])


# ## Query vector store
# 
# Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent. 
# 
# ### Query directly
# 
# #### Similarity search
# 
# Performing a simple similarity search with filtering on metadata can be done as follows:

# In[15]:


results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")


# #### Similarity search with score
# 
# You can also search with score:

# In[16]:


results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")


# #### Other search methods
# 
# There are a variety of other search methods that are not covered in this notebook, such as MMR search or searching by vector. For a full list of the search abilities available for `AstraDBVectorStore` check out the [API reference](https://python.langchain.com/api_reference/astradb/vectorstores/langchain_astradb.vectorstores.AstraDBVectorStore.html).

# ### Query by turning into retriever
# 
# You can also transform the vector store into a retriever for easier usage in your chains. 
# 
# Here is how to transform your vector store into a retriever and then invoke the retreiever with a simple query and filter.

# In[17]:


retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})


# ## Usage for retrieval-augmented generation
# 
# For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:
# 
# - [Tutorials](/docs/tutorials/)
# - [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
# - [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

# For more, check out a complete RAG template using Astra DB [here](https://github.com/langchain-ai/langchain/tree/master/templates/rag-astradb).

# ## Cleanup vector store

# If you want to completely delete the collection from your Astra DB instance, run this.
# 
# _(You will lose the data you stored in it.)_

# In[ ]:


vector_store.delete_collection()


# ## API reference
# 
# For detailed documentation of all `AstraDBVectorStore` features and configurations head to the API reference: https://python.langchain.com/api_reference/astradb/vectorstores/langchain_astradb.vectorstores.AstraDBVectorStore.html
