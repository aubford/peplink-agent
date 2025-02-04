#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Weaviate
---
# # Weaviate
# 
# This notebook covers how to get started with the Weaviate vector store in LangChain, using the `langchain-weaviate` package.
# 
# > [Weaviate](https://weaviate.io/) is an open-source vector database. It allows you to store data objects and vector embeddings from your favorite ML-models, and scale seamlessly into billions of data objects.
# 
# To use this integration, you need to have a running Weaviate database instance.
# 
# ## Minimum versions
# 
# This module requires Weaviate `1.23.7` or higher. However, we recommend you use the latest version of Weaviate.
# 
# ## Connecting to Weaviate
# 
# In this notebook, we assume that you have a local instance of Weaviate running on `http://localhost:8080` and port 50051 open for [gRPC traffic](https://weaviate.io/blog/grpc-performance-improvements). So, we will connect to Weaviate with:
# 
# ```python
# weaviate_client = weaviate.connect_to_local()
# ```
# 
# ### Other deployment options
# 
# Weaviate can be [deployed in many different ways](https://weaviate.io/developers/weaviate/starter-guides/which-weaviate) such as using [Weaviate Cloud Services (WCS)](https://console.weaviate.cloud), [Docker](https://weaviate.io/developers/weaviate/installation/docker-compose) or [Kubernetes](https://weaviate.io/developers/weaviate/installation/kubernetes). 
# 
# If your Weaviate instance is deployed in another way, [read more here](https://weaviate.io/developers/weaviate/client-libraries/python#instantiate-a-client) about different ways to connect to Weaviate. You can use different [helper functions](https://weaviate.io/developers/weaviate/client-libraries/python#python-client-v4-helper-functions) or [create a custom instance](https://weaviate.io/developers/weaviate/client-libraries/python#python-client-v4-explicit-connection).
# 
# > Note that you require a `v4` client API, which will create a `weaviate.WeaviateClient` object.
# 
# ### Authentication
# 
# Some Weaviate instances, such as those running on WCS, have authentication enabled, such as API key and/or username+password authentication.
# 
# Read the [client authentication guide](https://weaviate.io/developers/weaviate/client-libraries/python#authentication) for more information, as well as the [in-depth authentication configuration page](https://weaviate.io/developers/weaviate/configuration/authentication).

# ## Installation

# In[1]:


# install package
# %pip install -Uqq langchain-weaviate
# %pip install openai tiktoken langchain


# ## Environment Setup
# 
# This notebook uses the OpenAI API through `OpenAIEmbeddings`. We suggest obtaining an OpenAI API key and export it as an environment variable with the name `OPENAI_API_KEY`.
# 
# Once this is done, your OpenAI API key will be read automatically. If you are new to environment variables, read more about them [here](https://docs.python.org/3/library/os.html#os.environ) or in [this guide](https://www.twilio.com/en-us/blog/environment-variables-python).

# # Usage

# ## Find objects by similarity

# Here is an example of how to find objects by similarity to a query, from data import to querying the Weaviate instance.
# 
# ### Step 1: Data import
# 
# First, we will create data to add to `Weaviate` by loading and chunking the contents of a long text file. 

# In[2]:


from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# In[3]:


loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# Now, we can import the data. 
# 
# To do so, connect to the Weaviate instance and use the resulting `weaviate_client` object. For example, we can import the documents as shown below:

# In[4]:


import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore


# In[5]:


weaviate_client = weaviate.connect_to_local()
db = WeaviateVectorStore.from_documents(docs, embeddings, client=weaviate_client)


# ### Step 2: Perform the search

# We can now perform a similarity search. This will return the most similar documents to the query text, based on the embeddings stored in Weaviate and an equivalent embedding generated from the query text.

# In[6]:


query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

# Print the first 100 characters of each result
for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}:")
    print(doc.page_content[:100] + "...")


# You can also add filters, which will either include or exclude results based on the filter conditions. (See [more filter examples](https://weaviate.io/developers/weaviate/search/filters).)

# In[7]:


from weaviate.classes.query import Filter

for filter_str in ["blah.txt", "state_of_the_union.txt"]:
    search_filter = Filter.by_property("source").equal(filter_str)
    filtered_search_results = db.similarity_search(query, filters=search_filter)
    print(len(filtered_search_results))
    if filter_str == "state_of_the_union.txt":
        assert len(filtered_search_results) > 0  # There should be at least one result
    else:
        assert len(filtered_search_results) == 0  # There should be no results


# It is also possible to provide `k`, which is the upper limit of the number of results to return.

# In[8]:


search_filter = Filter.by_property("source").equal("state_of_the_union.txt")
filtered_search_results = db.similarity_search(query, filters=search_filter, k=3)
assert len(filtered_search_results) <= 3


# ### Quantify result similarity

# You can optionally retrieve a relevance "score". This is a relative score that indicates how good the particular search results is, amongst the pool of search results. 
# 
# Note that this is relative score, meaning that it should not be used to determine thresholds for relevance. However, it can be used to compare the relevance of different search results within the entire search result set.

# In[9]:


docs = db.similarity_search_with_score("country", k=5)

for doc in docs:
    print(f"{doc[1]:.3f}", ":", doc[0].page_content[:100] + "...")


# ## Search mechanism

# `similarity_search` uses Weaviate's [hybrid search](https://weaviate.io/developers/weaviate/api/graphql/search-operators#hybrid).
# 
# A hybrid search combines a vector and a keyword search, with `alpha` as the weight of the vector search. The `similarity_search` function allows you to pass additional arguments as kwargs. See this [reference doc](https://weaviate.io/developers/weaviate/api/graphql/search-operators#hybrid) for the available arguments.
# 
# So, you can perform a pure keyword search by adding `alpha=0` as shown below:

# In[10]:


docs = db.similarity_search(query, alpha=0)
docs[0]


# ## Persistence

# Any data added through `langchain-weaviate` will persist in Weaviate according to its configuration. 
# 
# WCS instances, for example, are configured to persist data indefinitely, and Docker instances can be set up to persist data in a volume. Read more about [Weaviate's persistence](https://weaviate.io/developers/weaviate/configuration/persistence).

# ## Multi-tenancy

# [Multi-tenancy](https://weaviate.io/developers/weaviate/concepts/data#multi-tenancy) allows you to have a high number of isolated collections of data, with the same collection configuration, in a single Weaviate instance. This is great for multi-user environments such as building a SaaS app, where each end user will have their own isolated data collection.
# 
# To use multi-tenancy, the vector store need to be aware of the `tenant` parameter. 
# 
# So when adding any data, provide the `tenant` parameter as shown below.

# In[11]:


db_with_mt = WeaviateVectorStore.from_documents(
    docs, embeddings, client=weaviate_client, tenant="Foo"
)


# And when performing queries, provide the `tenant` parameter also.

# In[12]:


db_with_mt.similarity_search(query, tenant="Foo")


# ## Retriever options

# Weaviate can also be used as a retriever

# ### Maximal marginal relevance search (MMR)

# In addition to using similaritysearch  in the retriever object, you can also use `mmr`.

# In[13]:


retriever = db.as_retriever(search_type="mmr")
retriever.invoke(query)[0]


# # Use with LangChain

# A known limitation of large language models (LLMs) is that their training data can be outdated, or not include the specific domain knowledge that you require.
# 
# Take a look at the example below:

# In[14]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm.predict("What did the president say about Justice Breyer")


# Vector stores complement LLMs by providing a way to store and retrieve relevant information. This allow you to combine the strengths of LLMs and vector stores, by using LLM's reasoning and linguistic capabilities with vector stores' ability to retrieve relevant information.
# 
# Two well-known applications for combining LLMs and vector stores are:
# - Question answering
# - Retrieval-augmented generation (RAG)

# ### Question Answering with Sources

# Question answering in langchain can be enhanced by the use of vector stores. Let's see how this can be done.
# 
# This section uses the `RetrievalQAWithSourcesChain`, which does the lookup of the documents from an Index. 
# 
# First, we will chunk the text again and import them into the Weaviate vector store.

# In[15]:


from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI


# In[16]:


with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)


# In[17]:


docsearch = WeaviateVectorStore.from_texts(
    texts,
    embeddings,
    client=weaviate_client,
    metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
)


# Now we can construct the chain, with the retriever specified:

# In[18]:


chain = RetrievalQAWithSourcesChain.from_chain_type(
    OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever()
)


# And run the chain, to ask the question:

# In[19]:


chain(
    {"question": "What did the president say about Justice Breyer"},
    return_only_outputs=True,
)


# ### Retrieval-Augmented Generation
# 
# Another very popular application of combining LLMs and vector stores is retrieval-augmented generation (RAG). This is a technique that uses a retriever to find relevant information from a vector store, and then uses an LLM to provide an output based on the retrieved data and a prompt.
# 
# We begin with a similar setup:

# In[20]:


with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)


# In[21]:


docsearch = WeaviateVectorStore.from_texts(
    texts,
    embeddings,
    client=weaviate_client,
    metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
)

retriever = docsearch.as_retriever()


# We need to construct a template for the RAG model so that the retrieved information will be populated in the template.

# In[22]:


from langchain_core.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)


# In[23]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# And running the cell, we get a very similar output.

# In[24]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What did the president say about Justice Breyer")


# But note that since the template is upto you to construct, you can customize it to your needs. 

# ### Wrap-up & resources
# 
# Weaviate is a scalable, production-ready vector store. 
# 
# This integration allows Weaviate to be used with LangChain to enhance the capabilities of large language models with a robust data store. Its scalability and production-readiness make it a great choice as a vector store for your LangChain applications, and it will reduce your time to production.
