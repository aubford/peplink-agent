#!/usr/bin/env python
# coding: utf-8

# # Vectara
# 
# [Vectara](https://vectara.com/) is the trusted AI Assistant and Agent platform which focuses on enterprise readiness for mission-critical applications.
# 
# Vectara serverless RAG-as-a-service provides all the components of RAG behind an easy-to-use API, including:
# 1. A way to extract text from files (PDF, PPT, DOCX, etc)
# 2. ML-based chunking that provides state of the art performance.
# 3. The [Boomerang](https://vectara.com/how-boomerang-takes-retrieval-augmented-generation-to-the-next-level-via-grounded-generation/) embeddings model.
# 4. Its own internal vector database where text chunks and embedding vectors are stored.
# 5. A query service that automatically encodes the query into embedding, and retrieves the most relevant text segments (including support for [Hybrid Search](https://docs.vectara.com/docs/api-reference/search-apis/lexical-matching) as well as multiple reranking options such as the [multi-lingual relevance reranker](https://www.vectara.com/blog/deep-dive-into-vectara-multilingual-reranker-v1-state-of-the-art-reranker-across-100-languages), [MMR](https://vectara.com/get-diverse-results-and-comprehensive-summaries-with-vectaras-mmr-reranker/), [UDF reranker](https://www.vectara.com/blog/rag-with-user-defined-functions-based-reranking). 
# 6. An LLM to for creating a [generative summary](https://docs.vectara.com/docs/learn/grounded-generation/grounded-generation-overview), based on the retrieved documents (context), including citations.
# 
# See the [Vectara API documentation](https://docs.vectara.com/docs/) for more information on how to use the API.
# 
# This notebook shows how to use the basic retrieval functionality, when utilizing Vectara just as a Vector Store (without summarization), incuding: `similarity_search` and `similarity_search_with_score` as well as using the LangChain `as_retriever` functionality.
# 
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

# # Getting Started
# 
# To get started, use the following steps:
# 1. If you don't already have one, [Sign up](https://www.vectara.com/integrations/langchain) for your free Vectara trial. Once you have completed your sign up you will have a Vectara customer ID. You can find your customer ID by clicking on your name, on the top-right of the Vectara console window.
# 2. Within your account you can create one or more corpora. Each corpus represents an area that stores text data upon ingest from input documents. To create a corpus, use the **"Create Corpus"** button. You then provide a name to your corpus as well as a description. Optionally you can define filtering attributes and apply some advanced options. If you click on your created corpus, you can see its name and corpus ID right on the top.
# 3. Next you'll need to create API keys to access the corpus. Click on the **"Access Control"** tab in the corpus view and then the **"Create API Key"** button. Give your key a name, and choose whether you want query-only or query+index for your key. Click "Create" and you now have an active API key. Keep this key confidential. 
# 
# To use LangChain with Vectara, you'll need to have these three values: `customer ID`, `corpus ID` and `api_key`.
# You can provide those to LangChain in two ways:
# 
# 1. Include in your environment these three variables: `VECTARA_CUSTOMER_ID`, `VECTARA_CORPUS_ID` and `VECTARA_API_KEY`.
# 
#    For example, you can set these variables using os.environ and getpass as follows:
# 
# ```python
# import os
# import getpass
# 
# os.environ["VECTARA_CUSTOMER_ID"] = getpass.getpass("Vectara Customer ID:")
# os.environ["VECTARA_CORPUS_ID"] = getpass.getpass("Vectara Corpus ID:")
# os.environ["VECTARA_API_KEY"] = getpass.getpass("Vectara API Key:")
# ```
# 
# 2. Add them to the `Vectara` vectorstore constructor:
# 
# ```python
# vectara = Vectara(
#                 vectara_customer_id=vectara_customer_id,
#                 vectara_corpus_id=vectara_corpus_id,
#                 vectara_api_key=vectara_api_key
#             )
# ```
# 
# In this notebook we assume they are provided in the environment.

# In[1]:


import os

os.environ["VECTARA_API_KEY"] = "<YOUR_VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_ID"] = "<YOUR_VECTARA_CORPUS_ID>"
os.environ["VECTARA_CUSTOMER_ID"] = "<YOUR_VECTARA_CUSTOMER_ID>"

from langchain_community.vectorstores import Vectara
from langchain_community.vectorstores.vectara import (
    RerankConfig,
    SummaryConfig,
    VectaraQueryConfig,
)


# First we load the state-of-the-union text into Vectara. 
# 
# Note that we use the `from_files` interface which does not require any local processing or chunking - Vectara receives the file content and performs all the necessary pre-processing, chunking and embedding of the file into its knowledge store.
# 
# In this case it uses a `.txt` file but the same works for many other [file types](https://docs.vectara.com/docs/api-reference/indexing-apis/file-upload/file-upload-filetypes).

# In[2]:


vectara = Vectara.from_files(["state_of_the_union.txt"])


# ## Basic Vectara RAG (retrieval augmented generation)
# 
# We now create a `VectaraQueryConfig` object to control the retrieval and summarization options:
# * We enable summarization, specifying we would like the LLM to pick the top 7 matching chunks and respond in English
# * We enable MMR (max marginal relevance) in the retrieval process, with a 0.2 diversity bias factor
# * We want the top-10 results, with hybrid search configured with a value of 0.025
# 
# Using this configuration, let's create a LangChain `Runnable` object that encpasulates the full Vectara RAG pipeline, using the `as_rag` method:

# In[3]:


summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
rerank_config = RerankConfig(reranker="mmr", rerank_k=50, mmr_diversity_bias=0.2)
config = VectaraQueryConfig(
    k=10, lambda_val=0.005, rerank_config=rerank_config, summary_config=summary_config
)

query_str = "what did Biden say?"

rag = vectara.as_rag(config)
rag.invoke(query_str)["answer"]


# We can also use the streaming interface like this:

# In[4]:


output = {}
curr_key = None
for chunk in rag.stream(query_str):
    for key in chunk:
        if key not in output:
            output[key] = chunk[key]
        else:
            output[key] += chunk[key]
        if key == "answer":
            print(chunk[key], end="", flush=True)
        curr_key = key


# ## Hallucination detection and Factual Consistency Score
# 
# Vectara created [HHEM](https://huggingface.co/vectara/hallucination_evaluation_model) - an open source model that can be used to evaluate RAG responses for factual consistency. 
# 
# As part of the Vectara RAG, the "Factual Consistency Score" (or FCS), which is an improved version of the open source HHEM is made available via the API. This is automatically included in the output of the RAG pipeline

# In[5]:


summary_config = SummaryConfig(is_enabled=True, max_results=5, response_lang="eng")
rerank_config = RerankConfig(reranker="mmr", rerank_k=50, mmr_diversity_bias=0.1)
config = VectaraQueryConfig(
    k=10, lambda_val=0.005, rerank_config=rerank_config, summary_config=summary_config
)

rag = vectara.as_rag(config)
resp = rag.invoke(query_str)
print(resp["answer"])
print(f"Vectara FCS = {resp['fcs']}")


# ## Vectara as a langchain retreiver
# 
# The Vectara component can also be used just as a retriever. 
# 
# In this case, it behaves just like any other LangChain retriever. The main use of this mode is for semantic search, and in this case we disable summarization:

# In[6]:


config.summary_config.is_enabled = False
config.k = 3
retriever = vectara.as_retriever(config=config)
retriever.invoke(query_str)


# For backwards compatibility, you can also enable summarization with a retriever, in which case the summary is added as an additional Document object:

# In[7]:


config.summary_config.is_enabled = True
config.k = 3
retriever = vectara.as_retriever(config=config)
retriever.invoke(query_str)


# ## Advanced LangChain query pre-processing with Vectara
# 
# Vectara's "RAG as a service" does a lot of the heavy lifting in creating question answering or chatbot chains. The integration with LangChain provides the option to use additional capabilities such as query pre-processing  like `SelfQueryRetriever` or `MultiQueryRetriever`. Let's look at an example of using the [MultiQueryRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever).
# 
# Since MQR uses an LLM we have to set that up - here we choose `ChatOpenAI`:

# In[8]:


from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)


def get_summary(documents):
    return documents[-1].page_content


(mqr | get_summary).invoke(query_str)


# In[ ]:




