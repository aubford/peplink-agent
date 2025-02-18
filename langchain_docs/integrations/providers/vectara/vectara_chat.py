#!/usr/bin/env python
# coding: utf-8

# # Vectara Chat
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
# This notebook shows how to use Vectara's [Chat](https://docs.vectara.com/docs/api-reference/chat-apis/chat-apis-overview) functionality, which provides automatic storage of conversation history and ensures follow up questions consider that history.

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


# ## Vectara Chat Explained
#
# In most uses of LangChain to create chatbots, one must integrate a special `memory` component that maintains the history of chat sessions and then uses that history to ensure the chatbot is aware of conversation history.
#
# With Vectara Chat - all of that is performed in the backend by Vectara automatically. You can look at the [Chat](https://docs.vectara.com/docs/api-reference/chat-apis/chat-apis-overview) documentation for the details, to learn more about the internals of how this is implemented, but with LangChain all you have to do is turn that feature on in the Vectara vectorstore.
#
# Let's see an example. First we load the SOTU document (remember, text extraction and chunking all occurs automatically on the Vectara platform):

# In[2]:


from langchain_community.document_loaders import TextLoader

loader = TextLoader("state_of_the_union.txt")
documents = loader.load()

vectara = Vectara.from_documents(documents, embedding=None)


# And now we create a Chat Runnable using the `as_chat` method:

# In[3]:


summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
rerank_config = RerankConfig(reranker="mmr", rerank_k=50, mmr_diversity_bias=0.2)
config = VectaraQueryConfig(
    k=10, lambda_val=0.005, rerank_config=rerank_config, summary_config=summary_config
)

bot = vectara.as_chat(config)


# Here's an example of asking a question with no chat history

# In[4]:


bot.invoke("What did the president say about Ketanji Brown Jackson?")["answer"]


# Here's an example of asking a question with some chat history

# In[5]:


bot.invoke("Did he mention who she suceeded?")["answer"]


# ## Chat with streaming
#
# Of course the chatbot interface also supports streaming.
# Instead of the `invoke` method you simply use `stream`:

# In[6]:


output = {}
curr_key = None
for chunk in bot.stream("what about her accopmlishments?"):
    for key in chunk:
        if key not in output:
            output[key] = chunk[key]
        else:
            output[key] += chunk[key]
        if key == "answer":
            print(chunk[key], end="", flush=True)
        curr_key = key
