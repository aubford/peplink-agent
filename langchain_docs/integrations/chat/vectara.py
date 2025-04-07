#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# [Vectara](https://vectara.com/) is the trusted AI Assistant and Agent platform which focuses on enterprise readiness for mission-critical applications.
# Vectara serverless RAG-as-a-service provides all the components of RAG behind an easy-to-use API, including:
# 1. A way to extract text from files (PDF, PPT, DOCX, etc)
# 2. ML-based chunking that provides state of the art performance.
# 3. The [Boomerang](https://vectara.com/how-boomerang-takes-retrieval-augmented-generation-to-the-next-level-via-grounded-generation/) embeddings model.
# 4. Its own internal vector database where text chunks and embedding vectors are stored.
# 5. A query service that automatically encodes the query into embedding, and retrieves the most relevant text segments, including support for [Hybrid Search](https://docs.vectara.com/docs/api-reference/search-apis/lexical-matching) as well as multiple reranking options such as the [multi-lingual relevance reranker](https://www.vectara.com/blog/deep-dive-into-vectara-multilingual-reranker-v1-state-of-the-art-reranker-across-100-languages), [MMR](https://vectara.com/get-diverse-results-and-comprehensive-summaries-with-vectaras-mmr-reranker/), [UDF reranker](https://www.vectara.com/blog/rag-with-user-defined-functions-based-reranking). 
# 6. An LLM to for creating a [generative summary](https://docs.vectara.com/docs/learn/grounded-generation/grounded-generation-overview), based on the retrieved documents (context), including citations.
# 
# For more information:
# - [Documentation](https://docs.vectara.com/docs/)
# - [API Playground](https://docs.vectara.com/docs/rest-api/)
# - [Quickstart](https://docs.vectara.com/docs/quickstart)
# 
# 
# This notebook shows how to use Vectara's [Chat](https://docs.vectara.com/docs/api-reference/chat-apis/chat-apis-overview) functionality, which provides automatic storage of conversation history and ensures follow up questions consider that history.
# 
# ### Setup
# 
# To use the `VectaraVectorStore` you first need to install the partner package.
# 

# In[ ]:


get_ipython().system('uv pip install -U pip && uv pip install -qU langchain-vectara')


# ## Getting Started
# 
# To get started, use the following steps:
# 1. If you don't already have one, [Sign up](https://www.vectara.com/integrations/langchain) for your free Vectara trial.
# 2. Within your account you can create one or more corpora. Each corpus represents an area that stores text data upon ingest from input documents. To create a corpus, use the **"Create Corpus"** button. You then provide a name to your corpus as well as a description. Optionally you can define filtering attributes and apply some advanced options. If you click on your created corpus, you can see its name and corpus ID right on the top.
# 3. Next you'll need to create API keys to access the corpus. Click on the **"Access Control"** tab in the corpus view and then the **"Create API Key"** button. Give your key a name, and choose whether you want query-only or query+index for your key. Click "Create" and you now have an active API key. Keep this key confidential. 
# 
# To use LangChain with Vectara, you'll need to have these two values: `corpus_key` and `api_key`.
# You can provide `VECTARA_API_KEY` to LangChain in two ways:
# 
# ## Instantiation
# 
# 1. Include in your environment these two variables: `VECTARA_API_KEY`.
# 
#    For example, you can set these variables using os.environ and getpass as follows:
# 
# ```python
# import os
# import getpass
# 
# os.environ["VECTARA_API_KEY"] = getpass.getpass("Vectara API Key:")
# ```
# 
# 2. Add them to the `Vectara` vectorstore constructor:
# 
# ```python
# vectara = Vectara(
#     vectara_api_key=vectara_api_key
# )
# ```
# 
# In this notebook we assume they are provided in the environment.

# In[2]:


import os

os.environ["VECTARA_API_KEY"] = "<VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_KEY"] = "<VECTARA_CORPUS_KEY>"

from langchain_vectara import Vectara
from langchain_vectara.vectorstores import (
    CorpusConfig,
    GenerationConfig,
    MmrReranker,
    SearchConfig,
    VectaraQueryConfig,
)


# ## Vectara Chat Explained
# 
# In most uses of LangChain to create chatbots, one must integrate a special `memory` component that maintains the history of chat sessions and then uses that history to ensure the chatbot is aware of conversation history.
# 
# With Vectara Chat - all of that is performed in the backend by Vectara automatically. You can look at the [Chat](https://docs.vectara.com/docs/api-reference/chat-apis/chat-apis-overview) documentation for the details, to learn more about the internals of how this is implemented, but with LangChain all you have to do is turn that feature on in the Vectara vectorstore.
# 
# Let's see an example. First we load the SOTU document (remember, text extraction and chunking all occurs automatically on the Vectara platform):

# In[3]:


from langchain_community.document_loaders import TextLoader

loader = TextLoader("../document_loaders/example_data/state_of_the_union.txt")
documents = loader.load()

corpus_key = os.getenv("VECTARA_CORPUS_KEY")
vectara = Vectara.from_documents(documents, embedding=None, corpus_key=corpus_key)


# And now we create a Chat Runnable using the `as_chat` method:

# In[4]:


generation_config = GenerationConfig(
    max_used_search_results=7,
    response_language="eng",
    generation_preset_name="vectara-summary-ext-24-05-med-omni",
    enable_factual_consistency_score=True,
)
search_config = SearchConfig(
    corpora=[CorpusConfig(corpus_key=corpus_key, limit=25)],
    reranker=MmrReranker(diversity_bias=0.2),
)

config = VectaraQueryConfig(
    search=search_config,
    generation=generation_config,
)


bot = vectara.as_chat(config)


# 
# ## Invocation
# 
# Here's an example of asking a question with no chat history

# In[5]:


bot.invoke("What did the president say about Ketanji Brown Jackson?")["answer"]


# Here's an example of asking a question with some chat history

# In[6]:


bot.invoke("Did he mention who she suceeded?")["answer"]


# ## Chat with streaming
# 
# Of course the chatbot interface also supports streaming.
# Instead of the `invoke` method you simply use `stream`:

# In[7]:


output = {}
curr_key = None
for chunk in bot.stream("what did he said about the covid?"):
    for key in chunk:
        if key not in output:
            output[key] = chunk[key]
        else:
            output[key] += chunk[key]
        if key == "answer":
            print(chunk[key], end="", flush=True)
        curr_key = key


# ## Chaining
# 
# For additional capabilities you can use chaining.

# In[33]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that explains the stuff to a five year old.  Vectara is providing the answer.",
        ),
        ("human", "{vectara_response}"),
    ]
)


def get_vectara_response(question: dict) -> str:
    """
    Calls Vectara as_chat and returns the answer string.  This encapsulates
    the Vectara call.
    """
    try:
        response = bot.invoke(question["question"])
        return response["answer"]
    except Exception as e:
        return "I'm sorry, I couldn't get an answer from Vectara."


# Create the chain
chain = get_vectara_response | prompt | llm | StrOutputParser()


# Invoke the chain
result = chain.invoke({"question": "what did he say about the covid?"})
print(result)


# ## API reference
# 
# You can look at the [Chat](https://docs.vectara.com/docs/api-reference/chat-apis/chat-apis-overview) documentation for the details.
