#!/usr/bin/env python
# coding: utf-8

# # Kay.ai
#
# >[Kai Data API](https://www.kay.ai/) built for RAG 🕵️ We are curating the world's largest datasets as high-quality embeddings so your AI agents can retrieve context on the fly. Latest models, fast retrieval, and zero infra.
#
# This notebook shows you how to retrieve datasets supported by [Kay](https://kay.ai/). You can currently search `SEC Filings` and `Press Releases of US companies`. Visit [kay.ai](https://kay.ai) for the latest data drops. For any questions, join our [discord](https://discord.gg/hAnE4e5T6M) or [tweet at us](https://twitter.com/vishalrohra_)

# ## Installation
#
# First, install the [`kay` package](https://pypi.org/project/kay/).

# In[ ]:


get_ipython().system("pip install kay")


# You will also need an API key: you can get one for free at [https://kay.ai](https://kay.ai/). Once you have an API key, you must set it as an environment variable `KAY_API_KEY`.
#
# `KayAiRetriever` has a static `.create()` factory method that takes the following arguments:
#
# * `dataset_id: string` required -- A Kay dataset id. This is a collection of data about a particular entity such as companies, people, or places. For example, try `"company"`
# * `data_type: List[string]` optional -- This is a category within a  dataset based on its origin or format, such as ‘SEC Filings’, ‘Press Releases’, or ‘Reports’ within the “company” dataset. For example, try ["10-K", "10-Q", "PressRelease"] under the “company” dataset. If left empty, Kay will retrieve the most relevant context across all types.
# * `num_contexts: int` optional, defaults to 6 -- The number of document chunks to retrieve on each call to `get_relevant_documents()`

# ## Examples
#
# ### Basic Retriever Usage

# In[2]:


# Setup API key
from getpass import getpass

KAY_API_KEY = getpass()


# In[3]:


import os

from langchain_community.retrievers import KayAiRetriever

os.environ["KAY_API_KEY"] = KAY_API_KEY
retriever = KayAiRetriever.create(
    dataset_id="company", data_types=["10-K", "10-Q", "PressRelease"], num_contexts=3
)
docs = retriever.invoke(
    "What were the biggest strategy changes and partnerships made by Roku in 2023??"
)


# In[4]:


docs


# ### Usage in a chain

# In[5]:


OPENAI_API_KEY = getpass()


# In[6]:


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# In[16]:


from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)


# In[22]:


questions = [
    "What were the biggest strategy changes and partnerships made by Roku in 2023?"
    # "Where is Wex making the most money in 2023?",
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")
