#!/usr/bin/env python
# coding: utf-8

# # Rememberizer
#
# >[Rememberizer](https://rememberizer.ai/) is a knowledge enhancement service for AI applications created by  SkyDeck AI Inc.
#
# This notebook shows how to retrieve documents from `Rememberizer` into the Document format that is used downstream.

# # Preparation
#
# You will need an API key: you can get one after creating a common knowledge at [https://rememberizer.ai](https://rememberizer.ai/). Once you have an API key, you must set it as an environment variable `REMEMBERIZER_API_KEY` or pass it as `rememberizer_api_key` when initializing `RememberizerRetriever`.
#
# `RememberizerRetriever` has these arguments:
# - optional `top_k_results`: default=10. Use it to limit number of returned documents.
# - optional `rememberizer_api_key`: required if you don't set the environment variable `REMEMBERIZER_API_KEY`.
#
# `get_relevant_documents()` has one argument, `query`: free text which used to find documents in the common knowledge of `Rememberizer.ai`

# # Examples
#
# ## Basic usage

# In[1]:


# Setup API key
from getpass import getpass

REMEMBERIZER_API_KEY = getpass()


# In[2]:


import os

from langchain_community.retrievers import RememberizerRetriever

os.environ["REMEMBERIZER_API_KEY"] = REMEMBERIZER_API_KEY
retriever = RememberizerRetriever(top_k_results=5)


# In[3]:


docs = retriever.get_relevant_documents(query="How does Large Language Models works?")


# In[4]:


docs[0].metadata  # meta-information of the Document


# In[5]:


print(docs[0].page_content[:400])  # a content of the Document


# # Usage in a chain

# In[6]:


OPENAI_API_KEY = getpass()


# In[7]:


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# In[8]:


from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model_name="gpt-3.5-turbo")
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)


# In[9]:


questions = [
    "What is RAG?",
    "How does Large Language Models works?",
]
chat_history = []

for question in questions:
    result = qa.invoke({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")


# In[ ]:
