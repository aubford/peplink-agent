#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Perplexity
---
# # ChatPerplexity
# 
# This notebook covers how to get started with `Perplexity` chat models.

# In[1]:


from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate


# The code provided assumes that your PPLX_API_KEY is set in your environment variables. If you would like to manually specify your API key and also choose a different model, you can use the following code:

# In[ ]:


chat = ChatPerplexity(
    temperature=0, pplx_api_key="YOUR_API_KEY", model="llama-3-sonar-small-32k-online"
)


# The code provided assumes that your PPLX_API_KEY is set in your environment variables. If you would like to manually specify your API key and also choose a different model, you can use the following code:
# 
# ```python
# chat = ChatPerplexity(temperature=0, pplx_api_key="YOUR_API_KEY", model="llama-3.1-sonar-small-128k-online")
# ```
# 
# You can check a list of available models [here](https://docs.perplexity.ai/docs/model-cards). For reproducibility, we can set the API key dynamically by taking it as an input in this notebook.

# In[2]:


import os
from getpass import getpass

PPLX_API_KEY = getpass()
os.environ["PPLX_API_KEY"] = PPLX_API_KEY


# In[3]:


chat = ChatPerplexity(temperature=0, model="llama-3.1-sonar-small-128k-online")


# In[4]:


system = "You are a helpful assistant."
human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
response = chain.invoke({"input": "Why is the Higgs Boson important?"})
response.content


# You can format and structure the prompts like you would typically. In the following example, we ask the model to tell us a joke about cats.

# In[5]:


chat = ChatPerplexity(temperature=0, model="llama-3.1-sonar-small-128k-online")
prompt = ChatPromptTemplate.from_messages([("human", "Tell me a joke about {topic}")])
chain = prompt | chat
response = chain.invoke({"topic": "cats"})
response.content


# ## `ChatPerplexity` also supports streaming functionality:

# In[6]:


chat = ChatPerplexity(temperature=0.7, model="llama-3.1-sonar-small-128k-online")
prompt = ChatPromptTemplate.from_messages(
    [("human", "Give me a list of famous tourist attractions in Pakistan")]
)
chain = prompt | chat
for chunk in chain.stream({}):
    print(chunk.content, end="", flush=True)

