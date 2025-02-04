#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Upstage
---
# # ChatUpstage
# 
# This notebook covers how to get started with Upstage chat models.
# 
# ## Installation
# 
# Install `langchain-upstage` package.
# 
# ```bash
# pip install -U langchain-upstage
# ```

# ## Environment Setup
# 
# Make sure to set the following environment variables:
# 
# - `UPSTAGE_API_KEY`: Your Upstage API key from [Upstage console](https://console.upstage.ai/).
# 
# ## Usage

# In[ ]:


import os

os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"


# In[ ]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage

chat = ChatUpstage()


# In[ ]:


# using chat invoke
chat.invoke("Hello, how are you?")


# In[ ]:


# using chat stream
for m in chat.stream("Hello, how are you?"):
    print(m)


# ## Chaining

# In[ ]:


# using chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates English to French."),
        ("human", "Translate this sentence from English to French. {english_text}."),
    ]
)
chain = prompt | chat

chain.invoke({"english_text": "Hello, how are you?"})

