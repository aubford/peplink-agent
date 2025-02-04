#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Llama API
---
# # ChatLlamaAPI
# 
# This notebook shows how to use LangChain with [LlamaAPI](https://llama-api.com/) - a hosted version of Llama2 that adds in support for function calling.

# %pip install --upgrade --quiet  llamaapi

# In[2]:


from llamaapi import LlamaAPI

# Replace 'Your_API_Token' with your actual API token
llama = LlamaAPI("Your_API_Token")


# In[4]:


from langchain_experimental.llms import ChatLlamaAPI


# In[5]:


model = ChatLlamaAPI(client=llama)


# In[6]:


from langchain.chains import create_tagging_chain

schema = {
    "properties": {
        "sentiment": {
            "type": "string",
            "description": "the sentiment encountered in the passage",
        },
        "aggressiveness": {
            "type": "integer",
            "description": "a 0-10 score of how aggressive the passage is",
        },
        "language": {"type": "string", "description": "the language of the passage"},
    }
}

chain = create_tagging_chain(schema, model)


# In[7]:


chain.run("give me your money")


# In[ ]:




