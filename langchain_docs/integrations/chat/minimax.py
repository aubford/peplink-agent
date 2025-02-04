#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: MiniMax
---
# # MiniMaxChat
# 
# [Minimax](https://api.minimax.chat) is a Chinese startup that provides LLM service for companies and individuals.
# 
# This example goes over how to use LangChain to interact with MiniMax Inference for Chat.

# In[ ]:


import os

os.environ["MINIMAX_GROUP_ID"] = "MINIMAX_GROUP_ID"
os.environ["MINIMAX_API_KEY"] = "MINIMAX_API_KEY"


# In[ ]:


from langchain_community.chat_models import MiniMaxChat
from langchain_core.messages import HumanMessage


# In[ ]:


chat = MiniMaxChat()


# In[ ]:


chat(
    [
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        )
    ]
)

