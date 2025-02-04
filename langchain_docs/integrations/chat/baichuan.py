#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Baichuan Chat
---
# # Chat with Baichuan-192K
# 
# Baichuan chat models API by Baichuan Intelligent Technology. For more information, see [https://platform.baichuan-ai.com/docs/api](https://platform.baichuan-ai.com/docs/api)

# In[1]:


from langchain_community.chat_models import ChatBaichuan
from langchain_core.messages import HumanMessage


# In[2]:


chat = ChatBaichuan(baichuan_api_key="YOUR_API_KEY")


# Alternatively, you can set your API key with:

# In[ ]:


import os

os.environ["BAICHUAN_API_KEY"] = "YOUR_API_KEY"


# In[3]:


chat([HumanMessage(content="我日薪8块钱，请问在闰年的二月，我月薪多少")])


# ## Chat with Baichuan-192K with Streaming

# In[5]:


chat = ChatBaichuan(
    baichuan_api_key="YOUR_API_KEY",
    streaming=True,
)


# In[6]:


chat([HumanMessage(content="我日薪8块钱，请问在闰年的二月，我月薪多少")])

