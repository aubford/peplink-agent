#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Coze Chat
---
# # Chat with Coze Bot
# 
# ChatCoze chat models API by coze.com. For more information, see [https://www.coze.com/open/docs/chat](https://www.coze.com/open/docs/chat)

# In[1]:


from langchain_community.chat_models import ChatCoze
from langchain_core.messages import HumanMessage


# In[2]:


chat = ChatCoze(
    coze_api_base="YOUR_API_BASE",
    coze_api_key="YOUR_API_KEY",
    bot_id="YOUR_BOT_ID",
    user="YOUR_USER_ID",
    conversation_id="YOUR_CONVERSATION_ID",
    streaming=False,
)


# Alternatively, you can set your API key and API base with:

# In[ ]:


import os

os.environ["COZE_API_KEY"] = "YOUR_API_KEY"
os.environ["COZE_API_BASE"] = "YOUR_API_BASE"


# In[3]:


chat([HumanMessage(content="什么是扣子(coze)")])


# ## Chat with Coze Streaming

# In[5]:


chat = ChatCoze(
    coze_api_base="YOUR_API_BASE",
    coze_api_key="YOUR_API_KEY",
    bot_id="YOUR_BOT_ID",
    user="YOUR_USER_ID",
    conversation_id="YOUR_CONVERSATION_ID",
    streaming=True,
)


# In[6]:


chat([HumanMessage(content="什么是扣子(coze)")])

