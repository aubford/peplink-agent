#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Tencent Hunyuan
---
# # Tencent Hunyuan
# 
# >[Tencent's hybrid model API](https://cloud.tencent.com/document/product/1729) (`Hunyuan API`) 
# > implements dialogue communication, content generation, 
# > analysis and understanding, and can be widely used in various scenarios such as intelligent 
# > customer service, intelligent marketing, role playing, advertising copywriting, product description,
# > script creation, resume generation, article writing, code generation, data analysis, and content
# > analysis.
# 
# See for [more information](https://cloud.tencent.com/document/product/1729).

# In[1]:


from langchain_community.chat_models import ChatHunyuan
from langchain_core.messages import HumanMessage


# In[2]:


chat = ChatHunyuan(
    hunyuan_app_id=111111111,
    hunyuan_secret_id="YOUR_SECRET_ID",
    hunyuan_secret_key="YOUR_SECRET_KEY",
)


# In[3]:


chat(
    [
        HumanMessage(
            content="You are a helpful assistant that translates English to French.Translate this sentence from English to French. I love programming."
        )
    ]
)


# ## For ChatHunyuan with Streaming

# In[2]:


chat = ChatHunyuan(
    hunyuan_app_id="YOUR_APP_ID",
    hunyuan_secret_id="YOUR_SECRET_ID",
    hunyuan_secret_key="YOUR_SECRET_KEY",
    streaming=True,
)


# In[3]:


chat(
    [
        HumanMessage(
            content="You are a helpful assistant that translates English to French.Translate this sentence from English to French. I love programming."
        )
    ]
)


# In[ ]:




