#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Volc Enging Maas
---
# # VolcEngineMaasChat
# 
# This notebook provides you with a guide on how to get started with volc engine maas chat models.

# In[ ]:


# Install the package
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  volcengine')


# In[20]:


from langchain_community.chat_models import VolcEngineMaasChat
from langchain_core.messages import HumanMessage


# In[25]:


chat = VolcEngineMaasChat(volc_engine_maas_ak="your ak", volc_engine_maas_sk="your sk")


# or you can set access_key and secret_key in your environment variables
# ```bash
# export VOLC_ACCESSKEY=YOUR_AK
# export VOLC_SECRETKEY=YOUR_SK
# ```

# In[26]:


chat([HumanMessage(content="给我讲个笑话")])


# # volc engine maas chat with stream

# In[27]:


chat = VolcEngineMaasChat(
    volc_engine_maas_ak="your ak",
    volc_engine_maas_sk="your sk",
    streaming=True,
)


# In[28]:


chat([HumanMessage(content="给我讲个笑话")])

