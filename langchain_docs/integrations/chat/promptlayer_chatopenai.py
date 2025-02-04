#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: PromptLayer ChatOpenAI
---
# # PromptLayerChatOpenAI
# 
# This example showcases how to connect to [PromptLayer](https://www.promptlayer.com) to start recording your ChatOpenAI requests.

# ## Install PromptLayer
# The `promptlayer` package is required to use PromptLayer with OpenAI. Install `promptlayer` using pip.

# In[ ]:


pip install promptlayer


# ## Imports

# In[2]:


import os

from langchain_community.chat_models import PromptLayerChatOpenAI
from langchain_core.messages import HumanMessage


# ## Set the Environment API Key
# You can create a PromptLayer API Key at [www.promptlayer.com](https://www.promptlayer.com) by clicking the settings cog in the navbar.
# 
# Set it as an environment variable called `PROMPTLAYER_API_KEY`.

# In[5]:


os.environ["PROMPTLAYER_API_KEY"] = "**********"


# ## Use the PromptLayerOpenAI LLM like normal
# *You can optionally pass in `pl_tags` to track your requests with PromptLayer's tagging feature.*

# In[4]:


chat = PromptLayerChatOpenAI(pl_tags=["langchain"])
chat([HumanMessage(content="I am a cat and I want")])


# **The above request should now appear on your [PromptLayer dashboard](https://www.promptlayer.com).**

# ## Using PromptLayer Track
# If you would like to use any of the [PromptLayer tracking features](https://magniv.notion.site/Track-4deee1b1f7a34c1680d085f82567dab9), you need to pass the argument `return_pl_id` when instantiating the PromptLayer LLM to get the request id.  

# In[ ]:


import promptlayer

chat = PromptLayerChatOpenAI(return_pl_id=True)
chat_results = chat.generate([[HumanMessage(content="I am a cat and I want")]])

for res in chat_results.generations:
    pl_request_id = res[0].generation_info["pl_request_id"]
    promptlayer.track.score(request_id=pl_request_id, score=100)


# Using this allows you to track the performance of your model in the PromptLayer dashboard. If you are using a prompt template, you can attach a template to a request as well.
# Overall, this gives you the opportunity to track the performance of different templates and models in the PromptLayer dashboard.
