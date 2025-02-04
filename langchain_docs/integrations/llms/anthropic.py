#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Anthropic
sidebar_class_name: hidden
---
# # AnthropicLLM
# 
# :::caution
# You are currently on a page documenting the use of Anthropic legacy Claude 2 models as [text completion models](/docs/concepts/text_llms). The latest and most popular Anthropic models are [chat completion models](/docs/concepts/chat_models), and the text completion models have been deprecated.
# 
# You are probably looking for [this page instead](/docs/integrations/chat/anthropic/).
# :::
# 
# This example goes over how to use LangChain to interact with `Anthropic` models.
# 
# ## Installation

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-anthropic')


# ## Environment Setup
# 
# We'll need to get an [Anthropic](https://console.anthropic.com/settings/keys) API key and set the `ANTHROPIC_API_KEY` environment variable:

# In[4]:


import os
from getpass import getpass

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass()


# ## Usage

# In[6]:


from langchain_anthropic import AnthropicLLM
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

model = AnthropicLLM(model="claude-2.1")

chain = prompt | model

chain.invoke({"question": "What is LangChain?"})

