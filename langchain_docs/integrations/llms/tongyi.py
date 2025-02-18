#!/usr/bin/env python
# coding: utf-8

# # Tongyi Qwen
# Tongyi Qwen is a large-scale language model developed by Alibaba's Damo Academy. It is capable of understanding user intent through natural language understanding and semantic analysis, based on user input in natural language. It provides services and assistance to users in different domains and tasks. By providing clear and detailed instructions, you can obtain results that better align with your expectations.

# ## Setting up

# In[ ]:


# Install the package
get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  langchain-community dashscope"
)


# In[1]:


# Get a new token: https://help.aliyun.com/document_detail/611472.html?spm=a2c4g.2399481.0.0
from getpass import getpass

DASHSCOPE_API_KEY = getpass()


# In[2]:


import os

os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY


# In[3]:


from langchain_community.llms import Tongyi


# In[4]:


Tongyi().invoke("What NFL team won the Super Bowl in the year Justin Bieber was born?")


# ## Using in a chain

# In[5]:


from langchain_core.prompts import PromptTemplate


# In[6]:


llm = Tongyi()


# In[7]:


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


# In[8]:


chain = prompt | llm


# In[9]:


question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

chain.invoke({"question": question})
