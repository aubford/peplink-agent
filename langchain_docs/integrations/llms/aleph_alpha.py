#!/usr/bin/env python
# coding: utf-8

# # Aleph Alpha
# 
# [The Luminous series](https://docs.aleph-alpha.com/docs/category/luminous/) is a family of large language models.
# 
# This example goes over how to use LangChain to interact with Aleph Alpha models

# In[ ]:


# Installing the langchain package needed to use the integration
get_ipython().run_line_magic('pip', 'install -qU langchain-community')


# In[ ]:


# Install the package
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  aleph-alpha-client')


# In[1]:


# create a new token: https://docs.aleph-alpha.com/docs/account/#create-a-new-token

from getpass import getpass

ALEPH_ALPHA_API_KEY = getpass()


# In[2]:


from langchain_community.llms import AlephAlpha
from langchain_core.prompts import PromptTemplate


# In[3]:


template = """Q: {question}

A:"""

prompt = PromptTemplate.from_template(template)


# In[4]:


llm = AlephAlpha(
    model="luminous-extended",
    maximum_tokens=20,
    stop_sequences=["Q:"],
    aleph_alpha_api_key=ALEPH_ALPHA_API_KEY,
)


# In[5]:


llm_chain = prompt | llm


# In[8]:


question = "What is AI?"

llm_chain.invoke({"question": question})


# In[ ]:




