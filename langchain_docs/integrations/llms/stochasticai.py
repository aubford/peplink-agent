#!/usr/bin/env python
# coding: utf-8

# # StochasticAI
# 
# >[Stochastic Acceleration Platform](https://docs.stochastic.ai/docs/introduction/) aims to simplify the life cycle of a Deep Learning model. From uploading and versioning the model, through training, compression and acceleration to putting it into production.
# 
# This example goes over how to use LangChain to interact with `StochasticAI` models.

# You have to get the API_KEY and the API_URL [here](https://app.stochastic.ai/workspace/profile/settings?tab=profile).

# In[3]:


from getpass import getpass

STOCHASTICAI_API_KEY = getpass()


# In[4]:


import os

os.environ["STOCHASTICAI_API_KEY"] = STOCHASTICAI_API_KEY


# In[10]:


YOUR_API_URL = getpass()


# In[5]:


from langchain.chains import LLMChain
from langchain_community.llms import StochasticAI
from langchain_core.prompts import PromptTemplate


# In[6]:


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


# In[11]:


llm = StochasticAI(api_url=YOUR_API_URL)


# In[12]:


llm_chain = LLMChain(prompt=prompt, llm=llm)


# In[13]:


question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)


# In[ ]:




