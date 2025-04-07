#!/usr/bin/env python
# coding: utf-8

# # Banana
# 
# 
# [Banana](https://www.banana.dev/about-us) is focused on building the machine learning infrastructure.
# 
# This example goes over how to use LangChain to interact with Banana models

# In[ ]:


##Installing the langchain packages needed to use the integration
get_ipython().run_line_magic('pip', 'install -qU  langchain-community')


# In[ ]:


# Install the package  https://docs.banana.dev/banana-docs/core-concepts/sdks/python
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  banana-dev')


# In[ ]:


# get new tokens: https://app.banana.dev/
# We need three parameters to make a Banana.dev API call:
# * a team api key
# * the model's unique key
# * the model's url slug

import os

# You can get this from the main dashboard
# at https://app.banana.dev
os.environ["BANANA_API_KEY"] = "YOUR_API_KEY"
# OR
# BANANA_API_KEY = getpass()


# In[ ]:


from langchain.chains import LLMChain
from langchain_community.llms import Banana
from langchain_core.prompts import PromptTemplate


# In[ ]:


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


# In[ ]:


# Both of these are found in your model's
# detail page in https://app.banana.dev
llm = Banana(model_key="YOUR_MODEL_KEY", model_url_slug="YOUR_MODEL_URL_SLUG")


# In[ ]:


llm_chain = LLMChain(prompt=prompt, llm=llm)


# In[ ]:


question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)

