#!/usr/bin/env python
# coding: utf-8

# # CerebriumAI
# 
# `Cerebrium` is an AWS Sagemaker alternative. It also provides API access to [several LLM models](https://docs.cerebrium.ai/cerebrium/prebuilt-models/deployment).
# 
# This notebook goes over how to use Langchain with [CerebriumAI](https://docs.cerebrium.ai/introduction).

# ## Install cerebrium
# The `cerebrium` package is required to use the `CerebriumAI` API. Install `cerebrium` using `pip3 install cerebrium`.

# In[ ]:


# Install the package
get_ipython().system('pip3 install cerebrium')


# ## Imports

# In[ ]:


import os

from langchain.chains import LLMChain
from langchain_community.llms import CerebriumAI
from langchain_core.prompts import PromptTemplate


# ## Set the Environment API Key
# Make sure to get your API key from CerebriumAI. See [here](https://dashboard.cerebrium.ai/login). You are given a 1 hour free of serverless GPU compute to test different models.

# In[ ]:


os.environ["CEREBRIUMAI_API_KEY"] = "YOUR_KEY_HERE"


# ## Create the CerebriumAI instance
# You can specify different parameters such as the model endpoint url, max length, temperature, etc. You must provide an endpoint url.

# In[ ]:


llm = CerebriumAI(endpoint_url="YOUR ENDPOINT URL HERE")


# ## Create a Prompt Template
# We will create a prompt template for Question and Answer.

# In[ ]:


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


# ## Initiate the LLMChain

# In[ ]:


llm_chain = LLMChain(prompt=prompt, llm=llm)


# ## Run the LLMChain
# Provide a question and run the LLMChain.

# In[ ]:


question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)

