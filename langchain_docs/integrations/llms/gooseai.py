#!/usr/bin/env python
# coding: utf-8

# # GooseAI
#
# `GooseAI` is a fully managed NLP-as-a-Service, delivered via API. GooseAI provides access to [these models](https://goose.ai/docs/models).
#
# This notebook goes over how to use Langchain with [GooseAI](https://goose.ai/).
#

# ## Install openai
# The `openai` package is required to use the GooseAI API. Install `openai` using `pip install openai`.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  langchain-openai")


# ## Imports

# In[ ]:


import os

from langchain.chains import LLMChain
from langchain_community.llms import GooseAI
from langchain_core.prompts import PromptTemplate


# ## Set the Environment API Key
# Make sure to get your API key from GooseAI. You are given $10 in free credits to test different models.

# In[ ]:


from getpass import getpass

GOOSEAI_API_KEY = getpass()


# In[ ]:


os.environ["GOOSEAI_API_KEY"] = GOOSEAI_API_KEY


# ## Create the GooseAI instance
# You can specify different parameters such as the model name, max tokens generated, temperature, etc.

# In[ ]:


llm = GooseAI()


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
