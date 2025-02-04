#!/usr/bin/env python
# coding: utf-8

# # OpenAI
# 
# :::caution
# You are currently on a page documenting the use of OpenAI [text completion models](/docs/concepts/text_llms). The latest and most popular OpenAI models are [chat completion models](/docs/concepts/chat_models).
# 
# Unless you are specifically using `gpt-3.5-turbo-instruct`, you are probably looking for [this page instead](/docs/integrations/chat/openai/).
# :::
# 
# [OpenAI](https://platform.openai.com/docs/introduction) offers a spectrum of models with different levels of power suitable for different tasks.
# 
# This example goes over how to use LangChain to interact with `OpenAI` [models](https://platform.openai.com/docs/models)

# ## Overview
# 
# ### Integration details
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/openai) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatOpenAI](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html) | [langchain-openai](https://python.langchain.com/api_reference/openai/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-openai?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-openai?style=flat-square&label=%20) |
# 
# 
# ## Setup
# 
# To access OpenAI models you'll need to create an OpenAI account, get an API key, and install the `langchain-openai` integration package.
# 
# ### Credentials
# 
# Head to https://platform.openai.com to sign up to OpenAI and generate an API key. Once you've done this set the OPENAI_API_KEY environment variable:

# In[1]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# If you want to get automated best in-class tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[2]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# The LangChain OpenAI integration lives in the `langchain-openai` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-openai')


# Should you need to specify your organization ID, you can use the following cell. However, it is not required if you are only part of a single organization or intend to use your default organization. You can check your default organization [here](https://platform.openai.com/account/api-keys).
# 
# To specify your organization, you can use this:
# ```python
# OPENAI_ORGANIZATION = getpass()
# 
# os.environ["OPENAI_ORGANIZATION"] = OPENAI_ORGANIZATION
# ```
# 
# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[2]:


from langchain_openai import OpenAI

llm = OpenAI()


# ## Invocation

# In[3]:


llm.invoke("Hello how are you?")


# ## Chaining

# In[4]:


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("How to say {input} in {output_language}:\n")

chain = prompt | llm
chain.invoke(
    {
        "output_language": "German",
        "input": "I love programming.",
    }
)


# ## Using a proxy
# 
# If you are behind an explicit proxy, you can specify the http_client to pass through

# In[ ]:


get_ipython().run_line_magic('pip', 'install httpx')

import httpx

openai = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    http_client=httpx.Client(proxies="http://proxy.yourcompany.com:8080"),
)

## API reference

For detailed documentation of all `OpenAI` llm features and configurations head to the API reference: https://python.langchain.com/api_reference/openai/llms/langchain_openai.llms.base.OpenAI.html