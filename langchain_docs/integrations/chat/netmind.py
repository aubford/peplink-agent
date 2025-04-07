#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Netmind
---
# # ChatNetmind
# 
# This will help you getting started with Netmind [chat models](https://www.netmind.ai/). For detailed documentation of all ChatNetmind features and configurations head to the [API reference](https://github.com/protagolabs/langchain-netmind).
# 
# -  See https://www.netmind.ai/ for an example.
# 
# ## Overview
# ### Integration details
# 
# | Class                                                                                        | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/) | Package downloads | Package latest |
# |:---------------------------------------------------------------------------------------------| :--- |:-----:|:------------:|:--------------------------------------------------------------:| :---: | :---: |
# | [ChatNetmind](https://python.langchain.com/api_reference/) | [langchain-netmind](https://python.langchain.com/api_reference/) |   ✅   |      ❌       |                               ❌                                | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-netmind?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-netmind?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](../../how_to/tool_calling.ipynb) | [Structured output](../../how_to/structured_output.ipynb) | JSON mode | [Image input](../../how_to/multimodal_inputs.ipynb) | Audio input | Video input | [Token-level streaming](../../how_to/chat_streaming.ipynb) | Native async | [Token usage](../../how_to/chat_token_usage_tracking.ipynb) | [Logprobs](../../how_to/logprobs.ipynb) |
# |:-----------------------------------------------:|:---------------------------------------------------------:|:---------:|:---------------------------------------------------:|:-----------:|:-----------:|:----------------------------------------------------------:|:------------:|:-----------------------------------------------------------:|:---------------------------------------:|
# |                        ✅                        |                             ✅                             |     ✅     |                          ❌                          |      ❌      |      ❌      |                             ✅                              |      ✅       |                              ✅                              |                    ✅                    | 
# 
# ## Setup
# 
# To access Netmind models you'll need to create a/an Netmind account, get an API key, and install the `langchain-netmind` integration package.
# 
# ### Credentials
# 
# Head to https://www.netmind.ai/ to sign up to Netmind and generate an API key. Once you've done this set the NETMIND_API_KEY environment variable:

# In[1]:


import getpass
import os

if not os.getenv("NETMIND_API_KEY"):
    os.environ["NETMIND_API_KEY"] = getpass.getpass("Enter your Netmind API key: ")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[2]:


# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain Netmind integration lives in the `langchain-netmind` package:

# In[3]:


get_ipython().run_line_magic('pip', 'install -qU langchain-netmind')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:
# 

# In[4]:


from langchain_netmind import ChatNetmind

llm = ChatNetmind(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


# ## Invocation
# 

# In[5]:


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg


# In[6]:


print(ai_msg.content)


# ## Chaining
# 
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:
# 

# In[7]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)


# 

# ## API reference
# 
# For detailed documentation of all ChatNetmind features and configurations head to the API reference:  
# * [API reference](https://python.langchain.com/api_reference/)  
# * [langchain-netmind](https://github.com/protagolabs/langchain-netmind)  
# * [pypi](https://pypi.org/project/langchain-netmind/)

# In[ ]:




