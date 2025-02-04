#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Pipeshift
---
# # ChatPipeshift
# 
# This will help you getting started with Pipeshift [chat models](/docs/concepts/chat_models/). For detailed documentation of all ChatPipeshift features and configurations head to the [API reference](https://dashboard.pipeshift.com/docs).
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatPipeshift](https://dashboard.pipeshift.com/docs) | [langchain-pipeshift](https://pypi.org/project/langchain-pipeshift/) | ❌ | -| ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-pipeshift?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-pipeshift?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | - | 
# 
# ## Setup
# 
# To access Pipeshift models you'll need to create an account on Pipeshift, get an API key, and install the `langchain-pipeshift` integration package.
# 
# ### Credentials
# 
# Head to [Pipeshift](https://dashboard.pipeshift.com) to sign up to Pipeshift and generate an API key. Once you've done this set the PIPESHIFT_API_KEY environment variable:

# In[1]:


import getpass
import os

if not os.getenv("PIPESHIFT_API_KEY"):
    os.environ["PIPESHIFT_API_KEY"] = getpass.getpass("Enter your Pipeshift API key: ")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[2]:


# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain Pipeshift integration lives in the `langchain-pipeshift` package:

# In[3]:


get_ipython().run_line_magic('pip', 'install -qU langchain-pipeshift')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[4]:


from langchain_pipeshift import ChatPipeshift

llm = ChatPipeshift(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature=0,
    max_tokens=512,
    # other params...
)


# ## Invocation

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


# ## API reference
# 
# For detailed documentation of all ChatPipeshift features and configurations head to the API reference: https://dashboard.pipeshift.com/docs
