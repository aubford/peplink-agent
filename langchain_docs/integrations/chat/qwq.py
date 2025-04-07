#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Qwen QwQ
---
# # ChatQwQ
# 
# This will help you getting started with QwQ [chat models](../../concepts/chat_models.mdx). For detailed documentation of all ChatQwQ features and configurations head to the [API reference](https://pypi.org/project/langchain-qwq/).
# 
# ## Overview
# ### Integration details
# 
# 
# | Class | Package | Local | Serializable | Package downloads | Package latest |
# | :--- | :--- | :---: |  :---: | :---: | :---: |
# | [ChatQwQ](https://pypi.org/project/langchain-qwq/) | [langchain-qwq](https://pypi.org/project/langchain-qwq/) | ❌ | beta | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-qwq?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-qwq?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](../../how_to/tool_calling.ipynb) | [Structured output](../../how_to/structured_output.ipynb) | JSON mode | [Image input](../../how_to/multimodal_inputs.ipynb) | Audio input | Video input | [Token-level streaming](../../how_to/chat_streaming.ipynb) | Native async | [Token usage](../../how_to/chat_token_usage_tracking.ipynb) | [Logprobs](../../how_to/logprobs.ipynb) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ |❌  | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 
# 
# ## Setup
# 
# To access QwQ models you'll need to create an Alibaba Cloud account, get an API key, and install the `langchain-qwq` integration package.
# 
# ### Credentials
# 
# Head to [Alibaba's API Key page](https://account.alibabacloud.com/login/login.htm?oauth_callback=https%3A%2F%2Fbailian.console.alibabacloud.com%2F%3FapiKey%3D1&lang=en#/api-key) to sign up to Alibaba Cloud and generate an API key. Once you've done this set the `DASHSCOPE_API_KEY` environment variable:

# In[2]:


import getpass
import os

if not os.getenv("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter your Dashscope API key: ")


# ### Installation
# 
# The LangChain QwQ integration lives in the `langchain-qwq` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-qwq')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[3]:


from langchain_qwq import ChatQwQ

llm = ChatQwQ(
    model="qwq-plus",
    max_tokens=3_000,
    timeout=None,
    max_retries=2,
    # other params...
)


# ## Invocation

# In[4]:


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French."
        "Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg


# ## Chaining
# 
# We can [chain](../../how_to/sequence.ipynb) our model with a prompt template like so:

# In[5]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates"
            "{input_language} to {output_language}.",
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


# ## Tool Calling
# ChatQwQ supports tool calling API that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool.

# ### Use with `bind_tools`

# In[6]:


from langchain_core.tools import tool
from langchain_qwq import ChatQwQ


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


llm = ChatQwQ()

llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke("What's 5 times forty two")

print(msg)


# ## API reference
# 
# For detailed documentation of all ChatQwQ features and configurations head to the [API reference](https://pypi.org/project/langchain-qwq/)
