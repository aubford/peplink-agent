#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: LiteLLM
---
# # ChatLiteLLM
# 
# [LiteLLM](https://github.com/BerriAI/litellm) is a library that simplifies calling Anthropic, Azure, Huggingface, Replicate, etc.
# 
# This notebook covers how to get started with using Langchain + the LiteLLM I/O library.
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS support| Package downloads | Package latest |
# | :---  | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatLiteLLM](https://python.langchain.com/docs/integrations/chat/litellm/) | [langchain-litellm](https://pypi.org/project/langchain-litellm/)| ❌ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-litellm?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-litellm?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](https://python.langchain.com/docs/how_to/tool_calling/) | [Structured output](https://python.langchain.com/docs/how_to/structured_output/) | JSON mode | Image input | Audio input | Video input | [Token-level streaming](https://python.langchain.com/docs/integrations/chat/litellm/#chatlitellm-also-supports-async-and-streaming-functionality) | [Native async](https://python.langchain.com/docs/integrations/chat/litellm/#chatlitellm-also-supports-async-and-streaming-functionality) | [Token usage](https://python.langchain.com/docs/how_to/chat_token_usage_tracking/) | [Logprobs](https://python.langchain.com/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
# 
# ### Setup
# To access ChatLiteLLM models you'll need to install the `langchain-litellm` package and create an OpenAI, Anthropic, Azure, Replicate, OpenRouter, Hugging Face, Together AI or Cohere account. Then you have to get an API key, and export it as an environment variable.

# ## Credentials
# 
# You have to choose the LLM provider you want and sign up with them to get their API key.
# 
# ### Example - Anthropic
# Head to https://console.anthropic.com/ to sign up for Anthropic and generate an API key. Once you've done this set the ANTHROPIC_API_KEY environment variable.
# 
# 
# ### Example - OpenAI
# Head to https://platform.openai.com/api-keys to sign up for OpenAI and generate an API key. Once you've done this set the OPENAI_API_KEY environment variable.

# In[1]:


## set ENV variables
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"


# ### Installation
# 
# The LangChain LiteLLM integration lives in the `langchain-litellm` package:

# In[2]:


get_ipython().run_line_magic('pip', 'install -qU langchain-litellm')


# ## Instantiation
# Now we can instantiate our model object and generate chat completions:

# In[3]:


from langchain_litellm.chat_models import ChatLiteLLM

llm = ChatLiteLLM(model="gpt-3.5-turbo")


# ## Invocation

# In[4]:


response = await llm.ainvoke(
    "Classify the text into neutral, negative or positive. Text: I think the food was okay. Sentiment:"
)
print(response)


# ## `ChatLiteLLM` also supports async and streaming functionality:

# In[5]:


async for token in llm.astream("Hello, please explain how antibiotics work"):
    print(token.text(), end="")


# ## API reference
# For detailed documentation of all `ChatLiteLLM` features and configurations head to the API reference: https://github.com/Akshay-Dongare/langchain-litellm
