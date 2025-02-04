#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: DeepSeek
---
# # ChatDeepSeek
# 
# 
# This will help you getting started with DeepSeek's hosted [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatDeepSeek features and configurations head to the [API reference](https://python.langchain.com/api_reference/deepseek/chat_models/langchain_deepseek.chat_models.ChatDeepSeek.html).
# 
# :::tip
# 
# DeepSeek's models are open source and can be run locally (e.g. in [Ollama](./ollama.ipynb)) or on other inference providers (e.g. [Fireworks](./fireworks.ipynb), [Together](./together.ipynb)) as well.
# 
# :::
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/deepseek) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatDeepSeek](https://python.langchain.com/api_reference/deepseek/chat_models/langchain_deepseek.chat_models.ChatDeepSeek.html) | [langchain-deepseek-official](https://python.langchain.com/api_reference/deepseek/) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-deepseek-official?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-deepseek-official?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 
# 
# ## Setup
# 
# To access DeepSeek models you'll need to create a/an DeepSeek account, get an API key, and install the `langchain-deepseek-official` integration package.
# 
# ### Credentials
# 
# Head to [DeepSeek's API Key page](https://platform.deepseek.com/api_keys) to sign up to DeepSeek and generate an API key. Once you've done this set the `DEEPSEEK_API_KEY` environment variable:

# In[ ]:


import getpass
import os

if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter your DeepSeek API key: ")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain DeepSeek integration lives in the `langchain-deepseek-official` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-deepseek-official')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[ ]:


from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


# ## Invocation

# In[ ]:


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg.content


# ## Chaining
# 
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:

# In[ ]:


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
# For detailed documentation of all ChatDeepSeek features and configurations head to the [API Reference](https://python.langchain.com/api_reference/deepseek/chat_models/langchain_deepseek.chat_models.ChatDeepSeek.html).
