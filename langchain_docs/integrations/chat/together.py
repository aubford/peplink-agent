#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Together
---
# # ChatTogether
# 
# 
# This page will help you get started with Together AI [chat models](../../concepts/chat_models.mdx). For detailed documentation of all ChatTogether features and configurations head to the [API reference](https://python.langchain.com/api_reference/together/chat_models/langchain_together.chat_models.ChatTogether.html).
# 
# [Together AI](https://www.together.ai/) offers an API to query [50+ leading open-source models](https://docs.together.ai/docs/chat-models)
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/togetherai) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatTogether](https://python.langchain.com/api_reference/together/chat_models/langchain_together.chat_models.ChatTogether.html) | [langchain-together](https://python.langchain.com/api_reference/together/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-together?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-together?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](../../how_to/tool_calling.ipynb) | [Structured output](../../how_to/structured_output.ipynb) | JSON mode | [Image input](../../how_to/multimodal_inputs.ipynb) | Audio input | Video input | [Token-level streaming](../../how_to/chat_streaming.ipynb) | Native async | [Token usage](../../how_to/chat_token_usage_tracking.ipynb) | [Logprobs](../../how_to/logprobs.ipynb) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
# 
# ## Setup
# 
# To access Together models you'll need to create a/an Together account, get an API key, and install the `langchain-together` integration package.
# 
# ### Credentials
# 
# Head to [this page](https://api.together.ai) to sign up to Together and generate an API key. Once you've done this set the TOGETHER_API_KEY environment variable:

# In[1]:


import getpass
import os

if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter your Together API key: ")


# To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[2]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# The LangChain Together integration lives in the `langchain-together` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-together')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[3]:


from langchain_together import ChatTogether

llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


# ## Invocation

# In[4]:


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg


# In[5]:


print(ai_msg.content)


# ## Chaining
# 
# We can [chain](../../how_to/sequence.ipynb) our model with a prompt template like so:

# In[7]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
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
# For detailed documentation of all ChatTogether features and configurations head to the API reference: https://python.langchain.com/api_reference/together/chat_models/langchain_together.chat_models.ChatTogether.html
