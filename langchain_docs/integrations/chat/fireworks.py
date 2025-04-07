#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Fireworks
---
# # ChatFireworks
# 
# This doc help you get started with Fireworks AI [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatFireworks features and configurations head to the [API reference](https://python.langchain.com/api_reference/fireworks/chat_models/langchain_fireworks.chat_models.ChatFireworks.html).
# 
# Fireworks AI is an AI inference platform to run and customize models. For a list of all models served by Fireworks see the [Fireworks docs](https://fireworks.ai/models).
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/fireworks) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatFireworks](https://python.langchain.com/api_reference/fireworks/chat_models/langchain_fireworks.chat_models.ChatFireworks.html) | [langchain-fireworks](https://python.langchain.com/api_reference/fireworks/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-fireworks?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-fireworks?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
# 
# ## Setup
# 
# To access Fireworks models you'll need to create a Fireworks account, get an API key, and install the `langchain-fireworks` integration package.
# 
# ### Credentials
# 
# Head to (ttps://fireworks.ai/login to sign up to Fireworks and generate an API key. Once you've done this set the FIREWORKS_API_KEY environment variable:

# In[ ]:


import getpass
import os

if "FIREWORKS_API_KEY" not in os.environ:
    os.environ["FIREWORKS_API_KEY"] = getpass.getpass("Enter your Fireworks API key: ")


# To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# The LangChain Fireworks integration lives in the `langchain-fireworks` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-fireworks')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:
# 
# - TODO: Update model instantiation with relevant params.

# In[1]:


from langchain_fireworks import ChatFireworks

llm = ChatFireworks(
    model="accounts/fireworks/models/llama-v3-70b-instruct",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


# ## Invocation

# In[2]:


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg


# In[3]:


print(ai_msg.content)


# ## Chaining
# 
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:

# In[4]:


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
# For detailed documentation of all ChatFireworks features and configurations head to the API reference: https://python.langchain.com/api_reference/fireworks/chat_models/langchain_fireworks.chat_models.ChatFireworks.html
