#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Groq
---
# # ChatGroq
# 
# This will help you getting started with Groq [chat models](../../concepts/chat_models.mdx). For detailed documentation of all ChatGroq features and configurations head to the [API reference](https://python.langchain.com/api_reference/groq/chat_models/langchain_groq.chat_models.ChatGroq.html). For a list of all Groq models, visit this [link](https://console.groq.com/docs/models?utm_source=langchain).
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/groq) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatGroq](https://python.langchain.com/api_reference/groq/chat_models/langchain_groq.chat_models.ChatGroq.html) | [langchain-groq](https://python.langchain.com/api_reference/groq/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-groq?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-groq?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](../../how_to/tool_calling.ipynb) | [Structured output](../../how_to/structured_output.ipynb) | JSON mode | [Image input](../../how_to/multimodal_inputs.ipynb) | Audio input | Video input | [Token-level streaming](../../how_to/chat_streaming.ipynb) | Native async | [Token usage](../../how_to/chat_token_usage_tracking.ipynb) | [Logprobs](../../how_to/logprobs.ipynb) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 
# 
# ## Setup
# 
# To access Groq models you'll need to create a Groq account, get an API key, and install the `langchain-groq` integration package.
# 
# ### Credentials
# 
# Head to the [Groq console](https://console.groq.com/login?utm_source=langchain&utm_content=chat_page) to sign up to Groq and generate an API key. Once you've done this set the GROQ_API_KEY environment variable:

# In[1]:


import getpass
import os

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[2]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# The LangChain Groq integration lives in the `langchain-groq` package:

# In[3]:


get_ipython().run_line_magic('pip', 'install -qU langchain-groq')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[4]:


from langchain_groq import ChatGroq

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
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
# For detailed documentation of all ChatGroq features and configurations head to the API reference: https://python.langchain.com/api_reference/groq/chat_models/langchain_groq.chat_models.ChatGroq.html
