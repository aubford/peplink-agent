#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Cerebras
---
# # ChatCerebras
# 
# This notebook provides a quick overview for getting started with Cerebras [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatCerebras features and configurations head to the [API reference](https://python.langchain.com/api_reference/cerebras/chat_models/langchain_cerebras.chat_models.ChatCerebras.html#).
# 
# At Cerebras, we've developed the world's largest and fastest AI processor, the Wafer-Scale Engine-3 (WSE-3). The Cerebras CS-3 system, powered by the WSE-3, represents a new class of AI supercomputer that sets the standard for generative AI training and inference with unparalleled performance and scalability.
# 
# With Cerebras as your inference provider, you can:
# - Achieve unprecedented speed for AI inference workloads
# - Build commercially with high throughput
# - Effortlessly scale your AI workloads with our seamless clustering technology
# 
# Our CS-3 systems can be quickly and easily clustered to create the largest AI supercomputers in the world, making it simple to place and run the largest models. Leading corporations, research institutions, and governments are already using Cerebras solutions to develop proprietary models and train popular open-source models.
# 
# Want to experience the power of Cerebras? Check out our [website](https://cerebras.ai) for more resources and explore options for accessing our technology through the Cerebras Cloud or on-premise deployments!
# 
# For more information about Cerebras Cloud, visit [cloud.cerebras.ai](https://cloud.cerebras.ai/). Our API reference is available at [inference-docs.cerebras.ai](https://inference-docs.cerebras.ai/).
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/cerebras) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatCerebras](https://python.langchain.com/api_reference/cerebras/chat_models/langchain_cerebras.chat_models.ChatCerebras.html#) | [langchain-cerebras](https://python.langchain.com/api_reference/cerebras/index.html) | ❌ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-cerebras?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-cerebras?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling/) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅  | ✅ | ❌ |
# 
# ## Setup
# 
# ```bash
# pip install langchain-cerebras
# ```
# 
# ### Credentials
# 
# Get an API Key from [cloud.cerebras.ai](https://cloud.cerebras.ai/) and add it to your environment variables:
# ```
# export CEREBRAS_API_KEY="your-api-key-here"
# ```

# In[8]:


import getpass
import os

if "CEREBRAS_API_KEY" not in os.environ:
    os.environ["CEREBRAS_API_KEY"] = getpass.getpass("Enter your Cerebras API key: ")


# To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[9]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# The LangChain Cerebras integration lives in the `langchain-cerebras` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-cerebras')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:
# 

# In[ ]:


from langchain_cerebras import ChatCerebras

llm = ChatCerebras(
    model="llama-3.3-70b",
    # other params...
)


# ## Invocation

# In[13]:


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg


# ## Chaining
# 
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:

# In[14]:


from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate

llm = ChatCerebras(
    model="llama-3.3-70b",
    # other params...
)

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


# ## Streaming

# In[15]:


from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate

llm = ChatCerebras(
    model="llama-3.3-70b",
    # other params...
)

system = "You are an expert on animals who must answer questions in a manner that a 5 year old can understand."
human = "I want to learn more about this animal: {animal}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | llm

for chunk in chain.stream({"animal": "Lion"}):
    print(chunk.content, end="", flush=True)


# ## Async

# In[19]:


from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate

llm = ChatCerebras(
    model="llama-3.3-70b",
    # other params...
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Let's play a game of opposites. What's the opposite of {topic}? Just give me the answer with no extra input.",
        )
    ]
)
chain = prompt | llm
await chain.ainvoke({"topic": "fire"})


# ## Async Streaming

# In[27]:


from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate

llm = ChatCerebras(
    model="llama-3.3-70b",
    # other params...
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Write a long convoluted story about {subject}. I want {num_paragraphs} paragraphs.",
        )
    ]
)
chain = prompt | llm

async for chunk in chain.astream({"num_paragraphs": 3, "subject": "blackholes"}):
    print(chunk.content, end="", flush=True)


# ## API reference
# 
# For detailed documentation of all ChatCerebras features and configurations head to the API reference: https://python.langchain.com/api_reference/cerebras/chat_models/langchain_cerebras.chat_models.ChatCerebras.html#
