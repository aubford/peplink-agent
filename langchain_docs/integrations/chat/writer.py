#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Writer
---
# # ChatWriter
# 
# This notebook provides a quick overview for getting started with Writer [chat models](/docs/concepts/chat_models).
# 
# Writer has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [Writer docs](https://dev.writer.com/home/models).
# 
# :::

# ## Overview
# 
# ### Integration details
# | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |:----------:| :---: | :---: |
# | ChatWriter | langchain-community | ❌ | ❌ |     ❌      | ❌ | ❌ |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | Structured output | JSON mode | Image input | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async |         [Token usage](/docs/how_to/chat_token_usage_tracking/)          | Logprobs |
# | :---: |:-----------------:| :---: | :---: |  :---: | :---: | :---: | :---: |:--------------------------------:|:--------:|
# | ✅ |         ❌         | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |                ✅                 |    ❌     |
# 
# ## Setup
# 
# To access Writer models you'll need to create a Writer account, get an API key, and install the `writer-sdk` and `langchain-community` packages.
# 
# ### Credentials
# 
# Head to [Writer AI Studio](https://app.writer.com/aistudio/signup?utm_campaign=devrel) to sign up to OpenAI and generate an API key. Once you've done this set the WRITER_API_KEY environment variable:

# In[1]:


import getpass
import os

if not os.environ.get("WRITER_API_KEY"):
    os.environ["WRITER_API_KEY"] = getpass.getpass("Enter your Writer API key:")


# ### Installation
# 
# The LangChain Writer integration lives in the `langchain-community` package:

# In[2]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community writer-sdk')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[3]:


from langchain_community.chat_models.writer import ChatWriter

llm = ChatWriter(
    model="palmyra-x-004",
    temperature=0.7,
    max_tokens=1000,
    # other params...
)


# ## Invocation

# In[4]:


messages = [
    (
        "system",
        "You are a helpful assistant that writes poems about the Python programming language.",
    ),
    ("human", "Write a poem about Python."),
]
ai_msg = llm.invoke(messages)


# In[5]:


print(ai_msg.content)


# ## Streaming

# In[6]:


ai_stream = llm.stream(messages)


# In[7]:


for chunk in ai_stream:
    print(chunk.content, end="")


# ## Chaining
# 
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:

# In[8]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that writes poems about the {input_language} programming language.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "Java",
        "input": "Write a poem about Java.",
    }
)


# ## Tool calling
# 
# Writer supports [tool calling](https://dev.writer.com/api-guides/tool-calling), which lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool.
# 
# ### ChatWriter.bind_tools()
# 
# With `ChatWriter.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model. Under the hood these are converted to tool schemas, which looks like:
# ```
# {
#     "name": "...",
#     "description": "...",
#     "parameters": {...}  # JSONSchema
# }
# ```
# and passed in every model invocation.

# In[9]:


from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])


# In[10]:


ai_msg = llm_with_tools.invoke(
    "what is the weather like in New York City",
)


# ### AIMessage.tool_calls
# Notice that the AIMessage has a `tool_calls` attribute. This contains in a standardized ToolCall format that is model-provider agnostic.

# In[11]:


print(ai_msg.tool_calls)


# For more on binding tools and tool call outputs, head to the [tool calling](/docs/how_to/function_calling) docs.

# ## API reference
# 
# For detailed documentation of all Writer features, head to our [API reference](https://dev.writer.com/api-guides/api-reference/completion-api/chat-completion).
