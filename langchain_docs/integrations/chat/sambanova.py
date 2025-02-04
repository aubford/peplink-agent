#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: SambaNovaCloud
---
# # ChatSambaNovaCloud
# 
# This will help you getting started with SambaNovaCloud [chat models](/docs/concepts/chat_models/). For detailed documentation of all ChatSambaNovaCloud features and configurations head to the [API reference](https://python.langchain.com/api_reference/sambanova/chat_models/langchain_sambanova.ChatSambaNovaCloud.html).
# 
# **[SambaNova](https://sambanova.ai/)'s** [SambaNova Cloud](https://cloud.sambanova.ai/) is a platform for performing inference with open-source models
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatSambaNovaCloud](https://python.langchain.com/api_reference/sambanova/chat_models/langchain_sambanova.ChatSambaNovaCloud.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ❌ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_sambanova?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_sambanova?style=flat-square&label=%20) |
# 
# ### Model features
# 
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](//docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | 
# 
# ## Setup
# 
# To access ChatSambaNovaCloud models you will need to create a [SambaNovaCloud](https://cloud.sambanova.ai/) account, get an API key, install the `langchain_sambanova` integration package.
# 
# ```bash
# pip install langchain-sambanova
# ```
# 
# ### Credentials
# 
# Get an API Key from [cloud.sambanova.ai](https://cloud.sambanova.ai/apis) and add it to your environment variables:
# 
# ``` bash
# export SAMBANOVA_API_KEY="your-api-key-here"
# ```

# In[1]:


import getpass
import os

if not os.getenv("SAMBANOVA_API_KEY"):
    os.environ["SAMBANOVA_API_KEY"] = getpass.getpass(
        "Enter your SambaNova Cloud API key: "
    )


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain __SambaNovaCloud__ integration lives in the `langchain_sambanova` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-sambanova')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[ ]:


from langchain_sambanova import ChatSambaNovaCloud

llm = ChatSambaNovaCloud(
    model="Meta-Llama-3.3-70B-Instruct",
    max_tokens=1024,
    temperature=0.7,
    top_p=0.01,
)


# ## Invocation

# In[3]:


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. "
        "Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg


# In[4]:


print(ai_msg.content)


# ## Chaining
# 
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:

# In[5]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} "
            "to {output_language}.",
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

# In[6]:


system = "You are a helpful assistant with pirate accent."
human = "I want to learn more about this animal: {animal}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | llm

for chunk in chain.stream({"animal": "owl"}):
    print(chunk.content, end="", flush=True)


# ## Async

# In[7]:


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "what is the capital of {country}?",
        )
    ]
)

chain = prompt | llm
await chain.ainvoke({"country": "France"})


# ## Async Streaming

# In[8]:


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "in less than {num_words} words explain me {topic} ",
        )
    ]
)
chain = prompt | llm

async for chunk in chain.astream({"num_words": 30, "topic": "quantum computers"}):
    print(chunk.content, end="", flush=True)


# ## Tool calling

# In[9]:


from datetime import datetime

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def get_time(kind: str = "both") -> str:
    """Returns current date, current time or both.
    Args:
        kind(str): date, time or both
    Returns:
        str: current date, current time or both
    """
    if kind == "date":
        date = datetime.now().strftime("%m/%d/%Y")
        return f"Current date: {date}"
    elif kind == "time":
        time = datetime.now().strftime("%H:%M:%S")
        return f"Current time: {time}"
    else:
        date = datetime.now().strftime("%m/%d/%Y")
        time = datetime.now().strftime("%H:%M:%S")
        return f"Current date: {date}, Current time: {time}"


tools = [get_time]


def invoke_tools(tool_calls, messages):
    available_functions = {tool.name: tool for tool in tools}
    for tool_call in tool_calls:
        selected_tool = available_functions[tool_call["name"]]
        tool_output = selected_tool.invoke(tool_call["args"])
        print(f"Tool output: {tool_output}")
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    return messages


# In[10]:


llm_with_tools = llm.bind_tools(tools=tools)
messages = [
    HumanMessage(
        content="I need to schedule a meeting for two weeks from today. "
        "Can you tell me the exact date of the meeting?"
    )
]


# In[11]:


response = llm_with_tools.invoke(messages)
while len(response.tool_calls) > 0:
    print(f"Intermediate model response: {response.tool_calls}")
    messages.append(response)
    messages = invoke_tools(response.tool_calls, messages)
    response = llm_with_tools.invoke(messages)

print(f"final response: {response.content}")


# ## Structured Outputs

# In[12]:


from pydantic import BaseModel, Field


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


structured_llm = llm.with_structured_output(Joke)

structured_llm.invoke("Tell me a joke about cats")


# ## Input Image

# In[13]:


multimodal_llm = ChatSambaNovaCloud(
    model="Llama-3.2-11B-Vision-Instruct",
    max_tokens=1024,
    temperature=0.7,
    top_p=0.01,
)


# In[14]:


import base64

import httpx

image_url = (
    "https://images.pexels.com/photos/147411/italy-mountains-dawn-daybreak-147411.jpeg"
)
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

message = HumanMessage(
    content=[
        {"type": "text", "text": "describe the weather in this image in 1 sentence"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
)
response = multimodal_llm.invoke([message])
print(response.content)


# ## API reference
# 
# For detailed documentation of all ChatSambaNovaCloud features and configurations head to the API reference: https://python.langchain.com/api_reference/sambanova/chat_models/langchain_sambanova.ChatSambaNovaCloud.html
