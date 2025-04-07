#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: SambaStudio
---
# # ChatSambaStudio
# 
# This will help you getting started with SambaStudio [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatStudio features and configurations head to the [API reference](https://docs.sambanova.ai/sambastudio/latest/index.html).
# 
# **[SambaNova](https://sambanova.ai/)'s** [SambaStudio](https://docs.sambanova.ai/sambastudio/latest/sambastudio-intro.html) SambaStudio is a rich, GUI-based platform that provides the functionality to train, deploy, and manage models in SambaNova [DataScale](https://sambanova.ai/products/datascale) systems.
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatSambaStudio](https://docs.sambanova.ai/sambastudio/latest/index.html) | [langchain-sambanova](https://python.langchain.com/docs/integrations/providers/sambanova/) | ❌ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_sambanova?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_sambanova?style=flat-square&label=%20) |
# 
# ### Model features
# 
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | 
# 
# ## Setup
# 
# To access ChatSambaStudio models you will need to [deploy an endpoint](https://docs.sambanova.ai/sambastudio/latest/language-models.html) in your SambaStudio platform, install the `langchain_sambanova` integration package.
# 
# ```bash
# pip install langchain-sambanova
# ```
# 
# ### Credentials
# 
# Get the URL and API Key from your SambaStudio deployed endpoint and add them to your environment variables:
# 
# ``` bash
# export SAMBASTUDIO_URL="sambastudio-url-key-here"
# export SAMBASTUDIO_API_KEY="your-api-key-here"
# ```

# In[1]:


import getpass
import os

if not os.getenv("SAMBASTUDIO_URL"):
    os.environ["SAMBASTUDIO_URL"] = getpass.getpass("Enter your SambaStudio URL: ")
if not os.getenv("SAMBASTUDIO_API_KEY"):
    os.environ["SAMBASTUDIO_API_KEY"] = getpass.getpass(
        "Enter your SambaStudio API key: "
    )


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain __SambaStudio__ integration lives in the `langchain_sambanova` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-sambanova')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[ ]:


from langchain_sambanova import ChatSambaStudio

llm = ChatSambaStudio(
    model="Meta-Llama-3-70B-Instruct-4096",  # set if using a Bundle endpoint
    max_tokens=1024,
    temperature=0.7,
    top_p=0.01,
    do_sample=True,
    process_prompt="True",  # set if using a Bundle endpoint
)


# ## Invocation

# In[3]:


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

# In[ ]:


from datetime import datetime

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def get_time(kind: str = "both") -> str:
    """Returns current date, current time or both.
    Args:
        kind: date, time or both
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


# In[ ]:


llm_with_tools = llm.bind_tools(tools=tools)
messages = [
    HumanMessage(
        content="I need to schedule a meeting for two weeks from today. "
        "Can you tell me the exact date of the meeting?"
    )
]


# In[ ]:


response = llm_with_tools.invoke(messages)
while len(response.tool_calls) > 0:
    print(f"Intermediate model response: {response.tool_calls}")
    messages.append(response)
    messages = invoke_tools(response.tool_calls, messages)
response = llm_with_tools.invoke(messages)

print(f"final response: {response.content}")


# ## Structured Outputs

# In[ ]:


from pydantic import BaseModel, Field


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


structured_llm = llm.with_structured_output(Joke)

structured_llm.invoke("Tell me a joke about cats")


# ## API reference
# 
# For detailed documentation of all SambaStudio features and configurations head to the API reference: https://docs.sambanova.ai/sambastudio/latest/api-ref-landing.html
