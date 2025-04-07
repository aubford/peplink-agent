#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Anthropic
---
# # ChatAnthropic
# 
# This notebook provides a quick overview for getting started with Anthropic [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatAnthropic features and configurations head to the [API reference](https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html).
# 
# Anthropic has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [Anthropic docs](https://docs.anthropic.com/en/docs/models-overview).
# 
# 
# :::info AWS Bedrock and Google VertexAI
# 
# Note that certain Anthropic models can also be accessed via AWS Bedrock and Google VertexAI. See the [ChatBedrock](/docs/integrations/chat/bedrock/) and [ChatVertexAI](/docs/integrations/chat/google_vertex_ai_palm/) integrations to use Anthropic models via these services.
# 
# :::
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/anthropic) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatAnthropic](https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html) | [langchain-anthropic](https://python.langchain.com/api_reference/anthropic/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-anthropic?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-anthropic?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
# 
# ## Setup
# 
# To access Anthropic models you'll need to create an Anthropic account, get an API key, and install the `langchain-anthropic` integration package.
# 
# ### Credentials
# 
# Head to https://console.anthropic.com/ to sign up for Anthropic and generate an API key. Once you've done this set the ANTHROPIC_API_KEY environment variable:

# In[1]:


import getpass
import os

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")


# To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[2]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# The LangChain Anthropic integration lives in the `langchain-anthropic` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-anthropic')


# :::info This guide requires ``langchain-anthropic>=0.3.10``
# 
# :::

# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[4]:


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    # other params...
)


# ## Invocation
# 

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
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:

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


# ## Content blocks
# 
# Content from a single Anthropic AI message can either be a single string or a **list of content blocks**. For example when an Anthropic model invokes a tool, the tool invocation is part of the message content (as well as being exposed in the standardized `AIMessage.tool_calls`):

# In[8]:


from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])
ai_msg = llm_with_tools.invoke("Which city is hotter today: LA or NY?")
ai_msg.content


# In[9]:


ai_msg.tool_calls


# ## Extended thinking
# 
# Claude 3.7 Sonnet supports an [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) feature, which will output the step-by-step reasoning process that led to its final answer.
# 
# To use it, specify the `thinking` parameter when initializing `ChatAnthropic`. It can also be passed in as a kwarg during invocation.
# 
# You will need to specify a token budget to use this feature. See usage example below:

# In[1]:


import json

from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    max_tokens=5000,
    thinking={"type": "enabled", "budget_tokens": 2000},
)

response = llm.invoke("What is the cube root of 50.653?")
print(json.dumps(response.content, indent=2))


# ## Prompt caching
# 
# Anthropic supports [caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) of [elements of your prompts](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#what-can-be-cached), including messages, tool definitions, tool results, images and documents. This allows you to re-use large documents, instructions, [few-shot documents](/docs/concepts/few_shot_prompting/), and other data to reduce latency and costs.
# 
# To enable caching on an element of a prompt, mark its associated content block using the `cache_control` key. See examples below:
# 
# ### Messages

# In[1]:


import requests
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")

# Pull LangChain readme
get_response = requests.get(
    "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
)
readme = get_response.text

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a technology expert.",
            },
            {
                "type": "text",
                "text": f"{readme}",
                # highlight-next-line
                "cache_control": {"type": "ephemeral"},
            },
        ],
    },
    {
        "role": "user",
        "content": "What's LangChain, according to its README?",
    },
]

response_1 = llm.invoke(messages)
response_2 = llm.invoke(messages)

usage_1 = response_1.usage_metadata["input_token_details"]
usage_2 = response_2.usage_metadata["input_token_details"]

print(f"First invocation:\n{usage_1}")
print(f"\nSecond:\n{usage_2}")


# ### Tools

# In[2]:


from langchain_anthropic import convert_to_anthropic_tool
from langchain_core.tools import tool

# For demonstration purposes, we artificially expand the
# tool description.
description = (
    f"Get the weather at a location. By the way, check out this readme: {readme}"
)


@tool(description=description)
def get_weather(location: str) -> str:
    return "It's sunny."


# Enable caching on the tool
# highlight-start
weather_tool = convert_to_anthropic_tool(get_weather)
weather_tool["cache_control"] = {"type": "ephemeral"}
# highlight-end

llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")
llm_with_tools = llm.bind_tools([weather_tool])
query = "What's the weather in San Francisco?"

response_1 = llm_with_tools.invoke(query)
response_2 = llm_with_tools.invoke(query)

usage_1 = response_1.usage_metadata["input_token_details"]
usage_2 = response_2.usage_metadata["input_token_details"]

print(f"First invocation:\n{usage_1}")
print(f"\nSecond:\n{usage_2}")


# ### Incremental caching in conversational applications
# 
# Prompt caching can be used in [multi-turn conversations](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#continuing-a-multi-turn-conversation) to maintain context from earlier messages without redundant processing.
# 
# We can enable incremental caching by marking the final message with `cache_control`. Claude will automatically use the longest previously-cached prefix for follow-up messages.
# 
# Below, we implement a simple chatbot that incorporates this feature. We follow the LangChain [chatbot tutorial](/docs/tutorials/chatbot/), but add a custom [reducer](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) that automatically marks the last content block in each user message with `cache_control`. See below:

# In[2]:


import requests
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, add_messages
from typing_extensions import Annotated, TypedDict

llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")

# Pull LangChain readme
get_response = requests.get(
    "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
)
readme = get_response.text


def messages_reducer(left: list, right: list) -> list:
    # Update last user message
    for i in range(len(right) - 1, -1, -1):
        if right[i].type == "human":
            right[i].content[-1]["cache_control"] = {"type": "ephemeral"}
            break

    return add_messages(left, right)


class State(TypedDict):
    messages: Annotated[list, messages_reducer]


workflow = StateGraph(state_schema=State)


# Define the function that calls the model
def call_model(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# In[3]:


from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."

input_message = HumanMessage([{"type": "text", "text": query}])
output = app.invoke({"messages": [input_message]}, config)
output["messages"][-1].pretty_print()
print(f'\n{output["messages"][-1].usage_metadata["input_token_details"]}')


# In[4]:


query = f"Check out this readme: {readme}"

input_message = HumanMessage([{"type": "text", "text": query}])
output = app.invoke({"messages": [input_message]}, config)
output["messages"][-1].pretty_print()
print(f'\n{output["messages"][-1].usage_metadata["input_token_details"]}')


# In[5]:


query = "What was my name again?"

input_message = HumanMessage([{"type": "text", "text": query}])
output = app.invoke({"messages": [input_message]}, config)
output["messages"][-1].pretty_print()
print(f'\n{output["messages"][-1].usage_metadata["input_token_details"]}')


# In the [LangSmith trace](https://smith.langchain.com/public/4d0584d8-5f9e-4b91-8704-93ba2ccf416a/r), toggling "raw output" will show exactly what messages are sent to the chat model, including `cache_control` keys.

# ## Token-efficient tool use
# 
# Anthropic supports a (beta) [token-efficient tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use) feature. To use it, specify the relevant beta-headers when instantiating the model.

# In[1]:


from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

llm = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
    temperature=0,
    # highlight-start
    model_kwargs={
        "extra_headers": {"anthropic-beta": "token-efficient-tools-2025-02-19"}
    },
    # highlight-end
)


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return "It's sunny."


llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in San Francisco?")
print(response.tool_calls)
print(f'\nTotal tokens: {response.usage_metadata["total_tokens"]}')


# ## Citations
# 
# Anthropic supports a [citations](https://docs.anthropic.com/en/docs/build-with-claude/citations) feature that lets Claude attach context to its answers based on source documents supplied by the user. When [document content blocks](https://docs.anthropic.com/en/docs/build-with-claude/citations#document-types) with `"citations": {"enabled": True}` are included in a query, Claude may generate citations in its response.
# 
# ### Simple example
# 
# In this example we pass a [plain text document](https://docs.anthropic.com/en/docs/build-with-claude/citations#plain-text-documents). In the background, Claude [automatically chunks](https://docs.anthropic.com/en/docs/build-with-claude/citations#plain-text-documents) the input text into sentences, which are used when generating citations.

# In[2]:


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-haiku-latest")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "The grass is green. The sky is blue.",
                },
                "title": "My Document",
                "context": "This is a trustworthy document.",
                "citations": {"enabled": True},
            },
            {"type": "text", "text": "What color is the grass and sky?"},
        ],
    }
]
response = llm.invoke(messages)
response.content


# ### Using with text splitters
# 
# Anthropic also lets you specify your own splits using [custom document](https://docs.anthropic.com/en/docs/build-with-claude/citations#custom-content-documents) types. LangChain [text splitters](/docs/concepts/text_splitters/) can be used to generate meaningful splits for this purpose. See the below example, where we split the LangChain README (a markdown document) and pass it to Claude as context:

# In[3]:


import requests
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import MarkdownTextSplitter


def format_to_anthropic_documents(documents: list[str]):
    return {
        "type": "document",
        "source": {
            "type": "content",
            "content": [{"type": "text", "text": document} for document in documents],
        },
        "citations": {"enabled": True},
    }


# Pull readme
get_response = requests.get(
    "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
)
readme = get_response.text

# Split into chunks
splitter = MarkdownTextSplitter(
    chunk_overlap=0,
    chunk_size=50,
)
documents = splitter.split_text(readme)

# Construct message
message = {
    "role": "user",
    "content": [
        format_to_anthropic_documents(documents),
        {"type": "text", "text": "Give me a link to LangChain's tutorials."},
    ],
}

# Query LLM
llm = ChatAnthropic(model="claude-3-5-haiku-latest")
response = llm.invoke([message])

response.content


# ## Built-in tools
# 
# Anthropic supports a variety of [built-in tools](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/text-editor-tool), which can be bound to the model in the [usual way](/docs/how_to/tool_calling/). Claude will generate tool calls adhering to its internal schema for the tool:

# In[1]:


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")

tool = {"type": "text_editor_20250124", "name": "str_replace_editor"}
llm_with_tools = llm.bind_tools([tool])

response = llm_with_tools.invoke(
    "There's a syntax error in my primes.py file. Can you help me fix it?"
)
print(response.text())
response.tool_calls


# ## API reference
# 
# For detailed documentation of all ChatAnthropic features and configurations head to the API reference: https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html
