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


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[2]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# The LangChain Anthropic integration lives in the `langchain-anthropic` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-anthropic')


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
# One key difference to note between Anthropic models and most others is that the contents of a single Anthropic AI message can either be a single string or a **list of content blocks**. For example when an Anthropic model invokes a tool, the tool invocation is part of the message content (as well as being exposed in the standardized `AIMessage.tool_calls`):

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


# ## API reference
# 
# For detailed documentation of all ChatAnthropic features and configurations head to the API reference: https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html
