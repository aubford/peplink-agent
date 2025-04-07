#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: OpenAI
---
# # ChatOpenAI
# 
# This notebook provides a quick overview for getting started with OpenAI [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatOpenAI features and configurations head to the [API reference](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html).
# 
# OpenAI has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [OpenAI docs](https://platform.openai.com/docs/models).
# 
# :::info Azure OpenAI
# 
# Note that certain OpenAI models can also be accessed via the [Microsoft Azure platform](https://azure.microsoft.com/en-us/products/ai-services/openai-service). To use the Azure OpenAI service use the [AzureChatOpenAI integration](/docs/integrations/chat/azure_chat_openai/).
# 
# :::

# ## Overview
# 
# ### Integration details
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/openai) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatOpenAI](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html) | [langchain-openai](https://python.langchain.com/api_reference/openai/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-openai?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-openai?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | Image input | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | 
# 
# ## Setup
# 
# To access OpenAI models you'll need to create an OpenAI account, get an API key, and install the `langchain-openai` integration package.
# 
# ### Credentials
# 
# Head to https://platform.openai.com to sign up to OpenAI and generate an API key. Once you've done this set the OPENAI_API_KEY environment variable:

# In[1]:


import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# The LangChain OpenAI integration lives in the `langchain-openai` package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-openai')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[2]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)


# ## Invocation

# In[3]:


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
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


# ## Tool calling
# 
# OpenAI has a [tool calling](https://platform.openai.com/docs/guides/function-calling) (we use "tool calling" and "function calling" interchangeably here) API that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool. tool-calling is extremely useful for building tool-using chains and agents, and for getting structured outputs from models more generally.
# 
# ### ChatOpenAI.bind_tools()
# 
# With `ChatOpenAI.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model. Under the hood these are converted to an OpenAI tool schemas, which looks like:
# ```
# {
#     "name": "...",
#     "description": "...",
#     "parameters": {...}  # JSONSchema
# }
# ```
# and passed in every model invocation.

# In[6]:


from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])


# In[7]:


ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg


# ### ``strict=True``
# 
# :::info Requires ``langchain-openai>=0.1.21``
# 
# :::
# 
# As of Aug 6, 2024, OpenAI supports a `strict` argument when calling tools that will enforce that the tool argument schema is respected by the model. See more here: https://platform.openai.com/docs/guides/function-calling
# 
# **Note**: If ``strict=True`` the tool definition will also be validated, and a subset of JSON schema are accepted. Crucially, schema cannot have optional args (those with default values). Read the full docs on what types of schema are supported here: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas. 

# In[8]:


llm_with_tools = llm.bind_tools([GetWeather], strict=True)
ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg


# ### AIMessage.tool_calls
# Notice that the AIMessage has a `tool_calls` attribute. This contains in a standardized ToolCall format that is model-provider agnostic.

# In[9]:


ai_msg.tool_calls


# For more on binding tools and tool call outputs, head to the [tool calling](/docs/how_to/function_calling) docs.

# ## Responses API
# 
# :::info Requires ``langchain-openai>=0.3.9``
# 
# :::
# 
# OpenAI supports a [Responses](https://platform.openai.com/docs/guides/responses-vs-chat-completions) API that is oriented toward building [agentic](/docs/concepts/agents/) applications. It includes a suite of [built-in tools](https://platform.openai.com/docs/guides/tools?api-mode=responses), including web and file search. It also supports management of [conversation state](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses), allowing you to continue a conversational thread without explicitly passing in previous messages.
# 
# `ChatOpenAI` will route to the Responses API if one of these features is used. You can also specify `use_responses_api=True` when instantiating `ChatOpenAI`.
# 
# ### Built-in tools
# 
# Equipping `ChatOpenAI` with built-in tools will ground its responses with outside information, such as via context in files or the web. The [AIMessage](/docs/concepts/messages/#aimessage) generated from the model will include information about the built-in tool invocation.
# 
# #### Web search
# 
# To trigger a web search, pass `{"type": "web_search_preview"}` to the model as you would another tool.
# 
# :::tip
# 
# You can also pass built-in tools as invocation params:
# ```python
# llm.invoke("...", tools=[{"type": "web_search_preview"}])
# ```
# 
# :::

# In[1]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

tool = {"type": "web_search_preview"}
llm_with_tools = llm.bind_tools([tool])

response = llm_with_tools.invoke("What was a positive news story from today?")


# Note that the response includes structured [content blocks](/docs/concepts/messages/#content-1) that include both the text of the response and OpenAI [annotations](https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses#output-and-citations) citing its sources:

# In[7]:


response.content


# :::tip
# 
# You can recover just the text content of the response as a string by using `response.text()`. For example, to stream response text:
# 
# ```python
# for token in llm_with_tools.stream("..."):
#     print(token.text(), end="|")
# ```
# 
# See the [streaming guide](/docs/how_to/chat_streaming/) for more detail.
# 
# :::

# The output message will also contain information from any tool invocations:

# In[14]:


response.additional_kwargs


# #### File search
# 
# To trigger a file search, pass a [file search tool](https://platform.openai.com/docs/guides/tools-file-search) to the model as you would another tool. You will need to populate an OpenAI-managed vector store and include the vector store ID in the tool definition. See [OpenAI documentation](https://platform.openai.com/docs/guides/tools-file-search) for more detail.

# In[24]:


llm = ChatOpenAI(model="gpt-4o-mini")

openai_vector_store_ids = [
    "vs_...",  # your IDs here
]

tool = {
    "type": "file_search",
    "vector_store_ids": openai_vector_store_ids,
}
llm_with_tools = llm.bind_tools([tool])

response = llm_with_tools.invoke("What is deep research by OpenAI?")
print(response.text())


# As with [web search](#web-search), the response will include content blocks with citations:

# In[22]:


response.content[0]["annotations"][:2]


# It will also include information from the built-in tool invocations:

# In[20]:


response.additional_kwargs


# #### Computer use
# 
# `ChatOpenAI` supports the `"computer-use-preview"` model, which is a specialized model for the built-in computer use tool. To enable, pass a [computer use tool](https://platform.openai.com/docs/guides/tools-computer-use) as you would pass another tool.
# 
# Currently, tool outputs for computer use are present in `AIMessage.additional_kwargs["tool_outputs"]`. To reply to the computer use tool call, construct a `ToolMessage` with `{"type": "computer_call_output"}` in its `additional_kwargs`. The content of the message will be a screenshot. Below, we demonstrate a simple example.
# 
# First, load two screenshots:

# In[2]:


import base64


def load_png_as_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")


screenshot_1_base64 = load_png_as_base64(
    "/path/to/screenshot_1.png"
)  # perhaps a screenshot of an application
screenshot_2_base64 = load_png_as_base64(
    "/path/to/screenshot_2.png"
)  # perhaps a screenshot of the Desktop


# In[3]:


from langchain_openai import ChatOpenAI

# Initialize model
llm = ChatOpenAI(
    model="computer-use-preview",
    model_kwargs={"truncation": "auto"},
)

# Bind computer-use tool
tool = {
    "type": "computer_use_preview",
    "display_width": 1024,
    "display_height": 768,
    "environment": "browser",
}
llm_with_tools = llm.bind_tools([tool])

# Construct input message
input_message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": (
                "Click the red X to close and reveal my Desktop. "
                "Proceed, no confirmation needed."
            ),
        },
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{screenshot_1_base64}",
        },
    ],
}

# Invoke model
response = llm_with_tools.invoke(
    [input_message],
    reasoning={
        "generate_summary": "concise",
    },
)


# The response will include a call to the computer-use tool in its `additional_kwargs`:

# In[4]:


response.additional_kwargs


# We next construct a ToolMessage with these properties:
# 
# 1. It has a `tool_call_id` matching the `call_id` from the computer-call.
# 2. It has `{"type": "computer_call_output"}` in its `additional_kwargs`.
# 3. Its content is either an `image_url` or an `input_image` output block (see [OpenAI docs](https://platform.openai.com/docs/guides/tools-computer-use#5-repeat) for formatting).

# In[5]:


from langchain_core.messages import ToolMessage

tool_call_id = response.additional_kwargs["tool_outputs"][0]["call_id"]

tool_message = ToolMessage(
    content=[
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{screenshot_2_base64}",
        }
    ],
    # content=f"data:image/png;base64,{screenshot_2_base64}",  # <-- also acceptable
    tool_call_id=tool_call_id,
    additional_kwargs={"type": "computer_call_output"},
)


# We can now invoke the model again using the message history:

# In[6]:


messages = [
    input_message,
    response,
    tool_message,
]

response_2 = llm_with_tools.invoke(
    messages,
    reasoning={
        "generate_summary": "concise",
    },
)


# In[7]:


response_2.text()


# Instead of passing back the entire sequence, we can also use the [previous_response_id](#passing-previous_response_id):

# In[14]:


previous_response_id = response.response_metadata["id"]

response_2 = llm_with_tools.invoke(
    [tool_message],
    previous_response_id=previous_response_id,
    reasoning={
        "generate_summary": "concise",
    },
)


# In[15]:


response_2.text()


# ### Managing conversation state
# 
# The Responses API supports management of [conversation state](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses).
# 
# #### Manually manage state
# 
# You can manage the state manually or using [LangGraph](/docs/tutorials/chatbot/), as with other chat models:

# In[4]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

tool = {"type": "web_search_preview"}
llm_with_tools = llm.bind_tools([tool])

first_query = "What was a positive news story from today?"
messages = [{"role": "user", "content": first_query}]

response = llm_with_tools.invoke(messages)
response_text = response.text()
print(f"{response_text[:100]}... {response_text[-100:]}")


# In[5]:


second_query = (
    "Repeat my question back to me, as well as the last sentence of your answer."
)

messages.extend(
    [
        response,
        {"role": "user", "content": second_query},
    ]
)
second_response = llm_with_tools.invoke(messages)
print(second_response.text())


# :::tip
# 
# You can use [LangGraph](https://langchain-ai.github.io/langgraph/) to manage conversational threads for you in a variety of backends, including in-memory and Postgres. See [this tutorial](/docs/tutorials/chatbot/) to get started.
# 
# :::
# 
# 
# #### Passing `previous_response_id`
# 
# When using the Responses API, LangChain messages will include an `"id"` field in its metadata. Passing this ID to subsequent invocations will continue the conversation. Note that this is [equivalent](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses#openai-apis-for-conversation-state) to manually passing in messages from a billing perspective.

# In[6]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    use_responses_api=True,
)
response = llm.invoke("Hi, I'm Bob.")
print(response.text())


# In[7]:


second_response = llm.invoke(
    "What is my name?",
    previous_response_id=response.response_metadata["id"],
)
print(second_response.text())


# ## Fine-tuning
# 
# You can call fine-tuned OpenAI models by passing in your corresponding `modelName` parameter.
# 
# This generally takes the form of `ft:{OPENAI_MODEL_NAME}:{ORG_NAME}::{MODEL_ID}`. For example:

# In[11]:


fine_tuned_model = ChatOpenAI(
    temperature=0, model_name="ft:gpt-3.5-turbo-0613:langchain::7qTVM5AR"
)

fine_tuned_model.invoke(messages)


# ## Multimodal Inputs
# 
# OpenAI has models that support multimodal inputs. You can pass in images or audio to these models. For more information on how to do this in LangChain, head to the [multimodal inputs](/docs/how_to/multimodal_inputs) docs.
# 
# You can see the list of models that support different modalities in [OpenAI's documentation](https://platform.openai.com/docs/models).
# 
# At the time of this doc's writing, the main OpenAI models you would use would be:
# 
# - Image inputs: `gpt-4o`, `gpt-4o-mini`
# - Audio inputs: `gpt-4o-audio-preview`
# 
# For an example of passing in image inputs, see the [multimodal inputs how-to guide](/docs/how_to/multimodal_inputs).
# 
# Below is an example of passing audio inputs to `gpt-4o-audio-preview`:

# In[8]:


import base64

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-audio-preview",
    temperature=0,
)

with open(
    "../../../../libs/partners/openai/tests/integration_tests/chat_models/audio_input.wav",
    "rb",
) as f:
    # b64 encode it
    audio = f.read()
    audio_b64 = base64.b64encode(audio).decode()


output_message = llm.invoke(
    [
        (
            "human",
            [
                {"type": "text", "text": "Transcribe the following:"},
                # the audio clip says "I'm sorry, but I can't create..."
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                },
            ],
        ),
    ]
)
output_message.content


# ## Predicted output
# 
# :::info
# Requires `langchain-openai>=0.2.6`
# :::
# 
# Some OpenAI models (such as their `gpt-4o` and `gpt-4o-mini` series) support [Predicted Outputs](https://platform.openai.com/docs/guides/latency-optimization#use-predicted-outputs), which allow you to pass in a known portion of the LLM's expected output ahead of time to reduce latency. This is useful for cases such as editing text or code, where only a small part of the model's output will change.
# 
# Here's an example:

# In[3]:


code = """
/// <summary>
/// Represents a user with a first name, last name, and username.
/// </summary>
public class User
{
    /// <summary>
    /// Gets or sets the user's first name.
    /// </summary>
    public string FirstName { get; set; }

    /// <summary>
    /// Gets or sets the user's last name.
    /// </summary>
    public string LastName { get; set; }

    /// <summary>
    /// Gets or sets the user's username.
    /// </summary>
    public string Username { get; set; }
}
"""

llm = ChatOpenAI(model="gpt-4o")
query = (
    "Replace the Username property with an Email property. "
    "Respond only with code, and with no markdown formatting."
)
response = llm.invoke(
    [{"role": "user", "content": query}, {"role": "user", "content": code}],
    prediction={"type": "content", "content": code},
)
print(response.content)
print(response.response_metadata)


# Note that currently predictions are billed as additional tokens and may increase your usage and costs in exchange for this reduced latency.

# ## Audio Generation (Preview)
# 
# :::info
# Requires `langchain-openai>=0.2.3`
# :::
# 
# OpenAI has a new [audio generation feature](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-out) that allows you to use audio inputs and outputs with the `gpt-4o-audio-preview` model.

# In[1]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-audio-preview",
    temperature=0,
    model_kwargs={
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"},
    },
)

output_message = llm.invoke(
    [
        ("human", "Are you made by OpenAI? Just answer yes or no"),
    ]
)


# `output_message.additional_kwargs['audio']` will contain a dictionary like
# ```python
# {
#     'data': '<audio data b64-encoded',
#     'expires_at': 1729268602,
#     'id': 'audio_67127d6a44348190af62c1530ef0955a',
#     'transcript': 'Yes.'
# }
# ```
# and the format will be what was passed in `model_kwargs['audio']['format']`.
# 
# We can also pass this message with audio data back to the model as part of a message history before openai `expires_at` is reached.
# 
# :::note
# Output audio is stored under the `audio` key in `AIMessage.additional_kwargs`, but input content blocks are typed with an `input_audio` type and key in `HumanMessage.content` lists. 
# 
# For more information, see OpenAI's [audio docs](https://platform.openai.com/docs/guides/audio).
# :::

# In[8]:


history = [
    ("human", "Are you made by OpenAI? Just answer yes or no"),
    output_message,
    ("human", "And what is your name? Just give your name."),
]
second_output_message = llm.invoke(history)


# ## API reference
# 
# For detailed documentation of all ChatOpenAI features and configurations head to the API reference: https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html
