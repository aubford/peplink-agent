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
# :::info Requires ``langchain-openai>=0.1.21rc1``
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
