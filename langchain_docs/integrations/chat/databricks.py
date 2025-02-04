#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Databricks
---
# # ChatDatabricks
# 
# > [Databricks](https://www.databricks.com/) Lakehouse Platform unifies data, analytics, and AI on one platform. 
# 
# This notebook provides a quick overview for getting started with Databricks [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatDatabricks features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.databricks.ChatDatabricks.html).
# 
# ## Overview
# 
# `ChatDatabricks` class wraps a chat model endpoint hosted on [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html). This example notebook shows how to wrap your serving endpoint and use it as a chat model in your LangChain application.
# 
# ### Integration details
# 
# | Class | Package | Local | Serializable | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: |
# | [ChatDatabricks](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.databricks.ChatDatabricks.html) | [databricks-langchain](https://python.langchain.com/docs/integrations/providers/databricks/) | ❌ | beta | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-databricks?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-databricks?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling/) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |  ✅ | ✅ | ✅ | ❌ | 
# 
# ### Supported Methods
# 
# `ChatDatabricks` supports all methods of `ChatModel` including async APIs.
# 
# 
# ### Endpoint Requirement
# 
# The serving endpoint `ChatDatabricks` wraps must have OpenAI-compatible chat input/output format ([reference](https://mlflow.org/docs/latest/llms/deployments/index.html#chat)). As long as the input format is compatible, `ChatDatabricks` can be used for any endpoint type hosted on [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html):
# 
# 1. Foundation Models - Curated list of state-of-the-art foundation models such as DRBX, Llama3, Mixtral-8x7B, and etc. These endpoint are ready to use in your Databricks workspace without any set up.
# 2. Custom Models - You can also deploy custom models to a serving endpoint via MLflow with
# your choice of framework such as LangChain, Pytorch, Transformers, etc.
# 3. External Models - Databricks endpoints can serve models that are hosted outside Databricks as a proxy, such as proprietary model service like OpenAI GPT4.
# 

# ## Setup
# 
# To access Databricks models you'll need to create a Databricks account, set up credentials (only if you are outside Databricks workspace), and install required packages.
# 
# ### Credentials (only if you are outside Databricks)
# 
# If you are running LangChain app inside Databricks, you can skip this step.
# 
# Otherwise, you need manually set the Databricks workspace hostname and personal access token to `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables, respectively. See [Authentication Documentation](https://docs.databricks.com/en/dev-tools/auth/index.html#databricks-personal-access-tokens) for how to get an access token.

# In[2]:


import getpass
import os

os.environ["DATABRICKS_HOST"] = "https://your-workspace.cloud.databricks.com"
if "DATABRICKS_TOKEN" not in os.environ:
    os.environ["DATABRICKS_TOKEN"] = getpass.getpass(
        "Enter your Databricks access token: "
    )


# ### Installation
# 
# The LangChain Databricks integration lives in the `databricks-langchain` package.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU databricks-langchain')


# We first demonstrates how to query DBRX-instruct model hosted as Foundation Models endpoint with `ChatDatabricks`.
# 
# For other type of endpoints, there are some difference in how to set up the endpoint itself, however, once the endpoint is ready, there is no difference in how to query it with `ChatDatabricks`. Please refer to the bottom of this notebook for the examples with other type of endpoints.

# ## Instantiation
# 

# In[ ]:


from databricks_langchain import ChatDatabricks

chat_model = ChatDatabricks(
    endpoint="databricks-dbrx-instruct",
    temperature=0.1,
    max_tokens=256,
    # See https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.databricks.ChatDatabricks.html for other supported parameters
)


# ## Invocation

# In[5]:


chat_model.invoke("What is MLflow?")


# In[6]:


# You can also pass a list of messages
messages = [
    ("system", "You are a chatbot that can answer questions about Databricks."),
    ("user", "What is Databricks Model Serving?"),
]
chat_model.invoke(messages)


# ## Chaining
# Similar to other chat models, `ChatDatabricks` can be used as a part of a complex chain.

# In[6]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a chatbot that can answer questions about {topic}.",
        ),
        ("user", "{question}"),
    ]
)

chain = prompt | chat_model
chain.invoke(
    {
        "topic": "Databricks",
        "question": "What is Unity Catalog?",
    }
)


# ## Invocation (streaming)

# In[5]:


for chunk in chat_model.stream("How are you?"):
    print(chunk.content, end="|")


# ## Async Invocation

# In[ ]:


import asyncio

country = ["Japan", "Italy", "Australia"]
futures = [chat_model.ainvoke(f"Where is the capital of {c}?") for c in country]
await asyncio.gather(*futures)


# ## Tool calling
# 
# ChatDatabricks supports OpenAI-compatible tool calling API that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool. tool-calling is extremely useful for building tool-using chains and agents, and for getting structured outputs from models more generally.
# 
# With `ChatDatabricks.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model. Under the hood these are converted to the OpenAI-compatible tool schemas, which looks like:
# 
# ```
# {
#     "name": "...",
#     "description": "...",
#     "parameters": {...}  # JSONSchema
# }
# ```
# 
# and passed in every model invocation.

# In[1]:


from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


class GetPopulation(BaseModel):
    """Get the current population in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = chat_model.bind_tools([GetWeather, GetPopulation])
ai_msg = llm_with_tools.invoke(
    "Which city is hotter today and which is bigger: LA or NY?"
)
print(ai_msg.tool_calls)


# ## Wrapping Custom Model Endpoint
# 
# Prerequisites:
# 
# * An LLM was registered and deployed to [a Databricks serving endpoint](https://docs.databricks.com/machine-learning/model-serving/index.html) via MLflow. The endpoint must have OpenAI-compatible chat input/output format ([reference](https://mlflow.org/docs/latest/llms/deployments/index.html#chat))
# * You have ["Can Query" permission](https://docs.databricks.com/security/auth-authz/access-control/serving-endpoint-acl.html) to the endpoint.
# 
# Once the endpoint is ready, the usage pattern is identical to that of Foundation Models.

# In[ ]:


chat_model_custom = ChatDatabricks(
    endpoint="YOUR_ENDPOINT_NAME",
    temperature=0.1,
    max_tokens=256,
)

chat_model_custom.invoke("How are you?")


# ## Wrapping External Models

# Prerequisite: Create Proxy Endpoint
# 
# First, create a new Databricks serving endpoint that proxies requests to the target external model. The endpoint creation should be fairy quick for proxying external models.
# 
# This requires registering your OpenAI API Key within the Databricks secret manager as follows:
# ```sh
# # Replace `<scope>` with your scope
# databricks secrets create-scope <scope>
# databricks secrets put-secret <scope> openai-api-key --string-value $OPENAI_API_KEY
# ```
# 
# For how to set up Databricks CLI and manage secrets, please refer to https://docs.databricks.com/en/security/secrets/secrets.html

# In[ ]:


from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

secret = "secrets/<scope>/openai-api-key"  # replace `<scope>` with your scope
endpoint_name = "my-chat"  # rename this if my-chat already exists
client.create_endpoint(
    name=endpoint_name,
    config={
        "served_entities": [
            {
                "name": "my-chat",
                "external_model": {
                    "name": "gpt-3.5-turbo",
                    "provider": "openai",
                    "task": "llm/v1/chat",
                    "openai_config": {
                        "openai_api_key": "{{" + secret + "}}",
                    },
                },
            }
        ],
    },
)


# Once the endpoint status has become "Ready", you can query the endpoint in the same way as other types of endpoints.

# In[ ]:


chat_model_external = ChatDatabricks(
    endpoint=endpoint_name,
    temperature=0.1,
    max_tokens=256,
)
chat_model_external.invoke("How to use Databricks?")


# ## Function calling on Databricks

# Databricks Function Calling is OpenAI-compatible and is only available during model serving as part of Foundation Model APIs.
# 
# See [Databricks function calling introduction](https://docs.databricks.com/en/machine-learning/model-serving/function-calling.html#supported-models) for supported models.

# In[ ]:


llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
            },
        },
    }
]

# supported tool_choice values: "auto", "required", "none", function name in string format,
# or a dictionary as {"type": "function", "function": {"name": <<tool_name>>}}
model = llm.bind_tools(tools, tool_choice="auto")

messages = [{"role": "user", "content": "What is the current temperature of Chicago?"}]
print(model.invoke(messages))


# See [Databricks Unity Catalog](docs/integrations/tools/databricks.ipynb) about how to use UC functions in chains.

# ## API reference
# 
# For detailed documentation of all ChatDatabricks features and configurations head to the API reference: https://python.langchain.com/api_reference/databricks/chat_models/langchain_databricks.chat_models.ChatDatabricks.html
