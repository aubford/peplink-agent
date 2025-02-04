#!/usr/bin/env python
# coding: utf-8

# # How to init any model in one line
# 
# Many LLM applications let end users specify what model provider and model they want the application to be powered by. This requires writing some logic to initialize different [chat models](/docs/concepts/chat_models/) based on some user configuration. The `init_chat_model()` helper method makes it easy to initialize a number of different model integrations without having to worry about import paths and class names.
# 
# :::tip Supported models
# 
# See the [init_chat_model()](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) API reference for a full list of supported integrations.
# 
# Make sure you have the [integration packages](/docs/integrations/chat/) installed for any model providers you want to support. E.g. you should have `langchain-openai` installed to init an OpenAI model.
# 
# :::

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain>=0.2.8 langchain-openai langchain-anthropic langchain-google-vertexai')


# ## Basic usage

# In[2]:


from langchain.chat_models import init_chat_model

# Returns a langchain_openai.ChatOpenAI instance.
gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0)
# Returns a langchain_anthropic.ChatAnthropic instance.
claude_opus = init_chat_model(
    "claude-3-opus-20240229", model_provider="anthropic", temperature=0
)
# Returns a langchain_google_vertexai.ChatVertexAI instance.
gemini_15 = init_chat_model(
    "gemini-1.5-pro", model_provider="google_vertexai", temperature=0
)

# Since all model integrations implement the ChatModel interface, you can use them in the same way.
print("GPT-4o: " + gpt_4o.invoke("what's your name").content + "\n")
print("Claude Opus: " + claude_opus.invoke("what's your name").content + "\n")
print("Gemini 1.5: " + gemini_15.invoke("what's your name").content + "\n")


# ## Inferring model provider
# 
# For common and distinct model names `init_chat_model()` will attempt to infer the model provider. See the [API reference](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) for a full list of inference behavior. E.g. any model that starts with `gpt-3...` or `gpt-4...` will be inferred as using model provider `openai`.

# In[3]:


gpt_4o = init_chat_model("gpt-4o", temperature=0)
claude_opus = init_chat_model("claude-3-opus-20240229", temperature=0)
gemini_15 = init_chat_model("gemini-1.5-pro", temperature=0)


# ## Creating a configurable model
# 
# You can also create a runtime-configurable model by specifying `configurable_fields`. If you don't specify a `model` value, then "model" and "model_provider" be configurable by default.

# In[4]:


configurable_model = init_chat_model(temperature=0)

configurable_model.invoke(
    "what's your name", config={"configurable": {"model": "gpt-4o"}}
)


# In[5]:


configurable_model.invoke(
    "what's your name", config={"configurable": {"model": "claude-3-5-sonnet-20240620"}}
)


# ### Configurable model with default values
# 
# We can create a configurable model with default model values, specify which parameters are configurable, and add prefixes to configurable params:

# In[6]:


first_llm = init_chat_model(
    model="gpt-4o",
    temperature=0,
    configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
    config_prefix="first",  # useful when you have a chain with multiple models
)

first_llm.invoke("what's your name")


# In[7]:


first_llm.invoke(
    "what's your name",
    config={
        "configurable": {
            "first_model": "claude-3-5-sonnet-20240620",
            "first_temperature": 0.5,
            "first_max_tokens": 100,
        }
    },
)


# ### Using a configurable model declaratively
# 
# We can call declarative operations like `bind_tools`, `with_structured_output`, `with_configurable`, etc. on a configurable model and chain a configurable model in the same way that we would a regularly instantiated chat model object.

# In[8]:


from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


class GetPopulation(BaseModel):
    """Get the current population in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm = init_chat_model(temperature=0)
llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])

llm_with_tools.invoke(
    "what's bigger in 2024 LA or NYC", config={"configurable": {"model": "gpt-4o"}}
).tool_calls


# In[9]:


llm_with_tools.invoke(
    "what's bigger in 2024 LA or NYC",
    config={"configurable": {"model": "claude-3-5-sonnet-20240620"}},
).tool_calls

