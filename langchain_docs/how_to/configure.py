#!/usr/bin/env python
# coding: utf-8
---
sidebar_position: 7
keywords: [ConfigurableField, configurable_fields, ConfigurableAlternatives, configurable_alternatives, LCEL]
---
# # How to configure runtime chain internals
# 
# :::info Prerequisites
# 
# This guide assumes familiarity with the following concepts:
# - [LangChain Expression Language (LCEL)](/docs/concepts/lcel)
# - [Chaining runnables](/docs/how_to/sequence/)
# - [Binding runtime arguments](/docs/how_to/binding/)
# 
# :::
# 
# Sometimes you may want to experiment with, or even expose to the end user, multiple different ways of doing things within your chains.
# This can include tweaking parameters such as temperature or even swapping out one model for another.
# In order to make this experience as easy as possible, we have defined two methods.
# 
# - A `configurable_fields` method. This lets you configure particular fields of a runnable.
#   - This is related to the [`.bind`](/docs/how_to/binding) method on runnables, but allows you to specify parameters for a given step in a chain at runtime rather than specifying them beforehand.
# - A `configurable_alternatives` method. With this method, you can list out alternatives for any particular runnable that can be set during runtime, and swap them for those specified alternatives.

# ## Configurable Fields
# 
# Let's walk through an example that configures chat model fields like temperature at runtime:

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain langchain-openai')

import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()


# In[2]:


from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)

model.invoke("pick a random number")


# Above, we defined `temperature` as a [`ConfigurableField`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.utils.ConfigurableField.html#langchain_core.runnables.utils.ConfigurableField) that we can set at runtime. To do so, we use the [`with_config`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.with_config) method like this:

# In[3]:


model.with_config(configurable={"llm_temperature": 0.9}).invoke("pick a random number")


# Note that the passed `llm_temperature` entry in the dict has the same key as the `id` of the `ConfigurableField`.
# 
# We can also do this to affect just one step that's part of a chain:

# In[4]:


prompt = PromptTemplate.from_template("Pick a random number above {x}")
chain = prompt | model

chain.invoke({"x": 0})


# In[5]:


chain.with_config(configurable={"llm_temperature": 0.9}).invoke({"x": 0})


# ### With HubRunnables
# 
# This is useful to allow for switching of prompts

# In[6]:


from langchain.runnables.hub import HubRunnable

prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
    owner_repo_commit=ConfigurableField(
        id="hub_commit",
        name="Hub Commit",
        description="The Hub commit to pull from",
    )
)

prompt.invoke({"question": "foo", "context": "bar"})


# In[7]:


prompt.with_config(configurable={"hub_commit": "rlm/rag-prompt-llama"}).invoke(
    {"question": "foo", "context": "bar"}
)


# ## Configurable Alternatives
# 
# 

# The `configurable_alternatives()` method allows us to swap out steps in a chain with an alternative. Below, we swap out one chat model for another:

# In[8]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet langchain-anthropic')

import os
from getpass import getpass

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass()


# In[18]:


from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

llm = ChatAnthropic(
    model="claude-3-haiku-20240307", temperature=0
).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    # This sets a default_key.
    # If we specify this key, the default LLM (ChatAnthropic initialized above) will be used
    default_key="anthropic",
    # This adds a new option, with name `openai` that is equal to `ChatOpenAI()`
    openai=ChatOpenAI(),
    # This adds a new option, with name `gpt4` that is equal to `ChatOpenAI(model="gpt-4")`
    gpt4=ChatOpenAI(model="gpt-4"),
    # You can add more configuration options here
)
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm

# By default it will call Anthropic
chain.invoke({"topic": "bears"})


# In[19]:


# We can use `.with_config(configurable={"llm": "openai"})` to specify an llm to use
chain.with_config(configurable={"llm": "openai"}).invoke({"topic": "bears"})


# In[20]:


# If we use the `default_key` then it uses the default
chain.with_config(configurable={"llm": "anthropic"}).invoke({"topic": "bears"})


# ### With Prompts
# 
# We can do a similar thing, but alternate between prompts
# 

# In[22]:


llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="prompt"),
    # This sets a default_key.
    # If we specify this key, the default prompt (asking for a joke, as initialized above) will be used
    default_key="joke",
    # This adds a new option, with name `poem`
    poem=PromptTemplate.from_template("Write a short poem about {topic}"),
    # You can add more configuration options here
)
chain = prompt | llm

# By default it will write a joke
chain.invoke({"topic": "bears"})


# In[23]:


# We can configure it write a poem
chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "bears"})


# ### With Prompts and LLMs
# 
# We can also have multiple things configurable!
# Here's an example doing that with both prompts and LLMs.

# In[25]:


llm = ChatAnthropic(
    model="claude-3-haiku-20240307", temperature=0
).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    # This sets a default_key.
    # If we specify this key, the default LLM (ChatAnthropic initialized above) will be used
    default_key="anthropic",
    # This adds a new option, with name `openai` that is equal to `ChatOpenAI()`
    openai=ChatOpenAI(),
    # This adds a new option, with name `gpt4` that is equal to `ChatOpenAI(model="gpt-4")`
    gpt4=ChatOpenAI(model="gpt-4"),
    # You can add more configuration options here
)
prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="prompt"),
    # This sets a default_key.
    # If we specify this key, the default prompt (asking for a joke, as initialized above) will be used
    default_key="joke",
    # This adds a new option, with name `poem`
    poem=PromptTemplate.from_template("Write a short poem about {topic}"),
    # You can add more configuration options here
)
chain = prompt | llm

# We can configure it write a poem with OpenAI
chain.with_config(configurable={"prompt": "poem", "llm": "openai"}).invoke(
    {"topic": "bears"}
)


# In[26]:


# We can always just configure only one if we want
chain.with_config(configurable={"llm": "openai"}).invoke({"topic": "bears"})


# ### Saving configurations
# 
# We can also easily save configured chains as their own objects

# In[27]:


openai_joke = chain.with_config(configurable={"llm": "openai"})

openai_joke.invoke({"topic": "bears"})


# ## Next steps
# 
# You now know how to configure a chain's internal steps at runtime.
# 
# To learn more, see the other how-to guides on runnables in this section, including:
# 
# - Using [.bind()](/docs/how_to/binding) as a simpler way to set a runnable's runtime parameters

# In[ ]:




