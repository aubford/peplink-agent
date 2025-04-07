#!/usr/bin/env python
# coding: utf-8

# # Fireworks
# 
# :::caution
# You are currently on a page documenting the use of Fireworks models as [text completion models](/docs/concepts/text_llms). Many popular Fireworks models are [chat completion models](/docs/concepts/chat_models).
# 
# You may be looking for [this page instead](/docs/integrations/chat/fireworks/).
# :::
# 
# >[Fireworks](https://app.fireworks.ai/) accelerates product development on generative AI by creating an innovative AI experiment and production platform. 
# 
# This example goes over how to use LangChain to interact with `Fireworks` models.
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/v0.1/docs/integrations/llms/fireworks/) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [Fireworks](https://python.langchain.com/api_reference/fireworks/llms/langchain_fireworks.llms.Fireworks.html#langchain_fireworks.llms.Fireworks) | [langchain_fireworks](https://python.langchain.com/api_reference/fireworks/index.html) | ❌ | ❌ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_fireworks?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_fireworks?style=flat-square&label=%20) |

# ## Setup
# 
# ### Credentials 
# 
# Sign in to [Fireworks AI](http://fireworks.ai) for the an API Key to access our models, and make sure it is set as the `FIREWORKS_API_KEY` environment variable.
# 3. Set up your model using a model id. If the model is not set, the default model is fireworks-llama-v2-7b-chat. See the full, most up-to-date model list on [fireworks.ai](https://fireworks.ai).

# In[1]:


import getpass
import os

if "FIREWORKS_API_KEY" not in os.environ:
    os.environ["FIREWORKS_API_KEY"] = getpass.getpass("Fireworks API Key:")


# ### Installation
# 
# You need to install the `langchain_fireworks` python package for the rest of the notebook to work.

# In[2]:


get_ipython().run_line_magic('pip', 'install -qU langchain-fireworks')


# ## Instantiation

# In[3]:


from langchain_fireworks import Fireworks

# Initialize a Fireworks model
llm = Fireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    base_url="https://api.fireworks.ai/inference/v1/completions",
)


# ## Invocation
# 
# You can call the model directly with string prompts to get completions.

# In[4]:


output = llm.invoke("Who's the best quarterback in the NFL?")
print(output)


# ### Invoking with multiple prompts

# In[5]:


# Calling multiple prompts
output = llm.generate(
    [
        "Who's the best cricket player in 2016?",
        "Who's the best basketball player in the league?",
    ]
)
print(output.generations)


# ### Invoking with additional parameters

# In[6]:


# Setting additional parameters: temperature, max_tokens, top_p
llm = Fireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    temperature=0.7,
    max_tokens=15,
    top_p=1.0,
)
print(llm.invoke("What's the weather like in Kansas City in December?"))


# ## Chaining

# You can use the LangChain Expression Language to create a simple chain with non-chat models.

# In[7]:


from langchain_core.prompts import PromptTemplate
from langchain_fireworks import Fireworks

llm = Fireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    temperature=0.7,
    max_tokens=15,
    top_p=1.0,
)
prompt = PromptTemplate.from_template("Tell me a joke about {topic}?")
chain = prompt | llm

print(chain.invoke({"topic": "bears"}))


# ## Streaming
# 
# You can stream the output, if you want.

# In[8]:


for token in chain.stream({"topic": "bears"}):
    print(token, end="", flush=True)


# ## API reference
# 
# For detailed documentation of all `Fireworks` LLM features and configurations head to the API reference: https://python.langchain.com/api_reference/fireworks/llms/langchain_fireworks.llms.Fireworks.html#langchain_fireworks.llms.Fireworks
