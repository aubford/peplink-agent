#!/usr/bin/env python
# coding: utf-8

# # How to save and load LangChain objects
#
# LangChain classes implement standard methods for serialization. Serializing LangChain objects using these methods confer some advantages:
#
# - Secrets, such as API keys, are separated from other parameters and can be loaded back to the object on de-serialization;
# - De-serialization is kept compatible across package versions, so objects that were serialized with one version of LangChain can be properly de-serialized with another.
#
# To save and load LangChain objects using this system, use the `dumpd`, `dumps`, `load`, and `loads` functions in the [load module](https://python.langchain.com/api_reference/core/load.html) of `langchain-core`. These functions support JSON and JSON-serializable objects.
#
# All LangChain objects that inherit from [Serializable](https://python.langchain.com/api_reference/core/load/langchain_core.load.serializable.Serializable.html) are JSON-serializable. Examples include [messages](https://python.langchain.com/api_reference//python/core_api_reference.html#module-langchain_core.messages), [document objects](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) (e.g., as returned from [retrievers](/docs/concepts/retrievers)), and most [Runnables](/docs/concepts/lcel), such as chat models, retrievers, and [chains](/docs/how_to/sequence) implemented with the LangChain Expression Language.
#
# Below we walk through an example with a simple [LLM chain](/docs/tutorials/llm_chain).
#
# :::caution
#
# De-serialization using `load` and `loads` can instantiate any serializable LangChain object. Only use this feature with trusted inputs!
#
# De-serialization is a beta feature and is subject to change.
# :::

# In[12]:


from langchain_core.load import dumpd, dumps, load, loads
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following into {language}:"),
        ("user", "{text}"),
    ],
)

llm = ChatOpenAI(model="gpt-4o-mini", api_key="llm-api-key")

chain = prompt | llm


# ## Saving objects
#
# ### To json

# In[2]:


string_representation = dumps(chain, pretty=True)
print(string_representation[:500])


# ### To a json-serializable Python dict

# In[3]:


dict_representation = dumpd(chain)

print(type(dict_representation))


# ### To disk

# In[4]:


import json

with open("/tmp/chain.json", "w") as fp:
    json.dump(string_representation, fp)


# Note that the API key is withheld from the serialized representations. Parameters that are considered secret are specified by the `.lc_secrets` attribute of the LangChain object:

# In[5]:


chain.last.lc_secrets


# ## Loading objects
#
# Specifying `secrets_map` in `load` and `loads` will load the corresponding secrets onto the de-serialized LangChain object.
#
# ### From string

# In[7]:


chain = loads(string_representation, secrets_map={"OPENAI_API_KEY": "llm-api-key"})


# ### From dict

# In[9]:


chain = load(dict_representation, secrets_map={"OPENAI_API_KEY": "llm-api-key"})


# ### From disk

# In[10]:


with open("/tmp/chain.json", "r") as fp:
    chain = loads(json.load(fp), secrets_map={"OPENAI_API_KEY": "llm-api-key"})


# Note that we recover the API key specified at the start of the guide:

# In[11]:


chain.last.openai_api_key.get_secret_value()


# In[ ]:
