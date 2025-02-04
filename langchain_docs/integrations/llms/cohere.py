#!/usr/bin/env python
# coding: utf-8

# # Cohere
# 
# :::caution
# You are currently on a page documenting the use of Cohere models as [text completion models](/docs/concepts/text_llms). Many popular Cohere models are [chat completion models](/docs/concepts/chat_models).
# 
# You may be looking for [this page instead](/docs/integrations/chat/cohere/).
# :::
# 
# >[Cohere](https://cohere.ai/about) is a Canadian startup that provides natural language processing models that help companies improve human-machine interactions.
# 
# Head to the [API reference](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.cohere.Cohere.html) for detailed documentation of all attributes and methods.
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/llms/cohere/) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [Cohere](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.cohere.Cohere.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |
# 

# ## Setup
# 
# The integration lives in the `langchain-community` package. We also need to install the `cohere` package itself. We can install these with:
# 
# ### Credentials
# 
# We'll need to get a [Cohere API key](https://cohere.com/) and set the `COHERE_API_KEY` environment variable:

# In[ ]:


import getpass
import os

if "COHERE_API_KEY" not in os.environ:
    os.environ["COHERE_API_KEY"] = getpass.getpass()


# ### Installation

# In[ ]:


pip install -U langchain-community langchain-cohere


# It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability

# In[ ]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# ## Invocation
# 
# Cohere supports all [LLM](/docs/how_to#llms) functionality:

# In[1]:


from langchain_cohere import Cohere
from langchain_core.messages import HumanMessage


# In[2]:


model = Cohere(max_tokens=256, temperature=0.75)


# In[6]:


message = "Knock knock"
model.invoke(message)


# In[8]:


await model.ainvoke(message)


# In[9]:


for chunk in model.stream(message):
    print(chunk, end="", flush=True)


# In[10]:


model.batch([message])


# ## Chaining
# 
# You can also easily combine with a prompt template for easy structuring of user input. We can do this using [LCEL](/docs/concepts/lcel)

# In[12]:


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | model


# In[13]:


chain.invoke({"topic": "bears"})


# ## API reference
# 
# For detailed documentation of all `Cohere` llm features and configurations head to the API reference: https://python.langchain.com/api_reference/community/llms/langchain_community.llms.cohere.Cohere.html
