#!/usr/bin/env python
# coding: utf-8
---
title: Tagging
sidebar_class_name: hidden
---
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/tagging.ipynb)
# 
# # Classify Text into Labels
# 
# Tagging means labeling a document with classes such as:
# 
# - Sentiment
# - Language
# - Style (formal, informal etc.)
# - Covered topics
# - Political tendency
# 
# ![Image description](../../static/img/tagging.png)
# 
# ## Overview
# 
# Tagging has a few components:
# 
# * `function`: Like [extraction](/docs/tutorials/extraction), tagging uses [functions](https://openai.com/blog/function-calling-and-other-api-updates) to specify how the model should tag a document
# * `schema`: defines how we want to tag the document
# 
# ## Quickstart
# 
# Let's see a very straightforward example of how we can use OpenAI tool calling for tagging in LangChain. We'll use the [`with_structured_output`](/docs/how_to/structured_output) method supported by OpenAI models.

# In[ ]:


pip install --upgrade --quiet langchain-core


# We'll need to load a [chat model](/docs/integrations/chat/):
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs customVarName="llm" />

# In[2]:


# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")


# Let's specify a Pydantic model with a few properties and their expected type in our schema.

# In[3]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


# LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
    Classification
)


# In[8]:


inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

response


# If we want dictionary output, we can just call `.dict()`

# In[10]:


inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

response.dict()


# As we can see in the examples, it correctly interprets what we want.
# 
# The results vary so that we may get, for example, sentiments in different languages ('positive', 'enojado' etc.).
# 
# We will see how to control these results in the next section.

# ## Finer control
# 
# Careful schema definition gives us more control over the model's output. 
# 
# Specifically, we can define:
# 
# - Possible values for each property
# - Description to make sure that the model understands the property
# - Required properties to be returned

# Let's redeclare our Pydantic model to control for each of the previously mentioned aspects using enums:

# In[11]:


class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )


# In[15]:


tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
    Classification
)


# Now the answers will be restricted in a way we expect!

# In[12]:


inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
llm.invoke(prompt)


# In[13]:


inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})
llm.invoke(prompt)


# In[14]:


inp = "Weather is ok here, I can go outside without much more than a coat"
prompt = tagging_prompt.invoke({"input": inp})
llm.invoke(prompt)


# The [LangSmith trace](https://smith.langchain.com/public/38294e04-33d8-4c5a-ae92-c2fe68be8332/r) lets us peek under the hood:
# 
# ![Image description](../../static/img/tagging_trace.png)

# ### Going deeper
# 
# * You can use the [metadata tagger](/docs/integrations/document_transformers/openai_metadata_tagger) document transformer to extract metadata from a LangChain `Document`. 
# * This covers the same basic functionality as the tagging chain, only applied to a LangChain `Document`.
