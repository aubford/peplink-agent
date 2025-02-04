#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: AI21 Labs
---
# # AI21LLM
# 
# :::caution This service is deprecated.
# See [this page](https://python.langchain.com/docs/integrations/chat/ai21/) for the updated ChatAI21 object. :::
# 
# This example goes over how to use LangChain to interact with `AI21` Jurassic models. To use the Jamba model, use the [ChatAI21 object](https://python.langchain.com/docs/integrations/chat/ai21/) instead.
# 
# [See a full list of AI21 models and tools on LangChain.](https://pypi.org/project/langchain-ai21/)
# 
# ## Installation

# In[4]:


get_ipython().system('pip install -qU langchain-ai21')


# ## Environment Setup
# 
# We'll need to get a [AI21 API key](https://docs.ai21.com/) and set the `AI21_API_KEY` environment variable:

# In[5]:


import os
from getpass import getpass

if "AI21_API_KEY" not in os.environ:
    os.environ["AI21_API_KEY"] = getpass()


# ## Usage

# In[6]:


from langchain_ai21 import AI21LLM
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

model = AI21LLM(model="j2-ultra")

chain = prompt | model

chain.invoke({"question": "What is LangChain?"})


# # AI21 Contextual Answer
# 
# You can use AI21's contextual answers model to receives text or document, serving as a context,
# and a question and returns an answer based entirely on this context.
# 
# This means that if the answer to your question is not in the document,
# the model will indicate it (instead of providing a false answer)

# In[9]:


from langchain_ai21 import AI21ContextualAnswers

tsm = AI21ContextualAnswers()

response = tsm.invoke(input={"context": "Your context", "question": "Your question"})


# You can also use it with chains and output parsers and vector DBs

# In[10]:


from langchain_ai21 import AI21ContextualAnswers
from langchain_core.output_parsers import StrOutputParser

tsm = AI21ContextualAnswers()
chain = tsm | StrOutputParser()

response = chain.invoke(
    {"context": "Your context", "question": "Your question"},
)

