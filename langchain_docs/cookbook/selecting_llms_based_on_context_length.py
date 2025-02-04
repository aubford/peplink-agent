#!/usr/bin/env python
# coding: utf-8

# # Selecting LLMs based on Context Length
# 
# Different LLMs have different context lengths. As a very immediate an practical example, OpenAI has two versions of GPT-3.5-Turbo: one with 4k context, another with 16k context. This notebook shows how to route between them based on input.

# In[24]:


from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_openai import ChatOpenAI


# In[3]:


short_context_model = ChatOpenAI(model="gpt-3.5-turbo")
long_context_model = ChatOpenAI(model="gpt-3.5-turbo-16k")


# In[4]:


def get_context_length(prompt: PromptValue):
    messages = prompt.to_messages()
    tokens = short_context_model.get_num_tokens_from_messages(messages)
    return tokens


# In[5]:


prompt = PromptTemplate.from_template("Summarize this passage: {context}")


# In[20]:


def choose_model(prompt: PromptValue):
    context_len = get_context_length(prompt)
    if context_len < 30:
        print("short model")
        return short_context_model
    else:
        print("long model")
        return long_context_model


# In[25]:


chain = prompt | choose_model | StrOutputParser()


# In[26]:


chain.invoke({"context": "a frog went to a pond"})


# In[27]:


chain.invoke(
    {"context": "a frog went to a pond and sat on a log and went to a different pond"}
)


# In[ ]:




