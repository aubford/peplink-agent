#!/usr/bin/env python
# coding: utf-8

# # How to use a SmartLLMChain
# 
# A SmartLLMChain is a form of self-critique chain that can help you if have particularly complex questions to answer. Instead of doing a single LLM pass, it instead performs these 3 steps:
# 1. Ideation: Pass the user prompt n times through the LLM to get n output proposals (called "ideas"), where n is a parameter you can set 
# 2. Critique: The LLM critiques all ideas to find possible flaws and picks the best one 
# 3. Resolve: The LLM tries to improve upon the best idea (as chosen in the critique step) and outputs it. This is then the final output.
# 
# SmartLLMChains are based on the SmartGPT workflow proposed in https://youtu.be/wVzuvf9D9BU.
# 
# Note that SmartLLMChains
# - use more LLM passes (ie n+2 instead of just 1)
# - only work then the underlying LLM has the capability for reflection, which smaller models often don't
# - only work with underlying models that return exactly 1 output, not multiple
# 
# This notebook demonstrates how to use a SmartLLMChain.

# ##### Same LLM for all steps

# In[1]:


import os

os.environ["OPENAI_API_KEY"] = "..."


# In[2]:


from langchain.prompts import PromptTemplate
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_openai import ChatOpenAI


# As example question, we will use "I have a 12 liter jug and a 6 liter jug. I want to measure 6 liters. How do I do it?". This is an example from the original SmartGPT video (https://youtu.be/wVzuvf9D9BU?t=384). While this seems like a very easy question, LLMs struggle do these kinds of questions that involve numbers and physical reasoning.
# 
# As we will see, all 3 initial ideas are completely wrong - even though we're using GPT4! Only when using self-reflection do we get a correct answer. 

# In[ ]:


hard_question = "I have a 12 liter jug and a 6 liter jug. I want to measure 6 liters. How do I do it?"


# So, we first create an LLM and prompt template

# In[3]:


prompt = PromptTemplate.from_template(hard_question)
llm = ChatOpenAI(temperature=0, model_name="gpt-4")


# Now we can create a SmartLLMChain

# In[4]:


chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=3, verbose=True)


# Now we can use the SmartLLM as a drop-in replacement for our LLM. E.g.:

# In[5]:


chain.invoke({})


# ##### Different LLM for different steps

# You can also use different LLMs for the different steps by passing `ideation_llm`, `critique_llm` and `resolve_llm`. You might want to do this to use a more creative (i.e., high-temperature) model for ideation and a more strict (i.e., low-temperature) model for critique and resolution.

# In[8]:


chain = SmartLLMChain(
    ideation_llm=ChatOpenAI(temperature=0.9, model_name="gpt-4"),
    llm=ChatOpenAI(
        temperature=0, model_name="gpt-4"
    ),  # will be used for critique and resolution as no specific llms are given
    prompt=prompt,
    n_ideas=3,
    verbose=True,
)


# In[ ]:




