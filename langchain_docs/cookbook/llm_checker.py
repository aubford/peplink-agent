#!/usr/bin/env python
# coding: utf-8

# # Self-checking chain
# This notebook showcases how to use LLMCheckerChain.

# In[1]:


from langchain.chains import LLMCheckerChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.7)

text = "What type of mammal lays the biggest eggs?"

checker_chain = LLMCheckerChain.from_llm(llm, verbose=True)

checker_chain.invoke(text)


# In[ ]:
