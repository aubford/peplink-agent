#!/usr/bin/env python
# coding: utf-8

# # Math chain
# 
# This notebook showcases using LLMs and Python REPLs to do complex word math problems.

# In[4]:


from langchain.chains import LLMMathChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
llm_math = LLMMathChain.from_llm(llm, verbose=True)

llm_math.invoke("What is 13 raised to the .3432 power?")


# In[ ]:




