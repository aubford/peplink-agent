#!/usr/bin/env python
# coding: utf-8

# # LLM Symbolic Math 
# This notebook showcases using LLMs and Python to Solve Algebraic Equations. Under the hood is makes use of [SymPy](https://www.sympy.org/en/index.html).

# In[3]:


from langchain_experimental.llm_symbolic_math.base import LLMSymbolicMathChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
llm_symbolic_math = LLMSymbolicMathChain.from_llm(llm)


# ## Integrals and derivates

# In[4]:


llm_symbolic_math.invoke("What is the derivative of sin(x)*exp(x) with respect to x?")


# In[5]:


llm_symbolic_math.invoke(
    "What is the integral of exp(x)*sin(x) + exp(x)*cos(x) with respect to x?"
)


# ## Solve linear and differential equations

# In[6]:


llm_symbolic_math.invoke('Solve the differential equation y" - y = e^t')


# In[7]:


llm_symbolic_math.invoke("What are the solutions to this equation y^3 + 1/3y?")


# In[8]:


llm_symbolic_math.invoke("x = y + 5, y = z - 3, z = x * y. Solve for x, y, z")

