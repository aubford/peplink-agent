#!/usr/bin/env python
# coding: utf-8

# # Program-aided language model (PAL) chain
#
# Implements Program-Aided Language Models, as in https://arxiv.org/pdf/2211.10435.pdf.
#

# In[1]:


from langchain_experimental.pal_chain import PALChain
from langchain_openai import OpenAI


# In[ ]:


llm = OpenAI(temperature=0, max_tokens=512)


# ## Math Prompt

# In[2]:


pal_chain = PALChain.from_math_prompt(llm, verbose=True)


# In[3]:


question = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?"


# In[4]:


pal_chain.run(question)


# ## Colored Objects

# In[5]:


pal_chain = PALChain.from_colored_object_prompt(llm, verbose=True)


# In[6]:


question = "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses. If I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"


# In[7]:


pal_chain.run(question)


# ## Intermediate Steps
# You can also use the intermediate steps flag to return the code executed that generates the answer.

# In[5]:


pal_chain = PALChain.from_colored_object_prompt(
    llm, verbose=True, return_intermediate_steps=True
)


# In[6]:


question = "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses. If I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"


# In[8]:


result = pal_chain({"question": question})


# In[11]:


result["intermediate_steps"]


# In[ ]:
