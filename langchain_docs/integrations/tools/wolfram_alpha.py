#!/usr/bin/env python
# coding: utf-8

# # Wolfram Alpha
# 
# This notebook goes over how to use the wolfram alpha component.
# 
# First, you need to set up your Wolfram Alpha developer account and get your APP ID:
# 
# 1. Go to wolfram alpha and sign up for a developer account [here](https://developer.wolframalpha.com/)
# 2. Create an app and get your APP ID
# 3. pip install wolframalpha
# 
# Then we will need to set some environment variables:
# 1. Save your APP ID into WOLFRAM_ALPHA_APPID env variable

# In[ ]:


pip install wolframalpha


# In[6]:


import os

os.environ["WOLFRAM_ALPHA_APPID"] = ""


# In[9]:


from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper


# In[10]:


wolfram = WolframAlphaAPIWrapper()


# In[11]:


wolfram.run("What is 2x+5 = -3x + 7?")


# In[ ]:




