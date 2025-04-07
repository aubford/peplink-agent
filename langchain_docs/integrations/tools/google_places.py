#!/usr/bin/env python
# coding: utf-8

# # Google Places
# 
# This notebook goes through how to use Google Places API

# In[10]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  googlemaps langchain-community')


# In[12]:


import os

os.environ["GPLACES_API_KEY"] = ""


# In[13]:


from langchain_community.tools import GooglePlacesTool


# In[14]:


places = GooglePlacesTool()


# In[16]:


places.run("al fornos")


# In[ ]:




