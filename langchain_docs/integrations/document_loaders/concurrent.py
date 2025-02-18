#!/usr/bin/env python
# coding: utf-8

# # Concurrent Loader
#
# Works just like the GenericLoader but concurrently for those who choose to optimize their workflow.
#

# In[3]:


from langchain_community.document_loaders import ConcurrentLoader


# In[8]:


loader = ConcurrentLoader.from_filesystem("example_data/", glob="**/*.txt")


# In[9]:


files = loader.load()


# In[12]:


len(files)


# In[ ]:
