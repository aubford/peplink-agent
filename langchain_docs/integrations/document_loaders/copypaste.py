#!/usr/bin/env python
# coding: utf-8

# # Copy Paste
# 
# This notebook covers how to load a document object from something you just want to copy and paste. In this case, you don't even need to use a DocumentLoader, but rather can just construct the Document directly.

# In[1]:


from langchain_core.documents import Document


# In[2]:


text = "..... put the text you copy pasted here......"


# In[3]:


doc = Document(page_content=text)


# ## Metadata
# If you want to add metadata about the where you got this piece of text, you easily can with the metadata key.

# In[4]:


metadata = {"source": "internet", "date": "Friday"}


# In[5]:


doc = Document(page_content=text, metadata=metadata)


# In[ ]:




