#!/usr/bin/env python
# coding: utf-8

# # GitBook
# 
# >[GitBook](https://docs.gitbook.com/) is a modern documentation platform where teams can document everything from products to internal knowledge bases and APIs.
# 
# This notebook shows how to pull page data from any `GitBook`.

# In[1]:


from langchain_community.document_loaders import GitbookLoader


# ### Load from single GitBook page

# In[2]:


loader = GitbookLoader("https://docs.gitbook.com")


# In[3]:


page_data = loader.load()


# In[4]:


page_data


# ### Load from all paths in a given GitBook
# For this to work, the GitbookLoader needs to be initialized with the root path (`https://docs.gitbook.com` in this example) and have `load_all_paths` set to `True`.

# In[6]:


loader = GitbookLoader("https://docs.gitbook.com", load_all_paths=True)
all_pages_data = loader.load()


# In[7]:


print(f"fetched {len(all_pages_data)} documents.")
# show second document
all_pages_data[2]


# In[ ]:




