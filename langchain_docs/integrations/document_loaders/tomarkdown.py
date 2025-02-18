#!/usr/bin/env python
# coding: utf-8

# # 2Markdown
#
# >[2markdown](https://2markdown.com/) service transforms website content into structured markdown files.
#

# In[2]:


# You will need to get your own API key. See https://2markdown.com/login

api_key = ""


# In[3]:


from langchain_community.document_loaders import ToMarkdownLoader


# In[8]:


loader = ToMarkdownLoader(url="/docs/get_started/introduction", api_key=api_key)


# In[9]:


docs = loader.load()


# In[10]:


print(docs[0].page_content)


# In[ ]:
