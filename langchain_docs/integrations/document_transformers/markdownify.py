#!/usr/bin/env python
# coding: utf-8

# # Markdownify

# > [markdownify](https://github.com/matthewwithanm/python-markdownify) is a Python package that converts HTML documents to Markdown format with customizable options for handling tags (links, images, ...), heading styles and other.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  markdownify")


# In[1]:


from langchain_community.document_loaders import AsyncHtmlLoader

urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()


# In[2]:


docs


# In[3]:


from langchain_community.document_transformers import MarkdownifyTransformer


# In[4]:


md = MarkdownifyTransformer()
converted_docs = md.transform_documents(docs)

print(converted_docs[0].page_content[:1000])


# In[5]:


md = MarkdownifyTransformer(strip="a")
converted_docs = md.transform_documents(docs)

print(converted_docs[0].page_content[:1000])


# In[6]:


md = MarkdownifyTransformer(strip=["h1", "a"])
converted_docs = md.transform_documents(docs)

print(converted_docs[0].page_content[:1000])
