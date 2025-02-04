#!/usr/bin/env python
# coding: utf-8

# # HTML to text
# 
# >[html2text](https://github.com/Alir3z4/html2text/) is a Python package that converts a page of `HTML` into clean, easy-to-read plain `ASCII text`. 
# 
# The ASCII also happens to be a valid `Markdown` (a text-to-HTML format).

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet html2text')


# In[2]:


from langchain_community.document_loaders import AsyncHtmlLoader

urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()


# In[3]:


from langchain_community.document_transformers import Html2TextTransformer

urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

print(docs_transformed[0].page_content[1000:2000])

print(docs_transformed[1].page_content[1000:2000])

