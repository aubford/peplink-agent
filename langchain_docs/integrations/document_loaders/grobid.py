#!/usr/bin/env python
# coding: utf-8

# # Grobid
# 
# GROBID is a machine learning library for extracting, parsing, and re-structuring raw documents.
# 
# It is designed and expected to be used to parse academic papers, where it works particularly well. Note: if the articles supplied to Grobid are large documents (e.g. dissertations) exceeding a certain number of elements, they might not be processed. 
# 
# This loader uses Grobid to parse PDFs into `Documents` that retain metadata associated with the section of text.
# 
# ---
# The best approach is to install Grobid via docker, see https://grobid.readthedocs.io/en/latest/Grobid-docker/. 
# 
# (Note: additional instructions can be found [here](/docs/integrations/providers/grobid).)
# 
# Once grobid is up-and-running you can interact as described below. 
# 

# Now, we can use the data loader.

# In[ ]:


from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser


# In[4]:


loader = GenericLoader.from_filesystem(
    "../Papers/",
    glob="*",
    suffixes=[".pdf"],
    parser=GrobidParser(segment_sentences=False),
)
docs = loader.load()


# In[5]:


docs[3].page_content


# In[6]:


docs[3].metadata

