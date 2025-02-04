#!/usr/bin/env python
# coding: utf-8

# # PubMed
# 
# >[PubMed®](https://pubmed.ncbi.nlm.nih.gov/) by `The National Center for Biotechnology Information, National Library of Medicine` comprises more than 35 million citations for biomedical literature from `MEDLINE`, life science journals, and online books. Citations may include links to full text content from `PubMed Central` and publisher web sites.

# In[1]:


from langchain_community.document_loaders import PubMedLoader


# In[4]:


loader = PubMedLoader("chatgpt")


# In[5]:


docs = loader.load()


# In[6]:


len(docs)


# In[8]:


docs[1].metadata


# In[9]:


docs[1].page_content


# In[ ]:




