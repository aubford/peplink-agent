#!/usr/bin/env python
# coding: utf-8

# # PubMed
# 
# 
# >[PubMed®](https://pubmed.ncbi.nlm.nih.gov/) by `The National Center for Biotechnology Information, National Library of Medicine` comprises more than 35 million citations for biomedical literature from `MEDLINE`, life science journals, and online books. Citations may include links to full text content from `PubMed Central` and publisher web sites.
# 
# This notebook goes over how to use `PubMed` as a retriever

# In[12]:


from langchain_community.retrievers import PubMedRetriever


# In[34]:


retriever = PubMedRetriever()


# In[35]:


retriever.invoke("chatgpt")


# In[ ]:




