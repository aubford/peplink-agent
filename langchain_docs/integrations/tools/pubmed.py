#!/usr/bin/env python
# coding: utf-8

# # PubMed
#
# >[PubMed®](https://pubmed.ncbi.nlm.nih.gov/) comprises more than 35 million citations for biomedical literature from `MEDLINE`, life science journals, and online books. Citations may include links to full text content from PubMed Central and publisher web sites.
#
# This notebook goes over how to use `PubMed` as a tool.

# In[ ]:


get_ipython().run_line_magic("pip", "install xmltodict")


# In[2]:


from langchain_community.tools.pubmed.tool import PubmedQueryRun


# In[3]:


tool = PubmedQueryRun()


# In[4]:


tool.invoke("What causes lung cancer?")


# In[ ]:
