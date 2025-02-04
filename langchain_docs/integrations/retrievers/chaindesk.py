#!/usr/bin/env python
# coding: utf-8

# # Chaindesk
# 
# >[Chaindesk platform](https://docs.chaindesk.ai/introduction) brings data from anywhere (Datsources: Text, PDF, Word, PowerPpoint, Excel, Notion, Airtable, Google Sheets, etc..) into Datastores (container of multiple Datasources).
# Then your Datastores can be connected to ChatGPT via Plugins or any other Large Langue Model (LLM) via the `Chaindesk API`.
# 
# This notebook shows how to use [Chaindesk's](https://www.chaindesk.ai/) retriever.
# 
# First, you will need to sign up for Chaindesk, create a datastore, add some data and get your datastore api endpoint url. You need the [API Key](https://docs.chaindesk.ai/api-reference/authentication).

# In[ ]:





# ## Query
# 
# Now that our index is set up, we can set up a retriever and start querying it.

# In[1]:


from langchain_community.retrievers import ChaindeskRetriever


# In[2]:


retriever = ChaindeskRetriever(
    datastore_url="https://clg1xg2h80000l708dymr0fxc.chaindesk.ai/query",
    # api_key="CHAINDESK_API_KEY", # optional if datastore is public
    # top_k=10 # optional
)


# In[6]:


retriever.invoke("What is Daftpage?")

