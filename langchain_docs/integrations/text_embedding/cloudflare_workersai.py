#!/usr/bin/env python
# coding: utf-8

# # Cloudflare Workers AI
# 
# >[Cloudflare, Inc. (Wikipedia)](https://en.wikipedia.org/wiki/Cloudflare) is an American company that provides content delivery network services, cloud cybersecurity, DDoS mitigation, and ICANN-accredited domain registration services.
# 
# >[Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) allows you to run machine learning models, on the `Cloudflare` network, from your code via REST API.
# 
# >[Cloudflare AI document](https://developers.cloudflare.com/workers-ai/models/text-embeddings/) listed all text embeddings models available.
# 
# ## Setting up
# 
# Both Cloudflare account ID and API token are required. Find how to obtain them from [this document](https://developers.cloudflare.com/workers-ai/get-started/rest-api/).
# 

# In[2]:


import getpass

my_account_id = getpass.getpass("Enter your Cloudflare account ID:\n\n")
my_api_token = getpass.getpass("Enter your Cloudflare API token:\n\n")


# ## Example

# In[1]:


from langchain_community.embeddings.cloudflare_workersai import (
    CloudflareWorkersAIEmbeddings,
)


# In[3]:


embeddings = CloudflareWorkersAIEmbeddings(
    account_id=my_account_id,
    api_token=my_api_token,
    model_name="@cf/baai/bge-small-en-v1.5",
)
# single string embeddings
query_result = embeddings.embed_query("test")
len(query_result), query_result[:3]


# In[4]:


# string embeddings in batches
batch_query_result = embeddings.embed_documents(["test1", "test2", "test3"])
len(batch_query_result), len(batch_query_result[0])


# In[ ]:




