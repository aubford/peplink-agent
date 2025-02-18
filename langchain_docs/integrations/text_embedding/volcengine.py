#!/usr/bin/env python
# coding: utf-8

# # Volc Engine
#
# This notebook provides you with a guide on how to load the Volcano Embedding class.
#
#
# ## API Initialization
#
# To use the LLM services based on [VolcEngine](https://www.volcengine.com/docs/82379/1099455), you have to initialize these parameters:
#
# You could either choose to init the AK,SK in environment variables or init params:
#
# ```base
# export VOLC_ACCESSKEY=XXX
# export VOLC_SECRETKEY=XXX
# ```

# In[1]:


"""For basic init and call"""
import os

from langchain_community.embeddings import VolcanoEmbeddings

os.environ["VOLC_ACCESSKEY"] = ""
os.environ["VOLC_SECRETKEY"] = ""

embed = VolcanoEmbeddings(volcano_ak="", volcano_sk="")
print("embed_documents result:")
res1 = embed.embed_documents(["foo", "bar"])
for r in res1:
    print("", r[:8])


# In[2]:


print("embed_query result:")
res2 = embed.embed_query("foo")
print("", r[:8])


# In[ ]:
