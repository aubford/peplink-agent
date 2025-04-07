#!/usr/bin/env python
# coding: utf-8

# # Clova Embeddings
# [Clova](https://api.ncloud-docs.com/docs/ai-naver-clovastudio-summary) offers an embeddings service
# 
# This example goes over how to use LangChain to interact with Clova inference for text embedding.
# 

# In[ ]:


import os

os.environ["CLOVA_EMB_API_KEY"] = ""
os.environ["CLOVA_EMB_APIGW_API_KEY"] = ""
os.environ["CLOVA_EMB_APP_ID"] = ""


# In[ ]:


from langchain_community.embeddings import ClovaEmbeddings


# In[ ]:


embeddings = ClovaEmbeddings()


# In[ ]:


query_text = "This is a test query."
query_result = embeddings.embed_query(query_text)


# In[ ]:


document_text = ["This is a test doc1.", "This is a test doc2."]
document_result = embeddings.embed_documents(document_text)

