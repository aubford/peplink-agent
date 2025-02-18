#!/usr/bin/env python
# coding: utf-8

# # Bookend AI
#
# Let's load the Bookend AI Embeddings class.

# In[ ]:


from langchain_community.embeddings import BookendEmbeddings


# In[ ]:


embeddings = BookendEmbeddings(
    domain="your_domain",
    api_token="your_api_token",
    model_id="your_embeddings_model_id",
)


# In[ ]:


text = "This is a test document."


# In[ ]:


query_result = embeddings.embed_query(text)


# In[ ]:


doc_result = embeddings.embed_documents([text])
