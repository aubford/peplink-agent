#!/usr/bin/env python
# coding: utf-8

# # LLMRails
#
# Let's load the LLMRails Embeddings class.
#
# To use LLMRails embedding you need to pass api key by argument or set it in environment with `LLM_RAILS_API_KEY` key.
# To gey API Key you need to sign up in https://console.llmrails.com/signup and then go to https://console.llmrails.com/api-keys and copy key from there after creating one key in platform.

# In[1]:


from langchain_community.embeddings import LLMRailsEmbeddings


# In[2]:


embeddings = LLMRailsEmbeddings(model="embedding-english-v1")  # or embedding-multi-v1


# In[3]:


text = "This is a test document."


# To generate embeddings, you can either query an invidivual text, or you can query a list of texts.

# In[4]:


query_result = embeddings.embed_query(text)
query_result[:5]


# In[6]:


doc_result = embeddings.embed_documents([text])
doc_result[0][:5]
