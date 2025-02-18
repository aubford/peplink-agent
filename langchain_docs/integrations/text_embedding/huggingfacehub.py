#!/usr/bin/env python
# coding: utf-8

# # Hugging Face
# Let's load the Hugging Face Embedding class.

# In[ ]:


get_ipython().run_line_magic(
    "pip",
    "install --upgrade --quiet  langchain langchain-huggingface sentence_transformers",
)


# In[2]:


from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# In[3]:


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# In[3]:


text = "This is a test document."


# In[5]:


query_result = embeddings.embed_query(text)


# In[6]:


query_result[:3]


# In[7]:


doc_result = embeddings.embed_documents([text])


# ## Hugging Face Inference API
# We can also access embedding models via the Hugging Face Inference API, which does not require us to install ``sentence_transformers`` and download models locally.

# In[1]:


import getpass

inference_api_key = getpass.getpass("Enter your HF Inference API Key:\n\n")


# In[4]:


from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

query_result = embeddings.embed_query(text)
query_result[:3]


# ## Hugging Face Hub
# We can also generate embeddings locally via the Hugging Face Hub package, which requires us to install ``huggingface_hub ``

# In[ ]:


get_ipython().system("pip install huggingface_hub")


# In[ ]:


from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings


# In[ ]:


embeddings = HuggingFaceEndpointEmbeddings()


# In[ ]:


text = "This is a test document."


# In[ ]:


query_result = embeddings.embed_query(text)


# In[ ]:


query_result[:3]
