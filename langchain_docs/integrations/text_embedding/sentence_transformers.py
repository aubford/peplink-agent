#!/usr/bin/env python
# coding: utf-8

# # Sentence Transformers on Hugging Face
#
# >[Hugging Face sentence-transformers](https://huggingface.co/sentence-transformers) is a Python framework for state-of-the-art sentence, text and image embeddings.
# >You can use these embedding models from the `HuggingFaceEmbeddings` class.
#
# :::caution
#
# Running sentence-transformers locally can be affected by your operating system and other global factors. It is recommended for experienced users only.
#
# :::
#
# ## Setup
#
# You'll need to install the `langchain_huggingface` package as a dependency:

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-huggingface")


# ## Usage

# In[5]:


from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text = "This is a test document."
query_result = embeddings.embed_query(text)

# show only the first 100 characters of the stringified vector
print(str(query_result)[:100] + "...")


# In[6]:


doc_result = embeddings.embed_documents([text, "This is not a test document."])
print(str(doc_result)[:100] + "...")


# ## Troubleshooting
#
# If you are having issues with the `accelerate` package not being found or failing to import, installing/upgrading it may help:

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU accelerate")
