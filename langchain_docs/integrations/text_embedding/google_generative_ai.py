#!/usr/bin/env python
# coding: utf-8

# # Google Generative AI Embeddings
# 
# Connect to Google's generative AI embeddings service using the `GoogleGenerativeAIEmbeddings` class, found in the [langchain-google-genai](https://pypi.org/project/langchain-google-genai/) package.

# ## Installation

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-google-genai')


# ## Credentials

# In[ ]:


import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Provide your Google API key here")


# ## Usage

# In[6]:


from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector = embeddings.embed_query("hello, world!")
vector[:5]


# ## Batch
# 
# You can also embed multiple strings at once for a processing speedup:

# In[7]:


vectors = embeddings.embed_documents(
    [
        "Today is Monday",
        "Today is Tuesday",
        "Today is April Fools day",
    ]
)
len(vectors), len(vectors[0])


# ## Task type
# `GoogleGenerativeAIEmbeddings` optionally support a `task_type`, which currently must be one of:
# 
# - task_type_unspecified
# - retrieval_query
# - retrieval_document
# - semantic_similarity
# - classification
# - clustering
# 
# By default, we use `retrieval_document` in the `embed_documents` method and `retrieval_query` in the `embed_query` method. If you provide a task type, we will use that for all methods.

# In[15]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  matplotlib scikit-learn')


# In[33]:


query_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", task_type="retrieval_query"
)
doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", task_type="retrieval_document"
)


# All of these will be embedded with the 'retrieval_query' task set
# ```python
# query_vecs = [query_embeddings.embed_query(q) for q in [query, query_2, answer_1]]
# ```
# All of these will be embedded with the 'retrieval_document' task set
# ```python
# doc_vecs = [doc_embeddings.embed_query(q) for q in [query, query_2, answer_1]]
# ```

# In retrieval, relative distance matters. In the image above, you can see the difference in similarity scores between the "relevant doc" and "simil stronger delta between the similar query and relevant doc on the latter case.

# ## Additional Configuration
# 
# You can pass the following parameters to ChatGoogleGenerativeAI in order to customize the SDK's behavior:
# 
# - `client_options`: [Client Options](https://googleapis.dev/python/google-api-core/latest/client_options.html#module-google.api_core.client_options) to pass to the Google API Client, such as a custom `client_options["api_endpoint"]`
# - `transport`: The transport method to use, such as `rest`, `grpc`, or `grpc_asyncio`.
