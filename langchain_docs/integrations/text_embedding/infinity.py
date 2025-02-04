#!/usr/bin/env python
# coding: utf-8

# # Infinity
# 
# `Infinity` allows to create `Embeddings` using a MIT-licensed Embedding Server. 
# 
# This notebook goes over how to use Langchain with Embeddings with the [Infinity Github Project](https://github.com/michaelfeil/infinity).
# 

# ## Imports

# In[1]:


from langchain_community.embeddings import InfinityEmbeddings, InfinityEmbeddingsLocal


# # Option 1: Use infinity from Python

# #### Optional: install infinity
# 
# To install infinity use the following command. For further details check out the [Docs on Github](https://github.com/michaelfeil/infinity).
# Install the torch and onnx dependencies. 
# 
# ```bash
# pip install infinity_emb[torch,optimum]
# ```

# In[2]:


documents = [
    "Baguette is a dish.",
    "Paris is the capital of France.",
    "numpy is a lib for linear algebra",
    "You escaped what I've escaped - You'd be in Paris getting fucked up too",
]
query = "Where is Paris?"


# In[3]:


embeddings = InfinityEmbeddingsLocal(
    model="sentence-transformers/all-MiniLM-L6-v2",
    # revision
    revision=None,
    # best to keep at 32
    batch_size=32,
    # for AMD/Nvidia GPUs via torch
    device="cuda",
    # warm up model before execution
)


async def embed():
    # TODO: This function is just to showcase that your call can run async.

    # important: use engine inside of `async with` statement to start/stop the batching engine.
    async with embeddings:
        # avoid closing and starting the engine often.
        # rather keep it running.
        # you may call `await embeddings.__aenter__()` and `__aexit__()
        # if you are sure when to manually start/stop execution` in a more granular way

        documents_embedded = await embeddings.aembed_documents(documents)
        query_result = await embeddings.aembed_query(query)
        print("embeddings created successful")
    return documents_embedded, query_result


# In[ ]:


# run the async code however you would like
# if you are in a jupyter notebook, you can use the following
documents_embedded, query_result = await embed()


# In[ ]:


# (demo) compute similarity
import numpy as np

scores = np.array(documents_embedded) @ np.array(query_result).T
dict(zip(documents, scores))


# # Option 2: Run the server, and connect via the API

# #### Optional: Make sure to start the Infinity instance
# 
# To install infinity use the following command. For further details check out the [Docs on Github](https://github.com/michaelfeil/infinity).
# ```bash
# pip install infinity_emb[all]
# ```

# # Install the infinity package
# %pip install --upgrade --quiet  infinity_emb[all]

# Start up the server - best to be done from a separate terminal, not inside Jupyter Notebook
# 
# ```bash
# model=sentence-transformers/all-MiniLM-L6-v2
# port=7797
# infinity_emb --port $port --model-name-or-path $model
# ```
# 
# or alternativley just use docker:
# ```bash
# model=sentence-transformers/all-MiniLM-L6-v2
# port=7797
# docker run -it --gpus all -p $port:$port michaelf34/infinity:latest --model-name-or-path $model --port $port
# ```

# ## Embed your documents using your Infinity instance 

# In[5]:


documents = [
    "Baguette is a dish.",
    "Paris is the capital of France.",
    "numpy is a lib for linear algebra",
    "You escaped what I've escaped - You'd be in Paris getting fucked up too",
]
query = "Where is Paris?"


# In[6]:


#
infinity_api_url = "http://localhost:7797/v1"
# model is currently not validated.
embeddings = InfinityEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2", infinity_api_url=infinity_api_url
)
try:
    documents_embedded = embeddings.embed_documents(documents)
    query_result = embeddings.embed_query(query)
    print("embeddings created successful")
except Exception as ex:
    print(
        "Make sure the infinity instance is running. Verify by clicking on "
        f"{infinity_api_url.replace('v1','docs')} Exception: {ex}. "
    )


# In[ ]:


# (demo) compute similarity
import numpy as np

scores = np.array(documents_embedded) @ np.array(query_result).T
dict(zip(documents, scores))

