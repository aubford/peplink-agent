#!/usr/bin/env python
# coding: utf-8

# # Llama.cpp
# 
# >[llama.cpp python](https://github.com/abetlen/llama-cpp-python) library is a simple Python bindings for `@ggerganov`
# >[llama.cpp](https://github.com/ggerganov/llama.cpp).
# >
# >This package provides:
# >
# > - Low-level access to C API via ctypes interface.
# > - High-level Python API for text completion
# >   - `OpenAI`-like API
# >   - `LangChain` compatibility
# >   - `LlamaIndex` compatibility
# > - OpenAI compatible web server
# >   - Local Copilot replacement
# >   - Function Calling support
# >   - Vision API support
# >   - Multiple Models
# 

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  llama-cpp-python')


# In[ ]:


from langchain_community.embeddings import LlamaCppEmbeddings


# In[ ]:


llama = LlamaCppEmbeddings(model_path="/path/to/model/ggml-model-q4_0.bin")


# In[ ]:


text = "This is a test document."


# In[ ]:


query_result = llama.embed_query(text)


# In[ ]:


doc_result = llama.embed_documents([text])

