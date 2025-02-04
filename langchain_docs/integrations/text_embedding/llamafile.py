#!/usr/bin/env python
# coding: utf-8

# # llamafile
# 
# Let's load the [llamafile](https://github.com/Mozilla-Ocho/llamafile) Embeddings class.
# 
# ## Setup
# 
# First, the are 3 setup steps:
# 
# 1. Download a llamafile. In this notebook, we use `TinyLlama-1.1B-Chat-v1.0.Q5_K_M` but there are many others available on [HuggingFace](https://huggingface.co/models?other=llamafile).
# 2. Make the llamafile executable.
# 3. Start the llamafile in server mode.
# 
# You can run the following bash script to do all this:

# In[ ]:


get_ipython().run_cell_magic('bash', '', '# llamafile setup\n\n# Step 1: Download a llamafile. The download may take several minutes.\nwget -nv -nc https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile\n\n# Step 2: Make the llamafile executable. Note: if you\'re on Windows, just append \'.exe\' to the filename.\nchmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile\n\n# Step 3: Start llamafile server in background. All the server logs will be written to \'tinyllama.log\'.\n# Alternatively, you can just open a separate terminal outside this notebook and run: \n#   ./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser --embedding\n./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser --embedding > tinyllama.log 2>&1 &\npid=$!\necho "${pid}" > .llamafile_pid  # write the process pid to a file so we can terminate the server later\n')


# ## Embedding texts using LlamafileEmbeddings
# 
# Now, we can use the `LlamafileEmbeddings` class to interact with the llamafile server that's currently serving our TinyLlama model at http://localhost:8080.

# In[ ]:


from langchain_community.embeddings import LlamafileEmbeddings


# In[ ]:


embedder = LlamafileEmbeddings()


# In[ ]:


text = "This is a test document."


# To generate embeddings, you can either query an invidivual text, or you can query a list of texts.

# In[ ]:


query_result = embedder.embed_query(text)
query_result[:5]


# In[ ]:


doc_result = embedder.embed_documents([text])
doc_result[0][:5]


# In[ ]:


get_ipython().run_cell_magic('bash', '', '# cleanup: kill the llamafile server process\nkill $(cat .llamafile_pid)\nrm .llamafile_pid\n')

