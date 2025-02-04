#!/usr/bin/env python
# coding: utf-8

# # OpenVINO
# [OpenVINO™](https://github.com/openvinotoolkit/openvino) is an open-source toolkit for optimizing and deploying AI inference. The OpenVINO™ Runtime supports various hardware [devices](https://github.com/openvinotoolkit/openvino?tab=readme-ov-file#supported-hardware-matrix) including x86 and ARM CPUs, and Intel GPUs. It can help to boost deep learning performance in Computer Vision, Automatic Speech Recognition, Natural Language Processing and other common tasks.
# 
# Hugging Face embedding model can be supported by OpenVINO through ``OpenVINOEmbeddings`` class. If you have an Intel GPU, you can specify `model_kwargs={"device": "GPU"}` to run inference on it.

# In[3]:


get_ipython().run_line_magic('pip', 'install --upgrade-strategy eager "optimum[openvino,nncf]" --quiet')


# In[1]:


from langchain_community.embeddings import OpenVINOEmbeddings


# In[ ]:


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "CPU"}
encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}

ov_embeddings = OpenVINOEmbeddings(
    model_name_or_path=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


# In[3]:


text = "This is a test document."


# In[4]:


query_result = ov_embeddings.embed_query(text)


# In[5]:


query_result[:3]


# In[6]:


doc_result = ov_embeddings.embed_documents([text])


# ## Export IR model
# It is possible to export your embedding model to the OpenVINO IR format with ``OVModelForFeatureExtraction``, and load the model from local folder.

# In[7]:


from pathlib import Path

ov_model_dir = "all-mpnet-base-v2-ov"
if not Path(ov_model_dir).exists():
    ov_embeddings.save_model(ov_model_dir)


# In[8]:


ov_embeddings = OpenVINOEmbeddings(
    model_name_or_path=ov_model_dir,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


# ## BGE with OpenVINO
# We can also access BGE embedding models via the ``OpenVINOBgeEmbeddings`` class with OpenVINO. 

# In[ ]:


from langchain_community.embeddings import OpenVINOBgeEmbeddings

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "CPU"}
encode_kwargs = {"normalize_embeddings": True}
ov_embeddings = OpenVINOBgeEmbeddings(
    model_name_or_path=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


# In[2]:


embedding = ov_embeddings.embed_query("hi this is harrison")
len(embedding)


# For more information refer to:
# 
# * [OpenVINO LLM guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html).
# 
# * [OpenVINO Documentation](https://docs.openvino.ai/2024/home.html).
# 
# * [OpenVINO Get Started Guide](https://www.intel.com/content/www/us/en/content-details/819067/openvino-get-started-guide.html).
# 
# * [RAG Notebook with LangChain](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-rag-langchain).
