#!/usr/bin/env python
# coding: utf-8

# # Jina
# 
# You can check the list of available models from [here](https://jina.ai/embeddings/).
# 
# ## Installation and setup

# Install requirements

# In[ ]:


pip install -U langchain-community


# Import libraries

# In[ ]:


import requests
from langchain_community.embeddings import JinaEmbeddings
from numpy import dot
from numpy.linalg import norm
from PIL import Image


# ## Embed text and queries with Jina embedding models through JinaAI API

# In[ ]:


text_embeddings = JinaEmbeddings(
    jina_api_key="jina_*", model_name="jina-embeddings-v2-base-en"
)


# In[ ]:


text = "This is a test document."


# In[ ]:


query_result = text_embeddings.embed_query(text)


# In[ ]:


print(query_result)


# In[ ]:


doc_result = text_embeddings.embed_documents([text])


# In[ ]:


print(doc_result)


# ## Embed images and queries with Jina CLIP through JinaAI API

# In[ ]:


multimodal_embeddings = JinaEmbeddings(jina_api_key="jina_*", model_name="jina-clip-v1")


# In[ ]:


image = "https://avatars.githubusercontent.com/u/126733545?v=4"

description = "Logo of a parrot and a chain on green background"

im = Image.open(requests.get(image, stream=True).raw)
print("Image:")
display(im)


# In[ ]:


image_result = multimodal_embeddings.embed_images([image])


# In[ ]:


print(image_result)


# In[ ]:


description_result = multimodal_embeddings.embed_documents([description])


# In[ ]:


print(description_result)


# In[ ]:


cosine_similarity = dot(image_result[0], description_result[0]) / (
    norm(image_result[0]) * norm(description_result[0])
)


# In[ ]:


print(cosine_similarity)

