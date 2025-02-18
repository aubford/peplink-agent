#!/usr/bin/env python
# coding: utf-8

# # HuggingFace dataset
#
# >The [Hugging Face Hub](https://huggingface.co/docs/hub/index) is home to over 5,000 [datasets](https://huggingface.co/docs/hub/index#datasets) in more than 100 languages that can be used for a broad range of tasks across NLP, Computer Vision, and Audio. They used for a diverse range of tasks such as translation,
# automatic speech recognition, and image classification.
#
#
# This notebook shows how to load `Hugging Face Hub` datasets to LangChain.

# In[1]:


from langchain_community.document_loaders import HuggingFaceDatasetLoader


# In[11]:


dataset_name = "imdb"
page_content_column = "text"


loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)


# In[ ]:


data = loader.load()


# In[14]:


data[:15]


# ### Example
# In this example, we use data from a dataset to answer a question

# In[8]:


from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders.hugging_face_dataset import (
    HuggingFaceDatasetLoader,
)


# In[24]:


dataset_name = "tweet_eval"
page_content_column = "text"
name = "stance_climate"


loader = HuggingFaceDatasetLoader(dataset_name, page_content_column, name)


# In[26]:


index = VectorstoreIndexCreator().from_loaders([loader])


# In[29]:


query = "What are the most used hashtag?"
result = index.query(query)


# In[30]:


result


# In[ ]:
