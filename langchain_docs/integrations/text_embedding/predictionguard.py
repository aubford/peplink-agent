#!/usr/bin/env python
# coding: utf-8

# # PredictionGuardEmbeddings

# >[Prediction Guard](https://predictionguard.com) is a secure, scalable GenAI platform that safeguards sensitive data, prevents common AI malfunctions, and runs on affordable hardware.

# ## Overview

# ### Integration details
# This integration shows how to use the Prediction Guard embeddings integration with Langchain. This integration supports text and images, separately or together in matched pairs.

# ## Setup
# To access Prediction Guard models, contact us [here](https://predictionguard.com/get-started) to get a Prediction Guard API key and get started.
#

# ### Credentials
# Once you have a key, you can set it with
#

# In[1]:


import os

os.environ["PREDICTIONGUARD_API_KEY"] = "<Prediction Guard API Key"


# ### Installation

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet langchain-predictionguard"
)


# ## Instantiation

# First, install the Prediction Guard and LangChain packages. Then, set the required env vars and set up package imports.

# In[2]:


from langchain_predictionguard import PredictionGuardEmbeddings


# In[3]:


embeddings = PredictionGuardEmbeddings(model="bridgetower-large-itm-mlm-itc")


#

# Prediction Guard embeddings generation supports both text and images. This integration includes that support spread across various functions.

# ## Indexing and Retrieval

# In[5]:


# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore

text = "LangChain is the framework for building context-aware reasoning applications."

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

# Show the retrieved document's content
retrieved_documents[0].page_content


# ## Direct Usage
# The vectorstore and retriever implementations are calling `embeddings.embed_documents(...)` and `embeddings.embed_query(...)` to create embeddings from the texts used in the `from_texts` and retrieval `invoke` operations.
#
# These methods can be directly called with the following commands.

# ### Embed single texts

# In[6]:


# Embedding a single string
text = "This is an embedding example."
single_vector = embeddings.embed_query(text)

single_vector[:5]


# ### Embed multiple texts

# In[7]:


# Embedding multiple strings
docs = [
    "This is an embedding example.",
    "This is another embedding example.",
]

two_vectors = embeddings.embed_documents(docs)

for vector in two_vectors:
    print(vector[:5])


# ### Embed single images

# In[8]:


# Embedding a single image. These functions accept image URLs, image files, data URIs, and base64 encoded strings.
image = [
    "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg",
]
single_vector = embeddings.embed_images(image)

print(single_vector[0][:5])


# ### Embed multiple images

# In[9]:


# Embedding multiple images
images = [
    "https://fastly.picsum.photos/id/866/200/300.jpg?hmac=rcadCENKh4rD6MAp6V_ma-AyWv641M4iiOpe1RyFHeI",
    "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg",
]

two_vectors = embeddings.embed_images(images)

for vector in two_vectors:
    print(vector[:5])


# ### Embed single text-image pairs

# In[10]:


# Embedding a single text-image pair
inputs = [
    {
        "text": "This is an embedding example.",
        "image": "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg",
    },
]
single_vector = embeddings.embed_image_text(inputs)

print(single_vector[0][:5])


# ### Embed multiple text-image pairs

# In[11]:


# Embedding multiple text-image pairs
inputs = [
    {
        "text": "This is an embedding example.",
        "image": "https://fastly.picsum.photos/id/866/200/300.jpg?hmac=rcadCENKh4rD6MAp6V_ma-AyWv641M4iiOpe1RyFHeI",
    },
    {
        "text": "This is another embedding example.",
        "image": "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg",
    },
]
two_vectors = embeddings.embed_image_text(inputs)

for vector in two_vectors:
    print(vector[:5])


# ## API Reference
# For detailed documentation of all PredictionGuardEmbeddings features and configurations check out the API reference: https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.predictionguard.PredictionGuardEmbeddings.html
