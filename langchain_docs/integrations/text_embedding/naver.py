#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Naver
---
# # ClovaXEmbeddings
# 
# This notebook covers how to get started with embedding models provided by CLOVA Studio. For detailed documentation on `ClovaXEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.naver.ClovaXEmbeddings.html).
# 
# ## Overview
# ### Integration details
# 
# | Provider | Package |
# |:--------:|:-------:|
# | [Naver](/docs/integrations/providers/naver.mdx) | [langchain-community](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.naver.ClovaXEmbeddings.html) |
# 
# ## Setup
# 
# Before using embedding models provided by CLOVA Studio, you must go through the three steps below.
# 
# 1. Creating [NAVER Cloud Platform](https://www.ncloud.com/) account 
# 2. Apply to use [CLOVA Studio](https://www.ncloud.com/product/aiService/clovaStudio)
# 3. Create a CLOVA Studio Test App or Service App of a model to use (See [here](https://guide.ncloud-docs.com/docs/clovastudio-explorer03#%ED%85%8C%EC%8A%A4%ED%8A%B8%EC%95%B1%EC%83%9D%EC%84%B1).)
# 4. Issue a Test or Service API key (See [here](https://api.ncloud-docs.com/docs/ai-naver-clovastudio-summary#API%ED%82%A4).)
# 
# ### Credentials
# 
# Set the `NCP_CLOVASTUDIO_API_KEY` environment variable with your API key.
#   - Note that if you are using a legacy API Key (that doesn't start with `nv-*` prefix), you might need two additional keys to be set as environment variables (`NCP_APIGW_API_KEY` and `NCP_CLOVASTUDIO_APP_ID`. They could be found by clicking `App Request Status` > `Service App, Test App List` > `Details` button for each app in [CLOVA Studio](https://clovastudio.ncloud.com/studio-application/service-app).

# In[ ]:


import getpass
import os

if not os.getenv("NCP_CLOVASTUDIO_API_KEY"):
    os.environ["NCP_CLOVASTUDIO_API_KEY"] = getpass.getpass(
        "Enter NCP CLOVA Studio API Key: "
    )


# Uncomment below to use a legacy API key:

# In[ ]:


# if not os.getenv("NCP_APIGW_API_KEY"):
#     os.environ["NCP_APIGW_API_KEY"] = getpass.getpass("Enter NCP API Gateway API Key: ")
# os.environ["NCP_CLOVASTUDIO_APP_ID"] = input("Enter NCP CLOVA Studio App ID: ")


# ### Installation
# 
# ClovaXEmbeddings integration lives in the `langchain_community` package:

# In[ ]:


# install package
get_ipython().system('pip install -U langchain-community')


# ## Instantiation
# 
# Now we can instantiate our embeddings object and embed query or document:
# 
# - There are several embedding models available in CLOVA Studio. Please refer [here](https://guide.ncloud-docs.com/docs/en/clovastudio-explorer03#임베딩API) for further details.
# - Note that you might need to normalize the embeddings depending on your specific use case.

# In[7]:


from langchain_community.embeddings import ClovaXEmbeddings

embeddings = ClovaXEmbeddings(
    model="clir-emb-dolphin"  # set with the model name of corresponding app id. Default is `clir-emb-dolphin`
)


# ## Indexing and Retrieval
# 
# Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/).
# 
# Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.

# In[8]:


# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore

text = "CLOVA Studio is an AI development tool that allows you to customize your own HyperCLOVA X models."

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is CLOVA Studio?")

# show the retrieved document's content
retrieved_documents[0].page_content


# ## Direct Usage
# 
# Under the hood, the vectorstore and retriever implementations are calling `embeddings.embed_documents(...)` and `embeddings.embed_query(...)` to create embeddings for the text(s) used in `from_texts` and retrieval `invoke` operations, respectively.
# 
# You can directly call these methods to get embeddings for your own use cases.
# 
# ### Embed single texts
# 
# You can embed single texts or documents with `embed_query`:

# In[9]:


single_vector = embeddings.embed_query(text)
print(str(single_vector)[:100])  # Show the first 100 characters of the vector


# ### Embed multiple texts
# 
# You can embed multiple texts with `embed_documents`:

# In[10]:


text2 = "LangChain is the framework for building context-aware reasoning applications"
two_vectors = embeddings.embed_documents([text, text2])
for vector in two_vectors:
    print(str(vector)[:100])  # Show the first 100 characters of the vector


# ## Additional functionalities
# 
# ### Service App
# 
# When going live with production-level application using CLOVA Studio, you should apply for and use Service App. (See [here](https://guide.ncloud-docs.com/docs/en/clovastudio-playground01#서비스앱신청).)
# 
# For a Service App, you should use a corresponding Service API key and can only be called with it.

# In[ ]:


# Update environment variables

os.environ["NCP_CLOVASTUDIO_API_KEY"] = getpass.getpass(
    "Enter NCP CLOVA Studio API Key for Service App: "
)
# Uncomment below to use a legacy API key:
os.environ["NCP_CLOVASTUDIO_APP_ID"] = input("Enter NCP CLOVA Studio Service App ID: ")


# In[ ]:


embeddings = ClovaXEmbeddings(
    service_app=True,
    model="clir-emb-dolphin",  # set with the model name of corresponding app id of your Service App
)


# ## API Reference
# 
# For detailed documentation on `ClovaXEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/latest/api_reference/community/embeddings/langchain_community.embeddings.naver.ClovaXEmbeddings.html).
