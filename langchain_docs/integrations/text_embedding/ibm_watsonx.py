#!/usr/bin/env python
# coding: utf-8

# # IBM watsonx.ai
#
# >WatsonxEmbeddings is a wrapper for IBM [watsonx.ai](https://www.ibm.com/products/watsonx-ai) foundation models.
#
# This example shows how to communicate with `watsonx.ai` models using `LangChain`.

# ## Overview
# ### Integration details
#
# import { ItemTable } from "@theme/FeatureTables";
#
# <ItemTable category="text_embedding" item="IBM" />

# ## Setup
#
# To access IBM watsonx.ai models you'll need to create an IBM watsonx.ai account, get an API key, and install the `langchain-ibm` integration package.
#
# ### Credentials
#
# This cell defines the WML credentials required to work with watsonx Embeddings.
#
# **Action:** Provide the IBM Cloud user API key. For details, see
# [documentation](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).

# In[1]:


import os
from getpass import getpass

watsonx_api_key = getpass()
os.environ["WATSONX_APIKEY"] = watsonx_api_key


# Additionaly you are able to pass additional secrets as an environment variable.

# In[ ]:


import os

os.environ["WATSONX_URL"] = "your service instance url"
os.environ["WATSONX_TOKEN"] = "your token for accessing the CPD cluster"
os.environ["WATSONX_PASSWORD"] = "your password for accessing the CPD cluster"
os.environ["WATSONX_USERNAME"] = "your username for accessing the CPD cluster"
os.environ["WATSONX_INSTANCE_ID"] = "your instance_id for accessing the CPD cluster"


# ### Installation
#
# The LangChain IBM integration lives in the `langchain-ibm` package:

# In[ ]:


get_ipython().system("pip install -qU langchain-ibm")


# ## Instantiation
#
# You might need to adjust model `parameters` for different models.

# In[1]:


from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}


# Initialize the `WatsonxEmbeddings` class with previously set parameters.
#
#
# **Note**:
#
# - To provide context for the API call, you must add `project_id` or `space_id`. For more information see [documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects).
# - Depending on the region of your provisioned service instance, use one of the urls described [here](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).
#
# In this example, we’ll use the `project_id` and Dallas url.
#
#
# You need to specify `model_id` that will be used for inferencing.

# In[3]:


from langchain_ibm import WatsonxEmbeddings

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=embed_params,
)


# Alternatively you can use Cloud Pak for Data credentials. For details, see [documentation](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).

# In[ ]:


watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="PASTE YOUR URL HERE",
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="4.8",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=embed_params,
)


# For certain requirements, there is an option to pass the IBM's [`APIClient`](https://ibm.github.io/watsonx-ai-python-sdk/base.html#apiclient) object into the `WatsonxEmbeddings` class.

# In[ ]:


from ibm_watsonx_ai import APIClient

api_client = APIClient(...)

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    watsonx_client=api_client,
)


# ## Indexing and Retrieval
#
# Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](/docs/tutorials/).
#
# Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.

# In[3]:


# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore

text = "LangChain is the framework for building context-aware reasoning applications"

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=watsonx_embedding,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

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

# In[4]:


text = "This is a test document."

query_result = watsonx_embedding.embed_query(text)
query_result[:5]


# ### Embed multiple texts
#
# You can embed multiple texts with `embed_documents`:

# In[5]:


texts = ["This is a content of the document", "This is another document"]

doc_result = watsonx_embedding.embed_documents(texts)
doc_result[0][:5]


# ## API Reference
#
# For detailed documentation of all `WatsonxEmbeddings` features and configurations head to the [API reference](https://python.langchain.com/api_reference/ibm/embeddings/langchain_ibm.embeddings.WatsonxEmbeddings.html).
