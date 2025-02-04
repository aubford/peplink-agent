#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: IBM watsonx.ai
---
# # WatsonxRerank
# 
# >WatsonxRerank is a wrapper for IBM [watsonx.ai](https://www.ibm.com/products/watsonx-ai) foundation models.
# 
# This notebook shows how to use [watsonx's rerank endpoint](https://cloud.ibm.com/apidocs/watsonx-ai#text-rerank) in a retriever. This builds on top of ideas in the [ContextualCompressionRetriever](/docs/how_to/contextual_compression).

# ## Overview
# 
# ### Integration details
# 
# | Class | Package | [JS support](https://js.langchain.com/docs/integrations/document_compressors/ibm/) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: | :---: |
# | [WatsonxRerank](https://python.langchain.com/api_reference/ibm/rerank/langchain_ibm.rerank.WatsonxRerank.html) | [langchain-ibm](https://python.langchain.com/api_reference/ibm/index.html) | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-ibm?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-ibm?style=flat-square&label=%20) |

# ## Setup
# 
# To access IBM watsonx.ai models you'll need to create an IBM watsonx.ai account, get an API key, and install the `langchain-ibm` integration package.
# 
# ### Credentials
# 
# The cell below defines the credentials required to work with watsonx Foundation Model inferencing.
# 
# **Action:** Provide the IBM Cloud user API key. For details, see
# [Managing user API keys](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).

# In[ ]:


import os
from getpass import getpass

watsonx_api_key = getpass()
os.environ["WATSONX_APIKEY"] = watsonx_api_key


# Additionally you are able to pass additional secrets as an environment variable. 

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

# In[1]:


get_ipython().system('pip install -qU langchain-ibm')
get_ipython().system('pip install -qU langchain-community')
get_ipython().system('pip install -qU langchain_text_splitters')


# For experiment purpose please also install `faiss` or `faiss-cpu` package:

# In[ ]:


get_ipython().system('pip install --upgrade --quiet  faiss')

# OR  (depending on Python version)

get_ipython().system('pip install --upgrade --quiet  faiss-cpu')


# Helper function for printing docs

# In[8]:


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# ## Instantiation
# 
# ### Set up the base vector store retriever
# Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.
# 
# Initialize the `WatsonxEmbeddings`. For more details see [WatsonxEmbeddings](/docs/integrations/text_embedding/ibm_watsonx).
# 
# **Note**: 
# 
# - To provide context for the API call, you must add `project_id` or `space_id`. For more information see [documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects).
# - Depending on the region of your provisioned service instance, use one of the urls described [here](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).
# 
# In this example, we’ll use the `project_id` and Dallas url.
# 
# You need to specify `model_id` that will be used for embedding. All available models you can find in [documentation](https://ibm.github.io/watsonx-ai-python-sdk/fm_embeddings.html#EmbeddingModels).

# In[ ]:


from langchain_ibm import WatsonxEmbeddings

wx_embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
)


# In[22]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, wx_embeddings).as_retriever(
    search_kwargs={"k": 20}
)

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs[:5])  # Printing the first 5 documents


# ## Usage
# 
# ### Doing reranking with WatsonxRerank
# Now let's wrap our base retriever with a `ContextualCompressionRetriever`. We'll add an `WatsonxRerank`, uses the watsonx rerank endpoint to rerank the returned results.
# Do note that it is mandatory to specify the model name in WatsonxRerank!

# In[ ]:


from langchain_ibm import WatsonxRerank

wx_rerank = WatsonxRerank(
    model_id="cross-encoder/ms-marco-minilm-l-12-v2",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
)


# In[23]:


from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

compression_retriever = ContextualCompressionRetriever(
    base_compressor=wx_rerank, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs[:5])  # Printing the first 5 compressed documents


# ## Use within a chain
# 
# You can of course use this retriever within a QA pipeline
# 
# Initialize the `ChatWatsonx`. For more details see [ChatWatsonx](/docs/integrations/chat/ibm_watsonx).

# In[ ]:


from langchain_ibm import ChatWatsonx

wx_chat = ChatWatsonx(
    model_id="meta-llama/llama-3-1-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
)


# In[19]:


from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(llm=wx_chat, retriever=compression_retriever)


# In[20]:


chain.invoke(query)


# ## API reference
# 
# For detailed documentation of all `WatsonxRerank` features and configurations head to the [API reference](https://python.langchain.com/api_reference/ibm/chat_models/langchain_ibm.rerank.WatsonxRerank.html).
