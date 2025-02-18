#!/usr/bin/env python
# coding: utf-8

# # Hippo
#
# >[Transwarp Hippo](https://www.transwarp.cn/en/subproduct/hippo) is an enterprise-level cloud-native distributed vector database that supports storage, retrieval, and management of massive vector-based datasets. It efficiently solves problems such as vector similarity search and high-density vector clustering. `Hippo` features high availability, high performance, and easy scalability. It has many functions, such as multiple vector search indexes, data partitioning and sharding, data persistence, incremental data ingestion, vector scalar field filtering, and mixed queries. It can effectively meet the high real-time search demands of enterprises for massive vector data
#
# ## Getting Started
#
# The only prerequisite here is an API key from the OpenAI website. Make sure you have already started a Hippo instance.

# ## Installing Dependencies
#
# Initially, we require the installation of certain dependencies, such as OpenAI, Langchain, and Hippo-API. Please note, that you should install the appropriate versions tailored to your environment.

# In[15]:


get_ipython().run_line_magic(
    "pip",
    "install --upgrade --quiet  langchain langchain_community tiktoken langchain-openai",
)
get_ipython().run_line_magic("pip", "install --upgrade --quiet  hippo-api==1.1.0.rc3")


# Note: Python version needs to be >=3.8.
#
# ## Best Practices
# ### Importing Dependency Packages

# In[16]:


import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.hippo import Hippo
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# ### Loading Knowledge Documents

# In[17]:


os.environ["OPENAI_API_KEY"] = "YOUR OPENAI KEY"
loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()


# ### Segmenting the Knowledge Document
#
# Here, we use Langchain's CharacterTextSplitter for segmentation. The delimiter is a period. After segmentation, the text segment does not exceed 1000 characters, and the number of repeated characters is 0.

# In[18]:


text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


# ### Declaring the Embedding Model
# Below, we create the OpenAI or Azure embedding model using the OpenAIEmbeddings method from Langchain.

# In[19]:


# openai
embeddings = OpenAIEmbeddings()
# azure
# embeddings = OpenAIEmbeddings(
#     openai_api_type="azure",
#     openai_api_base="x x x",
#     openai_api_version="x x x",
#     model="x x x",
#     deployment="x x x",
#     openai_api_key="x x x"
# )


# ### Declaring Hippo Client

# In[20]:


HIPPO_CONNECTION = {"host": "IP", "port": "PORT"}


# ### Storing the Document

# In[23]:


print("input...")
# insert docs
vector_store = Hippo.from_documents(
    docs,
    embedding=embeddings,
    table_name="langchain_test",
    connection_args=HIPPO_CONNECTION,
)
print("success")


# ### Conducting Knowledge-based Question and Answer
# #### Creating a Large Language Question-Answering Model
# Below, we create the OpenAI or Azure large language question-answering model respectively using the AzureChatOpenAI and ChatOpenAI methods from Langchain.

# In[24]:


# llm = AzureChatOpenAI(
#     openai_api_base="x x x",
#     openai_api_version="xxx",
#     deployment_name="xxx",
#     openai_api_key="xxx",
#     openai_api_type="azure"
# )

llm = ChatOpenAI(openai_api_key="YOUR OPENAI KEY", model_name="gpt-3.5-turbo-16k")


# ### Acquiring Related Knowledge Based on the Question：

# In[25]:


query = "Please introduce COVID-19"
# query = "Please introduce Hippo Core Architecture"
# query = "What operations does the Hippo Vector Database support for vector data?"
# query = "Does Hippo use hardware acceleration technology? Briefly introduce hardware acceleration technology."


# Retrieve similar content from the knowledge base,fetch the top two most similar texts.
res = vector_store.similarity_search(query, 2)
content_list = [item.page_content for item in res]
text = "".join(content_list)


# ### Constructing a Prompt Template

# In[26]:


prompt = f"""
Please use the content of the following [Article] to answer my question. If you don't know, please say you don't know, and the answer should be concise."
[Article]:{text}
Please answer this question in conjunction with the above article:{query}
"""


# ### Waiting for the Large Language Model to Generate an Answer

# In[27]:


response_with_hippo = llm.predict(prompt)
print(f"response_with_hippo:{response_with_hippo}")
response = llm.predict(query)
print("==========================================")
print(f"response_without_hippo:{response}")


# In[ ]:
