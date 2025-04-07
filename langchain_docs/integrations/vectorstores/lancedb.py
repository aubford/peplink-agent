#!/usr/bin/env python
# coding: utf-8

# # LanceDB
# 
# >[LanceDB](https://lancedb.com/) is an open-source database for vector-search built with persistent storage, which greatly simplifies retrevial, filtering and management of embeddings. Fully open source.
# 
# This notebook shows how to use functionality related to the `LanceDB` vector database based on the Lance data format.

# In[ ]:


get_ipython().system(' pip install tantivy')


# In[ ]:


get_ipython().system(' pip install -U langchain-openai langchain-community')


# In[ ]:


get_ipython().system(' pip install lancedb')


# We want to use OpenAIEmbeddings so we have to get the OpenAI API Key. 

# In[1]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[2]:


get_ipython().system(' rm -rf /tmp/lancedb')


# In[3]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

documents = CharacterTextSplitter().split_documents(documents)
embeddings = OpenAIEmbeddings()


# ##### For LanceDB cloud, you can invoke the vector store as follows :
# 
# 
# ```python
# db_url = "db://lang_test" # url of db you created
# api_key = "xxxxx" # your API key
# region="us-east-1-dev"  # your selected region
# 
# vector_store = LanceDB(
#     uri=db_url,
#     api_key=api_key,
#     region=region,
#     embedding=embeddings,
#     table_name='langchain_test'
#     )
# ```
# 
# You can also add `region`, `api_key`, `uri` to `from_documents()` classmethod
# 

# In[4]:


from lancedb.rerankers import LinearCombinationReranker

reranker = LinearCombinationReranker(weight=0.3)

docsearch = LanceDB.from_documents(documents, embeddings, reranker=reranker)
query = "What did the president say about Ketanji Brown Jackson"


# In[31]:


docs = docsearch.similarity_search_with_relevance_scores(query)
print("relevance score - ", docs[0][1])
print("text- ", docs[0][0].page_content[:1000])


# In[33]:


docs = docsearch.similarity_search_with_score(query="Headaches", query_type="hybrid")
print("distance - ", docs[0][1])
print("text- ", docs[0][0].page_content[:1000])


# In[8]:


print("reranker : ", docsearch._reranker)


# Additionaly, to explore the table you can load it into a df or save it in a csv file: 
# ```python
# tbl = docsearch.get_table()
# print("tbl:", tbl)
# pd_df = tbl.to_pandas()
# # pd_df.to_csv("docsearch.csv", index=False)
# 
# # you can also create a new vector store object using an older connection object:
# vector_store = LanceDB(connection=tbl, embedding=embeddings)
# ```

# In[15]:


docs = docsearch.similarity_search(
    query=query, filter={"metadata.source": "../../how_to/state_of_the_union.txt"}
)

print("metadata :", docs[0].metadata)

# or you can directly supply SQL string filters :

print("\nSQL filtering :\n")
docs = docsearch.similarity_search(query=query, filter="text LIKE '%Officer Rivera%'")
print(docs[0].page_content)


# ## Adding images 

# In[ ]:


get_ipython().system(' pip install -U langchain-experimental')


# In[ ]:


get_ipython().system(' pip install open_clip_torch torch')


# In[16]:


get_ipython().system(" rm -rf '/tmp/multimmodal_lance'")


# In[17]:


from langchain_experimental.open_clip import OpenCLIPEmbeddings


# In[18]:


import os

import requests

# List of image URLs to download
image_urls = [
    "https://github.com/raghavdixit99/assets/assets/34462078/abf47cc4-d979-4aaa-83be-53a2115bf318",
    "https://github.com/raghavdixit99/assets/assets/34462078/93be928e-522b-4e37-889d-d4efd54b2112",
]

texts = ["bird", "dragon"]

# Directory to save images
dir_name = "./photos/"

# Create directory if it doesn't exist
os.makedirs(dir_name, exist_ok=True)

image_uris = []
# Download and save each image
for i, url in enumerate(image_urls, start=1):
    response = requests.get(url)
    path = os.path.join(dir_name, f"image{i}.jpg")
    image_uris.append(path)
    with open(path, "wb") as f:
        f.write(response.content)


# In[21]:


from langchain_community.vectorstores import LanceDB

vec_store = LanceDB(
    table_name="multimodal_test",
    embedding=OpenCLIPEmbeddings(),
)


# In[22]:


vec_store.add_images(uris=image_uris)


# In[23]:


vec_store.add_texts(texts)


# In[24]:


img_embed = vec_store._embedding.embed_query("bird")


# In[25]:


vec_store.similarity_search_by_vector(img_embed)[0]


# In[26]:


vec_store._table

