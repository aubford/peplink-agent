#!/usr/bin/env python
# coding: utf-8

# # Marqo
# 
# This notebook shows how to use functionality related to the Marqo vectorstore.
# 
# >[Marqo](https://www.marqo.ai/) is an open-source vector search engine. Marqo allows you to store and query multi-modal data such as text and images. Marqo creates the vectors for you using a huge selection of open-source models, you can also provide your own fine-tuned models and Marqo will handle the loading and inference for you.
# 
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
# 
# To run this notebook with our docker image please run the following commands first to get Marqo:
# 
# ```
# docker pull marqoai/marqo:latest
# docker rm -f marqo
# docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
# ```

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  marqo')


# In[1]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Marqo
from langchain_text_splitters import CharacterTextSplitter


# In[2]:


from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


# In[3]:


import marqo

# initialize marqo
marqo_url = "http://localhost:8882"  # if using marqo cloud replace with your endpoint (console.marqo.ai)
marqo_api_key = ""  # if using marqo cloud replace with your api key (console.marqo.ai)

client = marqo.Client(url=marqo_url, api_key=marqo_api_key)

index_name = "langchain-demo"

docsearch = Marqo.from_documents(docs, index_name=index_name)

query = "What did the president say about Ketanji Brown Jackson"
result_docs = docsearch.similarity_search(query)


# In[4]:


print(result_docs[0].page_content)


# In[5]:


result_docs = docsearch.similarity_search_with_score(query)
print(result_docs[0][0].page_content, result_docs[0][1], sep="\n")


# ## Additional features
# 
# One of the powerful features of Marqo as a vectorstore is that you can use indexes created externally. For example:
# 
# + If you had a database of image and text pairs from another application, you can simply just use it in langchain with the Marqo vectorstore. Note that bringing your own multimodal indexes will disable the `add_texts` method.
# 
# + If you had a database of text documents, you can bring it into the langchain framework and add more texts through `add_texts`.
# 
# The documents that are returned are customised by passing your own function to the `page_content_builder` callback in the search methods.

# #### Multimodal Example

# In[6]:


# use a new index
index_name = "langchain-multimodal-demo"

# incase the demo is re-run
try:
    client.delete_index(index_name)
except Exception:
    print(f"Creating {index_name}")

# This index could have been created by another system
settings = {"treat_urls_and_pointers_as_images": True, "model": "ViT-L/14"}
client.create_index(index_name, **settings)
client.index(index_name).add_documents(
    [
        # image of a bus
        {
            "caption": "Bus",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
        },
        # image of a plane
        {
            "caption": "Plane",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
        },
    ],
)


# In[7]:


def get_content(res):
    """Helper to format Marqo's documents into text to be used as page_content"""
    return f"{res['caption']}: {res['image']}"


docsearch = Marqo(client, index_name, page_content_builder=get_content)


query = "vehicles that fly"
doc_results = docsearch.similarity_search(query)


# In[8]:


for doc in doc_results:
    print(doc.page_content)


# #### Text only example

# In[9]:


# use a new index
index_name = "langchain-byo-index-demo"

# incase the demo is re-run
try:
    client.delete_index(index_name)
except Exception:
    print(f"Creating {index_name}")

# This index could have been created by another system
client.create_index(index_name)
client.index(index_name).add_documents(
    [
        {
            "Title": "Smartphone",
            "Description": "A smartphone is a portable computer device that combines mobile telephone "
            "functions and computing functions into one unit.",
        },
        {
            "Title": "Telephone",
            "Description": "A telephone is a telecommunications device that permits two or more users to"
            "conduct a conversation when they are too far apart to be easily heard directly.",
        },
    ],
)


# In[10]:


# Note text indexes retain the ability to use add_texts despite different field names in documents
# this is because the page_content_builder callback lets you handle these document fields as required


def get_content(res):
    """Helper to format Marqo's documents into text to be used as page_content"""
    if "text" in res:
        return res["text"]
    return res["Description"]


docsearch = Marqo(client, index_name, page_content_builder=get_content)

docsearch.add_texts(["This is a document that is about elephants"])


# In[11]:


query = "modern communications devices"
doc_results = docsearch.similarity_search(query)

print(doc_results[0].page_content)


# In[12]:


query = "elephants"
doc_results = docsearch.similarity_search(query, page_content_builder=get_content)

print(doc_results[0].page_content)


# ## Weighted Queries
# 
# We also expose marqos weighted queries which are a powerful way to compose complex semantic searches.

# In[13]:


query = {"communications devices": 1.0}
doc_results = docsearch.similarity_search(query)
print(doc_results[0].page_content)


# In[14]:


query = {"communications devices": 1.0, "technology post 2000": -1.0}
doc_results = docsearch.similarity_search(query)
print(doc_results[0].page_content)


# # Question Answering with Sources
# 
# This section shows how to use Marqo as part of a `RetrievalQAWithSourcesChain`. Marqo will perform the searches for information in the sources.

# In[15]:


import getpass
import os

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[16]:


with open("../../how_to/state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)


# In[17]:


index_name = "langchain-qa-with-retrieval"
docsearch = Marqo.from_documents(docs, index_name=index_name)


# In[18]:


chain = RetrievalQAWithSourcesChain.from_chain_type(
    OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever()
)


# In[19]:


chain(
    {"question": "What did the president say about Justice Breyer"},
    return_only_outputs=True,
)

