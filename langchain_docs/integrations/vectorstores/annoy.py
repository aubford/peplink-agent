#!/usr/bin/env python
# coding: utf-8

# # Annoy
#
# > [Annoy](https://github.com/spotify/annoy) (`Approximate Nearest Neighbors Oh Yeah`) is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mapped into memory so that many processes may share the same data.
#
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
#
# This notebook shows how to use functionality related to the `Annoy` vector database.

# ```{note}
# NOTE: Annoy is read-only - once the index is built you cannot add any more embeddings!
# If you want to progressively add new entries to your VectorStore then better choose an alternative!
# ```

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  annoy")


# ## Create VectorStore from texts

# In[ ]:


from langchain_community.vectorstores import Annoy
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings_func = HuggingFaceEmbeddings(model_name=model_name)


# In[4]:


texts = ["pizza is great", "I love salad", "my car", "a dog"]

# default metric is angular
vector_store = Annoy.from_texts(texts, embeddings_func)


# In[4]:


# allows for custom annoy parameters, defaults are n_trees=100, n_jobs=-1, metric="angular"
vector_store_v2 = Annoy.from_texts(
    texts, embeddings_func, metric="dot", n_trees=100, n_jobs=1
)


# In[5]:


vector_store.similarity_search("food", k=3)


# In[6]:


# the score is a distance metric, so lower is better
vector_store.similarity_search_with_score("food", k=3)


# ## Create VectorStore from docs

# In[7]:


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txtn.txtn.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


# In[8]:


docs[:5]


# In[9]:


vector_store_from_docs = Annoy.from_documents(docs, embeddings_func)


# In[10]:


query = "What did the president say about Ketanji Brown Jackson"
docs = vector_store_from_docs.similarity_search(query)


# In[11]:


print(docs[0].page_content[:100])


# ## Create VectorStore via existing embeddings

# In[12]:


embs = embeddings_func.embed_documents(texts)


# In[13]:


data = list(zip(texts, embs))

vector_store_from_embeddings = Annoy.from_embeddings(data, embeddings_func)


# In[14]:


vector_store_from_embeddings.similarity_search_with_score("food", k=3)


# ## Search via embeddings

# In[15]:


motorbike_emb = embeddings_func.embed_query("motorbike")


# In[16]:


vector_store.similarity_search_by_vector(motorbike_emb, k=3)


# In[17]:


vector_store.similarity_search_with_score_by_vector(motorbike_emb, k=3)


# ## Search via docstore id

# In[18]:


vector_store.index_to_docstore_id


# In[19]:


some_docstore_id = 0  # texts[0]

vector_store.docstore._dict[vector_store.index_to_docstore_id[some_docstore_id]]


# In[20]:


# same document has distance 0
vector_store.similarity_search_with_score_by_index(some_docstore_id, k=3)


# ## Save and load

# In[21]:


vector_store.save_local("my_annoy_index_and_docstore")


# In[22]:


loaded_vector_store = Annoy.load_local(
    "my_annoy_index_and_docstore", embeddings=embeddings_func
)


# In[23]:


# same document has distance 0
loaded_vector_store.similarity_search_with_score_by_index(some_docstore_id, k=3)


# ## Construct from scratch

# In[25]:


import uuid

from annoy import AnnoyIndex
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

metadatas = [{"x": "food"}, {"x": "food"}, {"x": "stuff"}, {"x": "animal"}]

# embeddings
embeddings = embeddings_func.embed_documents(texts)

# embedding dim
f = len(embeddings[0])

# index
metric = "angular"
index = AnnoyIndex(f, metric=metric)
for i, emb in enumerate(embeddings):
    index.add_item(i, emb)
index.build(10)

# docstore
documents = []
for i, text in enumerate(texts):
    metadata = metadatas[i] if metadatas else {}
    documents.append(Document(page_content=text, metadata=metadata))
index_to_docstore_id = {i: str(uuid.uuid4()) for i in range(len(documents))}
docstore = InMemoryDocstore(
    {index_to_docstore_id[i]: doc for i, doc in enumerate(documents)}
)

db_manually = Annoy(
    embeddings_func.embed_query, index, metric, docstore, index_to_docstore_id
)


# In[26]:


db_manually.similarity_search_with_score("eating!", k=3)
