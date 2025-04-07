#!/usr/bin/env python
# coding: utf-8

# # How to use the Parent Document Retriever
# 
# When splitting documents for [retrieval](/docs/concepts/retrieval/), there are often conflicting desires:
# 
# 1. You may want to have small documents, so that their embeddings can most
#     accurately reflect their meaning. If too long, then the embeddings can
#     lose meaning.
# 2. You want to have long enough documents that the context of each chunk is
#     retained.
# 
# The `ParentDocumentRetriever` strikes that balance by splitting and storing
# small chunks of data. During retrieval, it first fetches the small chunks
# but then looks up the parent ids for those chunks and returns those larger
# documents.
# 
# Note that "parent document" refers to the document that a small chunk
# originated from. This can either be the whole raw document OR a larger
# chunk.

# In[1]:


from langchain.retrievers import ParentDocumentRetriever


# In[2]:


from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# In[3]:


loaders = [
    TextLoader("paul_graham_essay.txt"),
    TextLoader("state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


# ## Retrieving full documents
# 
# In this mode, we want to retrieve the full documents. Therefore, we only specify a child [splitter](/docs/concepts/text_splitters/).

# In[4]:


# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)


# In[5]:


retriever.add_documents(docs, ids=None)


# This should yield two keys, because we added two documents.

# In[6]:


list(store.yield_keys())


# Let's now call the vector store search functionality - we should see that it returns small chunks (since we're storing the small chunks).

# In[7]:


sub_docs = vectorstore.similarity_search("justice breyer")


# In[8]:


print(sub_docs[0].page_content)


# Let's now retrieve from the overall retriever. This should return large documents - since it returns the documents where the smaller chunks are located.

# In[9]:


retrieved_docs = retriever.invoke("justice breyer")


# In[10]:


len(retrieved_docs[0].page_content)


# ## Retrieving larger chunks
# 
# Sometimes, the full documents can be too big to want to retrieve them as is. In that case, what we really want to do is to first split the raw documents into larger chunks, and then split it into smaller chunks. We then index the smaller chunks, but on retrieval we retrieve the larger chunks (but still not the full documents).

# In[11]:


# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()


# In[12]:


retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)


# In[13]:


retriever.add_documents(docs)


# We can see that there are much more than two documents now - these are the larger chunks.

# In[14]:


len(list(store.yield_keys()))


# Let's make sure the underlying vector store still retrieves the small chunks.

# In[15]:


sub_docs = vectorstore.similarity_search("justice breyer")


# In[16]:


print(sub_docs[0].page_content)


# In[18]:


retrieved_docs = retriever.invoke("justice breyer")


# In[19]:


len(retrieved_docs[0].page_content)


# In[20]:


print(retrieved_docs[0].page_content)

