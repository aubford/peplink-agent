#!/usr/bin/env python
# coding: utf-8
---
keywords: [charactertextsplitter]
---
# # How to split by character
# 
# This is the simplest method. This [splits](/docs/concepts/text_splitters/) based on a given character sequence, which defaults to `"\n\n"`. Chunk length is measured by number of characters.
# 
# 1. How the text is split: by single character separator.
# 2. How the chunk size is measured: by number of characters.
# 
# To obtain the string content directly, use `.split_text`.
# 
# To create LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects (e.g., for use in downstream tasks), use `.create_documents`.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-text-splitters')


# In[1]:


from langchain_text_splitters import CharacterTextSplitter

# Load an example document
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])


# Use `.create_documents` to propagate metadata associated with each document to the output chunks:

# In[2]:


metadatas = [{"document": 1}, {"document": 2}]
documents = text_splitter.create_documents(
    [state_of_the_union, state_of_the_union], metadatas=metadatas
)
print(documents[0])


# Use `.split_text` to obtain the string content directly:

# In[7]:


text_splitter.split_text(state_of_the_union)[0]


# In[ ]:




