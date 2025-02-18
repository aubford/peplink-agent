#!/usr/bin/env python
# coding: utf-8

# # How to split JSON data
#
# This json splitter [splits](/docs/concepts/text_splitters/) json data while allowing control over chunk sizes. It traverses json data depth first and builds smaller json chunks. It attempts to keep nested json objects whole but will split them if needed to keep chunks between a min_chunk_size and the max_chunk_size.
#
# If the value is not a nested json, but rather a very large string the string will not be split. If you need a hard cap on the chunk size consider composing this with a Recursive Text splitter on those chunks. There is an optional pre-processing step to split lists, by first converting them to json (dict) and then splitting them as such.
#
# 1. How the text is split: json value.
# 2. How the chunk size is measured: by number of characters.

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-text-splitters")


# First we load some json data:

# In[1]:


import json

import requests

# This is a large nested json object and will be loaded as a python dict
json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()


# ## Basic usage
#
# Specify `max_chunk_size` to constrain chunk sizes:

# In[2]:


from langchain_text_splitters import RecursiveJsonSplitter

splitter = RecursiveJsonSplitter(max_chunk_size=300)


# To obtain json chunks, use the `.split_json` method:

# In[3]:


# Recursively split json data - If you need to access/manipulate the smaller json chunks
json_chunks = splitter.split_json(json_data=json_data)

for chunk in json_chunks[:3]:
    print(chunk)


# To obtain LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects, use the `.create_documents` method:

# In[4]:


# The splitter can also output documents
docs = splitter.create_documents(texts=[json_data])

for doc in docs[:3]:
    print(doc)


# Or use `.split_text` to obtain string content directly:

# In[5]:


texts = splitter.split_text(json_data=json_data)

print(texts[0])
print(texts[1])


# ## How to manage chunk sizes from list content
#
# Note that one of the chunks in this example is larger than the specified `max_chunk_size` of 300. Reviewing one of these chunks that was bigger we see there is a list object there:

# In[6]:


print([len(text) for text in texts][:10])
print()
print(texts[3])


# The json splitter by default does not split lists.
#
# Specify `convert_lists=True` to preprocess the json, converting list content to dicts with `index:item` as `key:val` pairs:

# In[7]:


texts = splitter.split_text(json_data=json_data, convert_lists=True)


# Let's look at the size of the chunks. Now they are all under the max

# In[8]:


print([len(text) for text in texts][:10])


# The list has been converted to a dict, but retains all the needed contextual information even if split into many chunks:

# In[9]:


print(texts[1])


# In[10]:


# We can also look at the documents
docs[1]
