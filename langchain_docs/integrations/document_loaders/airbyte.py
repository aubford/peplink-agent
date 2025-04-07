#!/usr/bin/env python
# coding: utf-8

# # AirbyteLoader

# >[Airbyte](https://github.com/airbytehq/airbyte) is a data integration platform for ELT pipelines from APIs, databases & files to warehouses & lakes. It has the largest catalog of ELT connectors to data warehouses and databases.
# 
# This covers how to load any source from Airbyte into LangChain documents
# 
# ## Installation
# 
# In order to use `AirbyteLoader` you need to install the `langchain-airbyte` integration package.

# In[1]:


get_ipython().run_line_magic('pip', 'install -qU langchain-airbyte')


# Note: Currently, the `airbyte` library does not support Pydantic v2.
# Please downgrade to Pydantic v1 to use this package.
# 
# Note: This package also currently requires Python 3.10+.
# 
# ## Loading Documents
# 
# By default, the `AirbyteLoader` will load any structured data from a stream and output yaml-formatted documents.

# In[6]:


from langchain_airbyte import AirbyteLoader

loader = AirbyteLoader(
    source="source-faker",
    stream="users",
    config={"count": 10},
)
docs = loader.load()
print(docs[0].page_content[:500])


# You can also specify a custom prompt template for formatting documents:

# In[7]:


from langchain_core.prompts import PromptTemplate

loader_templated = AirbyteLoader(
    source="source-faker",
    stream="users",
    config={"count": 10},
    template=PromptTemplate.from_template(
        "My name is {name} and I am {height} meters tall."
    ),
)
docs_templated = loader_templated.load()
print(docs_templated[0].page_content)


# ## Lazy Loading Documents
# 
# One of the powerful features of `AirbyteLoader` is its ability to load large documents from upstream sources. When working with large datasets, the default `.load()` behavior can be slow and memory-intensive. To avoid this, you can use the `.lazy_load()` method to load documents in a more memory-efficient manner.

# In[11]:


import time

loader = AirbyteLoader(
    source="source-faker",
    stream="users",
    config={"count": 3},
    template=PromptTemplate.from_template(
        "My name is {name} and I am {height} meters tall."
    ),
)

start_time = time.time()
my_iterator = loader.lazy_load()
print(
    f"Just calling lazy load is quick! This took {time.time() - start_time:.4f} seconds"
)


# And you can iterate over documents as they're yielded:

# In[12]:


for doc in my_iterator:
    print(doc.page_content)


# You can also lazy load documents in an async manner with `.alazy_load()`:

# In[13]:


loader = AirbyteLoader(
    source="source-faker",
    stream="users",
    config={"count": 3},
    template=PromptTemplate.from_template(
        "My name is {name} and I am {height} meters tall."
    ),
)

my_async_iterator = loader.alazy_load()

async for doc in my_async_iterator:
    print(doc.page_content)


# ## Configuration
# 
# `AirbyteLoader` can be configured with the following options:
# 
# - `source` (str, required): The name of the Airbyte source to load from.
# - `stream` (str, required): The name of the stream to load from (Airbyte sources can return multiple streams)
# - `config` (dict, required): The configuration for the Airbyte source
# - `template` (PromptTemplate, optional): A custom prompt template for formatting documents
# - `include_metadata` (bool, optional, default True): Whether to include all fields as metadata in the output documents
# 
# The majority of the configuration will be in `config`, and you can find the specific configuration options in the "Config field reference" for each source in the [Airbyte documentation](https://docs.airbyte.com/integrations/).

# 
