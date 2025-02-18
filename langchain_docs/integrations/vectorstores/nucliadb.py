#!/usr/bin/env python
# coding: utf-8

# # NucliaDB
#
# You can use a local NucliaDB instance or use [Nuclia Cloud](https://nuclia.cloud).
#
# When using a local instance, you need a Nuclia Understanding API key, so your texts are properly vectorized and indexed. You can get a key by creating a free account at [https://nuclia.cloud](https://nuclia.cloud), and then [create a NUA key](https://docs.nuclia.dev/docs/docs/using/understanding/intro).

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  langchain langchain-community nuclia"
)


# ## Usage with nuclia.cloud

# In[ ]:


from langchain_community.vectorstores.nucliadb import NucliaDB

API_KEY = "YOUR_API_KEY"

ndb = NucliaDB(knowledge_box="YOUR_KB_ID", local=False, api_key=API_KEY)


# ## Usage with a local instance
#
# Note: By default `backend` is set to `http://localhost:8080`.

# In[ ]:


from langchain_community.vectorstores.nucliadb import NucliaDB

ndb = NucliaDB(knowledge_box="YOUR_KB_ID", local=True, backend="http://my-local-server")


# ## Add and delete texts to your Knowledge Box

# In[ ]:


ids = ndb.add_texts(["This is a new test", "This is a second test"])


# In[ ]:


ndb.delete(ids=ids)


# ## Search in your Knowledge Box

# In[ ]:


results = ndb.similarity_search("Who was inspired by Ada Lovelace?")
print(results[0].page_content)
