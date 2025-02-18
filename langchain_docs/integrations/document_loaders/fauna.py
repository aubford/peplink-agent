#!/usr/bin/env python
# coding: utf-8

# # Fauna
#
# >[Fauna](https://fauna.com/) is a Document Database.
#
# Query `Fauna` documents

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  fauna")


# ## Query data example

# In[ ]:


from langchain_community.document_loaders.fauna import FaunaLoader

secret = "<enter-valid-fauna-secret>"
query = "Item.all()"  # Fauna query. Assumes that the collection is called "Item"
field = "text"  # The field that contains the page content. Assumes that the field is called "text"

loader = FaunaLoader(query, field, secret)
docs = loader.lazy_load()

for value in docs:
    print(value)


# ### Query with Pagination
# You get a `after` value if there are more data. You can get values after the curcor by passing in the `after` string in query.
#
# To learn more following [this link](https://fqlx-beta--fauna-docs.netlify.app/fqlx/beta/reference/schema_entities/set/static-paginate)

# In[ ]:


query = """
Item.paginate("hs+DzoPOg ... aY1hOohozrV7A")
Item.all()
"""
loader = FaunaLoader(query, field, secret)
