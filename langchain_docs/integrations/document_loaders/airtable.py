#!/usr/bin/env python
# coding: utf-8

# # Airtable

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  pyairtable')


# In[7]:


from langchain_community.document_loaders import AirtableLoader


# * Get your API key [here](https://support.airtable.com/docs/creating-and-using-api-keys-and-access-tokens).
# * Get ID of your base [here](https://airtable.com/developers/web/api/introduction).
# * Get your table ID from the table url as shown [here](https://www.highviewapps.com/kb/where-can-i-find-the-airtable-base-id-and-table-id/#:~:text=Both%20the%20Airtable%20Base%20ID,URL%20that%20begins%20with%20tbl).

# In[ ]:


api_key = "xxx"
base_id = "xxx"
table_id = "xxx"
view = "xxx"  # optional


# In[9]:


loader = AirtableLoader(api_key, table_id, base_id, view=view)
docs = loader.load()


# Returns each table row as `dict`.

# In[10]:


len(docs)


# In[11]:


eval(docs[0].page_content)

