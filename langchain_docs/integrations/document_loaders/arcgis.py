#!/usr/bin/env python
# coding: utf-8

# # ArcGIS
# 
# This notebook demonstrates the use of the `langchain_community.document_loaders.ArcGISLoader` class.
# 
# You will need to install the ArcGIS API for Python `arcgis` and, optionally, `bs4.BeautifulSoup`.
# 
# You can use an `arcgis.gis.GIS` object for authenticated data loading, or leave it blank to access public data.

# In[1]:


from langchain_community.document_loaders import ArcGISLoader

URL = "https://maps1.vcgov.org/arcgis/rest/services/Beaches/MapServer/7"
loader = ArcGISLoader(URL)

docs = loader.load()


# Let's measure loader latency.

# In[2]:


get_ipython().run_cell_magic('time', '', '\ndocs = loader.load()\n')


# In[3]:


docs[0].metadata


# ### Retrieving Geometries  
# 
# 
# If you want to retrieve feature geometries, you may do so with the `return_geometry` keyword.
# 
# Each document's geometry will be stored in its metadata dictionary.

# In[4]:


loader_geom = ArcGISLoader(URL, return_geometry=True)


# In[5]:


get_ipython().run_cell_magic('time', '', '\ndocs = loader_geom.load()\n')


# In[6]:


docs[0].metadata["geometry"]


# In[7]:


for doc in docs:
    print(doc.page_content)

