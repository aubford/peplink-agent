#!/usr/bin/env python
# coding: utf-8

# # Open City Data

# [Socrata](https://dev.socrata.com/foundry/data.sfgov.org/vw6y-z8j6) provides an API for city open data.
#
# For a dataset such as [SF crime](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry), to to the `API` tab on top right.
#
# That provides you with the `dataset identifier`.
#
# Use the dataset identifier to grab specific tables for a given city_id (`data.sfgov.org`) -
#
# E.g., `vw6y-z8j6` for [SF 311 data](https://dev.socrata.com/foundry/data.sfgov.org/vw6y-z8j6).
#
# E.g., `tmnf-yvry` for [SF Police data](https://dev.socrata.com/foundry/data.sfgov.org/tmnf-yvry).

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  sodapy")


# In[1]:


from langchain_community.document_loaders import OpenCityDataLoader


# In[2]:


dataset = "vw6y-z8j6"  # 311 data
dataset = "tmnf-yvry"  # crime data
loader = OpenCityDataLoader(city_id="data.sfgov.org", dataset_id=dataset, limit=2000)


# In[3]:


docs = loader.load()


# In[4]:


eval(docs[0].page_content)
