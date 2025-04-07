#!/usr/bin/env python
# coding: utf-8

# # Google Cloud Storage Directory
# 
# >[Google Cloud Storage](https://en.wikipedia.org/wiki/Google_Cloud_Storage) is a managed service for storing unstructured data.
# 
# This covers how to load document objects from an `Google Cloud Storage (GCS) directory (bucket)`.

# In[2]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-google-community[gcs]')


# In[1]:


from langchain_google_community import GCSDirectoryLoader


# In[3]:


loader = GCSDirectoryLoader(project_name="aist", bucket="testing-hwc")


# In[4]:


loader.load()


# ## Specifying a prefix
# You can also specify a prefix for more finegrained control over what files to load -including loading all files from a specific folder-.

# In[6]:


loader = GCSDirectoryLoader(project_name="aist", bucket="testing-hwc", prefix="fake")


# In[7]:


loader.load()


# ## Continue on failure to load a single file
# Files in a GCS bucket may cause errors during processing. Enable the `continue_on_failure=True` argument to allow silent failure. This means failure to process a single file will not break the function, it will log a warning instead. 

# In[ ]:


loader = GCSDirectoryLoader(
    project_name="aist", bucket="testing-hwc", continue_on_failure=True
)


# In[ ]:


loader.load()

