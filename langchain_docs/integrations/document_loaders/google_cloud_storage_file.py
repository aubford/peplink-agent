#!/usr/bin/env python
# coding: utf-8

# # Google Cloud Storage File
# 
# >[Google Cloud Storage](https://en.wikipedia.org/wiki/Google_Cloud_Storage) is a managed service for storing unstructured data.
# 
# This covers how to load document objects from an `Google Cloud Storage (GCS) file object (blob)`.

# In[2]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-google-community[gcs]')


# In[1]:


from langchain_google_community import GCSFileLoader


# In[3]:


loader = GCSFileLoader(project_name="aist", bucket="testing-hwc", blob="fake.docx")


# In[4]:


loader.load()


# If you want to use an alternative loader, you can provide a custom function, for example:

# In[ ]:


from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    return PyPDFLoader(file_path)


loader = GCSFileLoader(
    project_name="aist", bucket="testing-hwc", blob="fake.pdf", loader_func=load_pdf
)

