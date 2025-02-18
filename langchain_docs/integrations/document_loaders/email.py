#!/usr/bin/env python
# coding: utf-8

# # Email
#
# This notebook shows how to load email (`.eml`) or `Microsoft Outlook` (`.msg`) files.
#
# Please see [this guide](/docs/integrations/providers/unstructured/) for more instructions on setting up Unstructured locally, including setting up required system dependencies.

# ## Using Unstructured

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet unstructured")


# In[3]:


from langchain_community.document_loaders import UnstructuredEmailLoader

loader = UnstructuredEmailLoader("./example_data/fake-email.eml")

data = loader.load()

data


# ### Retain Elements
#
# Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.

# In[4]:


loader = UnstructuredEmailLoader("example_data/fake-email.eml", mode="elements")

data = loader.load()

data[0]


# ### Processing Attachments
#
# You can process attachments with `UnstructuredEmailLoader` by setting `process_attachments=True` in the constructor. By default, attachments will be partitioned using the `partition` function from `unstructured`. You can use a different partitioning function by passing the function to the `attachment_partitioner` kwarg.

# In[5]:


loader = UnstructuredEmailLoader(
    "example_data/fake-email.eml",
    mode="elements",
    process_attachments=True,
)

data = loader.load()

data[0]


# ## Using OutlookMessageLoader

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet extract_msg")


# In[7]:


from langchain_community.document_loaders import OutlookMessageLoader

loader = OutlookMessageLoader("example_data/fake-email.msg")

data = loader.load()

data[0]


# In[ ]:
