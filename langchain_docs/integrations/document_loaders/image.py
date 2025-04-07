#!/usr/bin/env python
# coding: utf-8

# # Images
# 
# This covers how to load images into a document format that we can use downstream with other LangChain modules.
# 
# It uses [Unstructured](https://unstructured.io/) to handle a wide variety of image formats, such as `.jpg` and `.png`. Please see [this guide](/docs/integrations/providers/unstructured/) for more instructions on setting up Unstructured locally, including setting up required system dependencies.

# ## Using Unstructured

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet "unstructured[all-docs]"')


# In[2]:


from langchain_community.document_loaders.image import UnstructuredImageLoader

loader = UnstructuredImageLoader("./example_data/layout-parser-paper-screenshot.png")

data = loader.load()

data[0]


# ### Retain Elements
# 
# Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can keep that separation by specifying `mode="elements"`.

# In[3]:


loader = UnstructuredImageLoader(
    "./example_data/layout-parser-paper-screenshot.png", mode="elements"
)

data = loader.load()

data[0]

