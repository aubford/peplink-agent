#!/usr/bin/env python
# coding: utf-8

# # EPub
#
# >[EPUB](https://en.wikipedia.org/wiki/EPUB) is an e-book file format that uses the ".epub" file extension. The term is short for electronic publication and is sometimes styled ePub. `EPUB` is supported by many e-readers, and compatible software is available for most smartphones, tablets, and computers.
#
# This covers how to load `.epub` documents into the Document format that we can use downstream. You'll need to install the [`pandoc`](https://pandoc.org/installing.html) package for this loader to work with e.g. `brew install pandoc` for OSX.
#
# Please see [this guide](/docs/integrations/providers/unstructured/) for more instructions on setting up Unstructured locally, including setting up required system dependencies.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet unstructured")


# In[5]:


from langchain_community.document_loaders import UnstructuredEPubLoader

loader = UnstructuredEPubLoader("./example_data/childrens-literature.epub")

data = loader.load()

data[0]


# ## Retain Elements
#
# Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.

# In[7]:


loader = UnstructuredEPubLoader(
    "./example_data/childrens-literature.epub", mode="elements"
)

data = loader.load()

data[0]


# In[ ]:
