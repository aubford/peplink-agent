#!/usr/bin/env python
# coding: utf-8

# # ArxivLoader
#
# [arXiv](https://arxiv.org/) is an open-access archive for 2 million scholarly articles in the fields of physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems science, and economics.

# ## Setup
#
# To access Arxiv document loader you'll need to install the `arxiv`, `PyMuPDF` and `langchain-community` integration packages. PyMuPDF transforms PDF files downloaded from the arxiv.org site into the text format.

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-community arxiv pymupdf")


# ## Instantiation
#
# Now we can instantiate our model object and load documents:

# In[1]:


from langchain_community.document_loaders import ArxivLoader

# Supports all arguments of `ArxivAPIWrapper`
loader = ArxivLoader(
    query="reasoning",
    load_max_docs=2,
    # doc_content_chars_max=1000,
    # load_all_available_meta=False,
    # ...
)


# ## Load
#
# Use ``.load()`` to synchronously load into memory all Documents, with one
# Document per one arxiv paper.
#
# Let's run through a basic example of how to use the `ArxivLoader` searching for papers of reasoning:

# In[2]:


docs = loader.load()
docs[0]


# In[3]:


print(docs[0].metadata)


# ## Lazy Load
#
# If we're loading a  large number of Documents and our downstream operations can be done over subsets of all loaded Documents, we can lazily load our Documents one at a time to minimize our memory footprint:

# In[4]:


docs = []

for doc in loader.lazy_load():
    docs.append(doc)

    if len(docs) >= 10:
        # do some paged operation, e.g.
        # index.upsert(doc)

        docs = []


# In this example we never have more than 10 Documents loaded into memory at a time.

# ## Use papers summaries as documents
#
# You can use summaries of Arvix paper as documents rather than raw papers:

# In[5]:


docs = loader.get_summaries_as_docs()
docs[0]


# ## API reference
#
# For detailed documentation of all ArxivLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.arxiv.ArxivLoader.html#langchain_community.document_loaders.arxiv.ArxivLoader
