#!/usr/bin/env python
# coding: utf-8

# # Org-mode
#
# >A [Org Mode document](https://en.wikipedia.org/wiki/Org-mode) is a document editing, formatting, and organizing mode, designed for notes, planning, and authoring within the free software text editor Emacs.

# ## `UnstructuredOrgModeLoader`
#
# You can load data from Org-mode files with `UnstructuredOrgModeLoader` using the following workflow.

# In[1]:


from langchain_community.document_loaders import UnstructuredOrgModeLoader

loader = UnstructuredOrgModeLoader(
    file_path="./example_data/README.org", mode="elements"
)
docs = loader.load()

print(docs[0])


# In[ ]:
