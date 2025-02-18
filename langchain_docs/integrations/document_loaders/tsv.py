#!/usr/bin/env python
# coding: utf-8

# # TSV
#
# >A [tab-separated values (TSV)](https://en.wikipedia.org/wiki/Tab-separated_values) file is a simple, text-based file format for storing tabular data.[3] Records are separated by newlines, and values within a record are separated by tab characters.

# ## `UnstructuredTSVLoader`
#
# You can also load the table using the `UnstructuredTSVLoader`. One advantage of using `UnstructuredTSVLoader` is that if you use it in `"elements"` mode, an HTML representation of the table will be available in the metadata.

# In[1]:


from langchain_community.document_loaders.tsv import UnstructuredTSVLoader

loader = UnstructuredTSVLoader(
    file_path="./example_data/mlb_teams_2012.csv", mode="elements"
)
docs = loader.load()

print(docs[0].metadata["text_as_html"])


# In[ ]:
