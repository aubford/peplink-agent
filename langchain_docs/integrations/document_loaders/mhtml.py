#!/usr/bin/env python
# coding: utf-8

# # mhtml
#
# MHTML is a is used both for emails but also for archived webpages. MHTML, sometimes referred as MHT, stands for MIME HTML is a single file in which entire webpage is archived. When one saves a webpage as MHTML format, this file extension will contain HTML code, images, audio files, flash animation etc.

# In[ ]:


from langchain_community.document_loaders import MHTMLLoader


# In[12]:


# Create a new loader object for the MHTML file
loader = MHTMLLoader(
    file_path="../../../../../../tests/integration_tests/examples/example.mht"
)

# Load the document from the file
documents = loader.load()

# Print the documents to see the results
for doc in documents:
    print(doc)
