#!/usr/bin/env python
# coding: utf-8

# # College Confidential
#
# >[College Confidential](https://www.collegeconfidential.com/) gives information on 3,800+ colleges and universities.
#
# This covers how to load `College Confidential` webpages into a document format that we can use downstream.

# In[1]:


from langchain_community.document_loaders import CollegeConfidentialLoader


# In[2]:


loader = CollegeConfidentialLoader(
    "https://www.collegeconfidential.com/colleges/brown-university/"
)


# In[3]:


data = loader.load()


# In[4]:


data


# In[ ]:
