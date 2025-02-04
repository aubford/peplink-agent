#!/usr/bin/env python
# coding: utf-8

# # Git
# 
# >[Git](https://en.wikipedia.org/wiki/Git) is a distributed version control system that tracks changes in any set of computer files, usually used for coordinating work among programmers collaboratively developing source code during software development.
# 
# This notebook shows how to load text files from `Git` repository.

# ## Load existing repository from disk

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  GitPython')


# In[2]:


from git import Repo

repo = Repo.clone_from(
    "https://github.com/langchain-ai/langchain", to_path="./example_data/test_repo1"
)
branch = repo.head.reference


# In[ ]:


from langchain_community.document_loaders import GitLoader


# In[6]:


loader = GitLoader(repo_path="./example_data/test_repo1/", branch=branch)


# In[8]:


data = loader.load()


# In[ ]:


len(data)


# In[9]:


print(data[0])


# ## Clone repository from url

# In[5]:


from langchain_community.document_loaders import GitLoader


# In[2]:


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./example_data/test_repo2/",
    branch="master",
)


# In[10]:


data = loader.load()


# In[11]:


len(data)


# ## Filtering files to load

# In[12]:


from langchain_community.document_loaders import GitLoader

# e.g. loading only python files
loader = GitLoader(
    repo_path="./example_data/test_repo1/",
    file_filter=lambda file_path: file_path.endswith(".py"),
)


# In[ ]:




