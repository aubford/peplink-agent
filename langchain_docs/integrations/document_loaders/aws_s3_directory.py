#!/usr/bin/env python
# coding: utf-8

# # AWS S3 Directory
# 
# >[Amazon Simple Storage Service (Amazon S3)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html) is an object storage service
# 
# >[AWS S3 Directory](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html)
# 
# This covers how to load document objects from an `AWS S3 Directory` object.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  boto3')


# In[2]:


from langchain_community.document_loaders import S3DirectoryLoader


# In[3]:


loader = S3DirectoryLoader("testing-hwc")


# In[ ]:


loader.load()


# ## Specifying a prefix
# You can also specify a prefix for more finegrained control over what files to load.

# In[5]:


loader = S3DirectoryLoader("testing-hwc", prefix="fake")


# In[6]:


loader.load()


# ## Configuring the AWS Boto3 client
# You can configure the AWS [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) client by passing
# named arguments when creating the S3DirectoryLoader.
# This is useful for instance when AWS credentials can't be set as environment variables.
# See the [list of parameters](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html#boto3.session.Session) that can be configured.

# In[ ]:


loader = S3DirectoryLoader(
    "testing-hwc", aws_access_key_id="xxxx", aws_secret_access_key="yyyy"
)


# In[ ]:


loader.load()

