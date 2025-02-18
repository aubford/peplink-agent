#!/usr/bin/env python
# coding: utf-8

# # AWS S3 File
#
# >[Amazon Simple Storage Service (Amazon S3)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html) is an object storage service.
#
# >[AWS S3 Buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingBucket.html)
#
# This covers how to load document objects from an `AWS S3 File` object.

# In[1]:


from langchain_community.document_loaders import S3FileLoader


# In[2]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  boto3")


# In[8]:


loader = S3FileLoader("testing-hwc", "fake.docx")


# In[9]:


loader.load()


# ## Configuring the AWS Boto3 client
# You can configure the AWS [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) client by passing
# named arguments when creating the S3DirectoryLoader.
# This is useful for instance when AWS credentials can't be set as environment variables.
# See the [list of parameters](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html#boto3.session.Session) that can be configured.

# In[ ]:


loader = S3FileLoader(
    "testing-hwc", "fake.docx", aws_access_key_id="xxxx", aws_secret_access_key="yyyy"
)


# In[ ]:


loader.load()
