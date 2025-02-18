#!/usr/bin/env python
# coding: utf-8

# # PySpark
#
# This notebook goes over how to load data from a [PySpark](https://spark.apache.org/docs/latest/api/python/) DataFrame.

# In[1]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  pyspark")


# In[2]:


from pyspark.sql import SparkSession


# In[3]:


spark = SparkSession.builder.getOrCreate()


# In[4]:


df = spark.read.csv("example_data/mlb_teams_2012.csv", header=True)


# In[5]:


from langchain_community.document_loaders import PySparkDataFrameLoader


# In[6]:


loader = PySparkDataFrameLoader(spark, df, page_content_column="Team")


# In[7]:


loader.load()
