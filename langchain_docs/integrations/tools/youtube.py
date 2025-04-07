#!/usr/bin/env python
# coding: utf-8

# # YouTube
# 
# >[YouTube Search](https://github.com/joetats/youtube_search) package searches `YouTube` videos avoiding using their heavily rate-limited API.
# >
# >It uses the form on the `YouTube` homepage and scrapes the resulting page.
# 
# This notebook shows how to use a tool to search YouTube.
# 
# Adapted from [https://github.com/venuv/langchain_yt_tools](https://github.com/venuv/langchain_yt_tools)

# In[5]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  youtube_search')


# In[1]:


from langchain_community.tools import YouTubeSearchTool


# In[2]:


tool = YouTubeSearchTool()


# In[6]:


tool.run("lex fridman")


# You can also specify the number of results that are returned

# In[7]:


tool.run("lex friedman,5")


# In[ ]:




