#!/usr/bin/env python
# coding: utf-8

# # RSS Feeds
# 
# This covers how to load HTML news articles from a list of RSS feed URLs into a document format that we can use downstream.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  feedparser newspaper3k listparser')


# In[32]:


from langchain_community.document_loaders import RSSFeedLoader


# In[33]:


urls = ["https://news.ycombinator.com/rss"]


# Pass in urls to load them into Documents

# In[ ]:


loader = RSSFeedLoader(urls=urls)
data = loader.load()
print(len(data))


# In[35]:


print(data[0].page_content)


# You can pass arguments to the NewsURLLoader which it uses to load articles.

# In[36]:


loader = RSSFeedLoader(urls=urls, nlp=True)
data = loader.load()
print(len(data))


# In[37]:


data[0].metadata["keywords"]


# In[38]:


data[0].metadata["summary"]


# You can also use an OPML file such as a Feedly export.  Pass in either a URL or the OPML contents.

# In[39]:


with open("example_data/sample_rss_feeds.opml", "r") as f:
    loader = RSSFeedLoader(opml=f.read())
data = loader.load()
print(len(data))


# In[40]:


data[0].page_content


# In[ ]:




