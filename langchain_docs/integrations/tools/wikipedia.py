#!/usr/bin/env python
# coding: utf-8

# # Wikipedia
#
# >[Wikipedia](https://wikipedia.org/) is a multilingual free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing system called MediaWiki. `Wikipedia` is the largest and most-read reference work in history.
#
# First, you need to install `wikipedia` python package.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  wikipedia")


# In[2]:


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# In[3]:


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


# In[4]:


wikipedia.run("HUNTER X HUNTER")
