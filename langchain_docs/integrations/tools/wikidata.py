#!/usr/bin/env python
# coding: utf-8

# # Wikidata
#
# >[Wikidata](https://wikidata.org/) is a free and open knowledge base that can be read and edited by both humans and machines. Wikidata is one of the world's largest open knowledge bases.
#
# First, you need to install `wikibase-rest-api-client` and `mediawikiapi` python packages.

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet wikibase-rest-api-client mediawikiapi"
)


# In[2]:


from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

print(wikidata.run("Alan Turing"))


# In[ ]:
