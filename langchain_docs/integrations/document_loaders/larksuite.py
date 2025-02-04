#!/usr/bin/env python
# coding: utf-8

# # LarkSuite (FeiShu)
# 
# >[LarkSuite](https://www.larksuite.com/) is an enterprise collaboration platform developed by ByteDance.
# 
# This notebook covers how to load data from the `LarkSuite` REST API into a format that can be ingested into LangChain, along with example usage for text summarization.
# 
# The LarkSuite API requires an access token (tenant_access_token or user_access_token), checkout [LarkSuite open platform document](https://open.larksuite.com/document) for API details.

# In[1]:


from getpass import getpass

from langchain_community.document_loaders.larksuite import (
    LarkSuiteDocLoader,
    LarkSuiteWikiLoader,
)

DOMAIN = input("larksuite domain")
ACCESS_TOKEN = getpass("larksuite tenant_access_token or user_access_token")
DOCUMENT_ID = input("larksuite document id")


# ## Load From Document

# In[3]:


from pprint import pprint

larksuite_loader = LarkSuiteDocLoader(DOMAIN, ACCESS_TOKEN, DOCUMENT_ID)
docs = larksuite_loader.load()

pprint(docs)


# ## Load From Wiki

# In[4]:


from pprint import pprint

DOCUMENT_ID = input("larksuite wiki id")
larksuite_loader = LarkSuiteWikiLoader(DOMAIN, ACCESS_TOKEN, DOCUMENT_ID)
docs = larksuite_loader.load()

pprint(docs)


# In[ ]:


# see https://python.langchain.com/docs/use_cases/summarization for more details
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms.fake import FakeListLLM

llm = FakeListLLM()
chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(docs)

