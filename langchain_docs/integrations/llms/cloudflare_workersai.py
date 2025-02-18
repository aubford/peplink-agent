#!/usr/bin/env python
# coding: utf-8

# # Cloudflare Workers AI
#
# [Cloudflare AI documentation](https://developers.cloudflare.com/workers-ai/models/) listed all generative text models available.
#
# Both Cloudflare account ID and API token are required. Find how to obtain them from [this document](https://developers.cloudflare.com/workers-ai/get-started/rest-api/).

# In[1]:


from langchain.chains import LLMChain
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain_core.prompts import PromptTemplate

template = """Human: {question}

AI Assistant: """

prompt = PromptTemplate.from_template(template)


# Get authentication before running LLM.

# In[2]:


import getpass

my_account_id = getpass.getpass("Enter your Cloudflare account ID:\n\n")
my_api_token = getpass.getpass("Enter your Cloudflare API token:\n\n")
llm = CloudflareWorkersAI(account_id=my_account_id, api_token=my_api_token)


# In[3]:


llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Why are roses red?"
llm_chain.run(question)


# In[4]:


# Using streaming
for chunk in llm.stream("Why is sky blue?"):
    print(chunk, end=" | ", flush=True)


# In[ ]:
