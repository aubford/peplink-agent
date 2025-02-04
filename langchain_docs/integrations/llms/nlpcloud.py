#!/usr/bin/env python
# coding: utf-8

# # NLP Cloud
# 
# The [NLP Cloud](https://nlpcloud.io) serves high performance pre-trained or custom models for NER, sentiment-analysis, classification, summarization, paraphrasing, grammar and spelling correction, keywords and keyphrases extraction, chatbot, product description and ad generation, intent classification, text generation, image generation, blog post generation, code generation, question answering, automatic speech recognition, machine translation, language detection, semantic search, semantic similarity, tokenization, POS tagging, embeddings, and dependency parsing. It is ready for production, served through a REST API.
# 
# 
# This example goes over how to use LangChain to interact with `NLP Cloud` [models](https://docs.nlpcloud.com/#models).

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  nlpcloud')


# In[3]:


# get a token: https://docs.nlpcloud.com/#authentication

from getpass import getpass

NLPCLOUD_API_KEY = getpass()


# In[5]:


import os

os.environ["NLPCLOUD_API_KEY"] = NLPCLOUD_API_KEY


# In[6]:


from langchain.chains import LLMChain
from langchain_community.llms import NLPCloud
from langchain_core.prompts import PromptTemplate


# In[7]:


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


# In[8]:


llm = NLPCloud()


# In[9]:


llm_chain = LLMChain(prompt=prompt, llm=llm)


# In[10]:


question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)

