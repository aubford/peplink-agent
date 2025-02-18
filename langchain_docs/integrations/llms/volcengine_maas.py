#!/usr/bin/env python
# coding: utf-8

# # Volc Engine Maas
#
# This notebook provides you with a guide on how to get started with Volc Engine's MaaS llm models.

# In[ ]:


# Install the package
get_ipython().run_line_magic("pip", "install --upgrade --quiet  volcengine")


# In[2]:


from langchain_community.llms import VolcEngineMaasLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# In[3]:


llm = VolcEngineMaasLLM(volc_engine_maas_ak="your ak", volc_engine_maas_sk="your sk")


# or you can set access_key and secret_key in your environment variables
# ```bash
# export VOLC_ACCESSKEY=YOUR_AK
# export VOLC_SECRETKEY=YOUR_SK
# ```

# In[8]:


chain = PromptTemplate.from_template("给我讲个笑话") | llm | StrOutputParser()
chain.invoke({})
