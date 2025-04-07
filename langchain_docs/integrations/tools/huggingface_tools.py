#!/usr/bin/env python
# coding: utf-8

# # HuggingFace Hub Tools
# 
# >[Huggingface Tools](https://huggingface.co/docs/transformers/v4.29.0/en/custom_tools) that supporting text I/O can be
# loaded directly using the `load_huggingface_tool` function.

# In[ ]:


# Requires transformers>=4.29.0 and huggingface_hub>=0.14.1
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  transformers huggingface_hub > /dev/null')


# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-community')


# In[1]:


from langchain_community.agent_toolkits.load_tools import load_huggingface_tool

tool = load_huggingface_tool("lysandre/hf-model-downloads")

print(f"{tool.name}: {tool.description}")


# In[2]:


tool.run("text-classification")


# In[ ]:




