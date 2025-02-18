#!/usr/bin/env python
# coding: utf-8

# # ChatGPT Data
#
# >[ChatGPT](https://chat.openai.com) is an artificial intelligence (AI) chatbot developed by OpenAI.
#
#
# This notebook covers how to load `conversations.json` from your `ChatGPT` data export folder.
#
# You can get your data export by email by going to: https://chat.openai.com/ -> (Profile) - Settings -> Export data -> Confirm export.

# In[ ]:


from langchain_community.document_loaders.chatgpt import ChatGPTLoader


# In[2]:


loader = ChatGPTLoader(log_file="./example_data/fake_conversations.json", num_logs=1)


# In[3]:


loader.load()
