#!/usr/bin/env python
# coding: utf-8

# # Telegram
#
# >[Telegram Messenger](https://web.telegram.org/a/) is a globally accessible freemium, cross-platform, encrypted, cloud-based and centralized instant messaging service. The application also provides optional end-to-end encrypted chats and video calling, VoIP, file sharing and several other features.
#
# This notebook covers how to load data from `Telegram` into a format that can be ingested into LangChain.

# In[1]:


from langchain_community.document_loaders import (
    TelegramChatApiLoader,
    TelegramChatFileLoader,
)


# In[2]:


loader = TelegramChatFileLoader("example_data/telegram.json")


# In[3]:


loader.load()


# `TelegramChatApiLoader` loads data directly from any specified chat from Telegram. In order to export the data, you will need to authenticate your Telegram account.
#
# You can get the API_HASH and API_ID from https://my.telegram.org/auth?to=apps
#
# chat_entity – recommended to be the [entity](https://docs.telethon.dev/en/stable/concepts/entities.html?highlight=Entity#what-is-an-entity) of a channel.
#
#

# In[ ]:


loader = TelegramChatApiLoader(
    chat_entity="<CHAT_URL>",  # recommended to use Entity here
    api_hash="<API HASH >",
    api_id="<API_ID>",
    username="",  # needed only for caching the session.
)


# In[ ]:


loader.load()


# In[ ]:
