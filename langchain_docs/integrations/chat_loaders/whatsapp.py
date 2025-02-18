#!/usr/bin/env python
# coding: utf-8

# # WhatsApp
#
# This notebook shows how to use the WhatsApp chat loader. This class helps map exported WhatsApp conversations to LangChain chat messages.
#
# The process has three steps:
# 1. Export the chat conversations to computer
# 2. Create the `WhatsAppChatLoader` with the file path pointed to the json file or directory of JSON files
# 3. Call `loader.load()` (or `loader.lazy_load()`) to perform the conversion.
#
# ## 1. Create message dump
#
# To make the export of your WhatsApp conversation(s), complete the following steps:
#
# 1. Open the target conversation
# 2. Click the three dots in the top right corner and select "More".
# 3. Then select "Export chat" and choose "Without media".
#
# An example of the data format for each conversation is below:

# In[1]:


get_ipython().run_cell_magic(
    "writefile",
    "whatsapp_chat.txt",
    "[8/15/23, 9:12:33 AM] Dr. Feather: \u200eMessages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.\n[8/15/23, 9:12:43 AM] Dr. Feather: I spotted a rare Hyacinth Macaw yesterday in the Amazon Rainforest. Such a magnificent creature!\n\u200e[8/15/23, 9:12:48 AM] Dr. Feather: \u200eimage omitted\n[8/15/23, 9:13:15 AM] Jungle Jane: That's stunning! Were you able to observe its behavior?\n\u200e[8/15/23, 9:13:23 AM] Dr. Feather: \u200eimage omitted\n[8/15/23, 9:14:02 AM] Dr. Feather: Yes, it seemed quite social with other macaws. They're known for their playful nature.\n[8/15/23, 9:14:15 AM] Jungle Jane: How's the research going on parrot communication?\n\u200e[8/15/23, 9:14:30 AM] Dr. Feather: \u200eimage omitted\n[8/15/23, 9:14:50 AM] Dr. Feather: It's progressing well. We're learning so much about how they use sound and color to communicate.\n[8/15/23, 9:15:10 AM] Jungle Jane: That's fascinating! Can't wait to read your paper on it.\n[8/15/23, 9:15:20 AM] Dr. Feather: Thank you! I'll send you a draft soon.\n[8/15/23, 9:25:16 PM] Jungle Jane: Looking forward to it! Keep up the great work.\n",
)


# ## 2. Create the Chat Loader
#
# The WhatsAppChatLoader accepts the resulting zip file, unzipped directory, or the path to any of the chat `.txt` files therein.
#
# Provide that as well as the user name you want to take on the role of "AI" when fine-tuning.

# In[7]:


from langchain_community.chat_loaders.whatsapp import WhatsAppChatLoader


# In[12]:


loader = WhatsAppChatLoader(
    path="./whatsapp_chat.txt",
)


# ## 3. Load messages
#
# The `load()` (or `lazy_load`) methods return a list of "ChatSessions" that currently store the list of messages per loaded conversation.

# In[13]:


from typing import List

from langchain_community.chat_loaders.utils import (
    map_ai_messages,
    merge_chat_runs,
)
from langchain_core.chat_sessions import ChatSession

raw_messages = loader.lazy_load()
# Merge consecutive messages from the same sender into a single message
merged_messages = merge_chat_runs(raw_messages)
# Convert messages from "Dr. Feather" to AI messages
messages: List[ChatSession] = list(
    map_ai_messages(merged_messages, sender="Dr. Feather")
)


# ### Next Steps
#
# You can then use these messages how you see fit, such as fine-tuning a model, few-shot example selection, or directly make predictions for the next message.

# In[14]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

for chunk in llm.stream(messages[0]["messages"]):
    print(chunk.content, end="", flush=True)


# In[ ]:
