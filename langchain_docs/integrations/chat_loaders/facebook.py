#!/usr/bin/env python
# coding: utf-8

# # Facebook Messenger
#
# This notebook shows how to load data from Facebook in a format you can fine-tune on. The overall steps are:
#
# 1. Download your messenger data to disk.
# 2. Create the Chat Loader and call `loader.load()` (or `loader.lazy_load()`) to perform the conversion.
# 3. Optionally use `merge_chat_runs` to combine message from the same sender in sequence, and/or `map_ai_messages` to convert messages from the specified sender to the "AIMessage" class. Once you've done this, call `convert_messages_for_finetuning` to prepare your data for fine-tuning.
#
#
# Once this has been done, you can fine-tune your model. To do so you would complete the following steps:
#
# 4. Upload your messages to OpenAI and run a fine-tuning job.
# 6. Use the resulting model in your LangChain app!
#
#
# Let's begin.
#
#
# ## 1. Download Data
#
# To download your own messenger data, following instructions [here](https://www.zapptales.com/en/download-facebook-messenger-chat-history-how-to/). IMPORTANT - make sure to download them in JSON format (not HTML).
#
# We are hosting an example dump at [this google drive link](https://drive.google.com/file/d/1rh1s1o2i7B-Sk1v9o8KNgivLVGwJ-osV/view?usp=sharing) that we will use in this walkthrough.

# In[1]:


# This uses some example data
import zipfile

import requests


def download_and_unzip(url: str, output_path: str = "file.zip") -> None:
    file_id = url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(download_url)
    if response.status_code != 200:
        print("Failed to download the file.")
        return

    with open(output_path, "wb") as file:
        file.write(response.content)
        print(f"File {output_path} downloaded.")

    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall()
        print(f"File {output_path} has been unzipped.")


# URL of the file to download
url = (
    "https://drive.google.com/file/d/1rh1s1o2i7B-Sk1v9o8KNgivLVGwJ-osV/view?usp=sharing"
)

# Download and unzip
download_and_unzip(url)


# ## 2. Create Chat Loader
#
# We have 2 different `FacebookMessengerChatLoader` classes, one for an entire directory of chats, and one to load individual files. We

# In[2]:


directory_path = "./hogwarts"


# In[3]:


from langchain_community.chat_loaders.facebook_messenger import (
    FolderFacebookMessengerChatLoader,
    SingleFileFacebookMessengerChatLoader,
)


# In[4]:


loader = SingleFileFacebookMessengerChatLoader(
    path="./hogwarts/inbox/HermioneGranger/messages_Hermione_Granger.json",
)


# In[5]:


chat_session = loader.load()[0]
chat_session["messages"][:3]


# In[6]:


loader = FolderFacebookMessengerChatLoader(
    path="./hogwarts",
)


# In[7]:


chat_sessions = loader.load()
len(chat_sessions)


# ## 3. Prepare for fine-tuning
#
# Calling `load()` returns all the chat messages we could extract as human messages. When conversing with chat bots, conversations typically follow a more strict alternating dialogue pattern relative to real conversations.
#
# You can choose to merge message "runs" (consecutive messages from the same sender) and select a sender to represent the "AI". The fine-tuned LLM will learn to generate these AI messages.

# In[8]:


from langchain_community.chat_loaders.utils import (
    map_ai_messages,
    merge_chat_runs,
)


# In[9]:


merged_sessions = merge_chat_runs(chat_sessions)
alternating_sessions = list(map_ai_messages(merged_sessions, "Harry Potter"))


# In[10]:


# Now all of Harry Potter's messages will take the AI message class
# which maps to the 'assistant' role in OpenAI's training format
alternating_sessions[0]["messages"][:3]


# #### Now we can convert to OpenAI format dictionaries

# In[11]:


from langchain_community.adapters.openai import convert_messages_for_finetuning


# In[12]:


training_data = convert_messages_for_finetuning(alternating_sessions)
print(f"Prepared {len(training_data)} dialogues for training")


# In[13]:


training_data[0][:3]


# OpenAI currently requires at least 10 training examples for a fine-tuning job, though they recommend between 50-100 for most tasks. Since we only have 9 chat sessions, we can subdivide them (optionally with some overlap) so that each training example is comprised of a portion of a whole conversation.
#
# Facebook chat sessions (1 per person) often span multiple days and conversations,
# so the long-range dependencies may not be that important to model anyhow.

# In[14]:


# Our chat is alternating, we will make each datapoint a group of 8 messages,
# with 2 messages overlapping
chunk_size = 8
overlap = 2

training_examples = [
    conversation_messages[i : i + chunk_size]
    for conversation_messages in training_data
    for i in range(0, len(conversation_messages) - chunk_size + 1, chunk_size - overlap)
]

len(training_examples)


# ## 4. Fine-tune the model
#
# It's time to fine-tune the model. Make sure you have `openai` installed
# and have set your `OPENAI_API_KEY` appropriately

# In[15]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  langchain-openai")


# In[16]:


import json
import time
from io import BytesIO

import openai

# We will write the jsonl file in memory
my_file = BytesIO()
for m in training_examples:
    my_file.write((json.dumps({"messages": m}) + "\n").encode("utf-8"))

my_file.seek(0)
training_file = openai.files.create(file=my_file, purpose="fine-tune")

# OpenAI audits each training file for compliance reasons.
# This make take a few minutes
status = openai.files.retrieve(training_file.id).status
start_time = time.time()
while status != "processed":
    print(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    status = openai.files.retrieve(training_file.id).status
print(f"File {training_file.id} ready after {time.time() - start_time:.2f} seconds.")


# With the file ready, it's time to kick off a training job.

# In[17]:


job = openai.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo",
)


# Grab a cup of tea while your model is being prepared. This may take some time!

# In[18]:


status = openai.fine_tuning.jobs.retrieve(job.id).status
start_time = time.time()
while status != "succeeded":
    print(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    job = openai.fine_tuning.jobs.retrieve(job.id)
    status = job.status


# In[19]:


print(job.fine_tuned_model)


# ## 5. Use in LangChain
#
# You can use the resulting model ID directly the `ChatOpenAI` model class.

# In[20]:


from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model=job.fine_tuned_model,
    temperature=1,
)


# In[21]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
    ]
)

chain = prompt | model | StrOutputParser()


# In[22]:


for tok in chain.stream({"input": "What classes are you taking?"}):
    print(tok, end="", flush=True)
