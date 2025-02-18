#!/usr/bin/env python
# coding: utf-8

# ## Getting started with LangChain and Gemma, running locally or in the Cloud

# ### Installing dependencies

# In[1]:


get_ipython().system("pip install --upgrade langchain langchain-google-vertexai")


# ### Running the model

# Go to the VertexAI Model Garden on Google Cloud [console](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/335), and deploy the desired version of Gemma to VertexAI. It will take a few minutes, and after the endpoint it ready, you need to copy its number.

# In[1]:


# @title Basic parameters
project: str = "PUT_YOUR_PROJECT_ID_HERE"  # @param {type:"string"}
endpoint_id: str = "PUT_YOUR_ENDPOINT_ID_HERE"  # @param {type:"string"}
location: str = "PUT_YOUR_ENDPOINT_LOCAtION_HERE"  # @param {type:"string"}


# In[3]:


from langchain_google_vertexai import (
    GemmaChatVertexAIModelGarden,
    GemmaVertexAIModelGarden,
)


# In[4]:


llm = GemmaVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)


# In[5]:


output = llm.invoke("What is the meaning of life?")
print(output)


# We can also use Gemma as a multi-turn chat model:

# In[7]:


from langchain_core.messages import HumanMessage

llm = GemmaChatVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

message1 = HumanMessage(content="How much is 2+2?")
answer1 = llm.invoke([message1])
print(answer1)

message2 = HumanMessage(content="How much is 3+3?")
answer2 = llm.invoke([message1, answer1, message2])

print(answer2)


# You can post-process response to avoid repetitions:

# In[8]:


answer1 = llm.invoke([message1], parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], parse_response=True)

print(answer2)


# ## Running Gemma locally from Kaggle

# In order to run Gemma locally, you can download it from Kaggle first. In order to do this, you'll need to login into the Kaggle platform, create a API key and download a `kaggle.json` Read more about Kaggle auth [here](https://www.kaggle.com/docs/api).

# ### Installation

# In[7]:


get_ipython().system("mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/kaggle.json")


# In[11]:


get_ipython().system("pip install keras>=3 keras_nlp")


# ### Usage

# In[1]:


from langchain_google_vertexai import GemmaLocalKaggle


# You can specify the keras backend (by default it's `tensorflow`, but you can change it be `jax` or `torch`).

# In[2]:


# @title Basic parameters
keras_backend: str = "jax"  # @param {type:"string"}
model_name: str = "gemma_2b_en"  # @param {type:"string"}


# In[3]:


llm = GemmaLocalKaggle(model_name=model_name, keras_backend=keras_backend)


# In[7]:


output = llm.invoke("What is the meaning of life?", max_tokens=30)
print(output)


# ### ChatModel

# Same as above, using Gemma locally as a multi-turn chat model. You might need to re-start the notebook and clean your GPU memory in order to avoid OOM errors:

# In[1]:


from langchain_google_vertexai import GemmaChatLocalKaggle


# In[2]:


# @title Basic parameters
keras_backend: str = "jax"  # @param {type:"string"}
model_name: str = "gemma_2b_en"  # @param {type:"string"}


# In[3]:


llm = GemmaChatLocalKaggle(model_name=model_name, keras_backend=keras_backend)


# In[4]:


from langchain_core.messages import HumanMessage

message1 = HumanMessage(content="Hi! Who are you?")
answer1 = llm.invoke([message1], max_tokens=30)
print(answer1)


# In[5]:


message2 = HumanMessage(content="What can you help me with?")
answer2 = llm.invoke([message1, answer1, message2], max_tokens=60)

print(answer2)


# You can post-process the response if you want to avoid multi-turn statements:

# In[7]:


answer1 = llm.invoke([message1], max_tokens=30, parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], max_tokens=60, parse_response=True)
print(answer2)


# ## Running Gemma locally from HuggingFace

# In[1]:


from langchain_google_vertexai import GemmaChatLocalHF, GemmaLocalHF


# In[2]:


# @title Basic parameters
hf_access_token: str = "PUT_YOUR_TOKEN_HERE"  # @param {type:"string"}
model_name: str = "google/gemma-2b"  # @param {type:"string"}


# In[4]:


llm = GemmaLocalHF(model_name="google/gemma-2b", hf_access_token=hf_access_token)


# In[6]:


output = llm.invoke("What is the meaning of life?", max_tokens=50)
print(output)


# Same as above, using Gemma locally as a multi-turn chat model. You might need to re-start the notebook and clean your GPU memory in order to avoid OOM errors:

# In[3]:


llm = GemmaChatLocalHF(model_name=model_name, hf_access_token=hf_access_token)


# In[4]:


from langchain_core.messages import HumanMessage

message1 = HumanMessage(content="Hi! Who are you?")
answer1 = llm.invoke([message1], max_tokens=60)
print(answer1)


# In[8]:


message2 = HumanMessage(content="What can you help me with?")
answer2 = llm.invoke([message1, answer1, message2], max_tokens=140)

print(answer2)


# And the same with posprocessing:

# In[11]:


answer1 = llm.invoke([message1], max_tokens=60, parse_response=True)
print(answer1)

answer2 = llm.invoke([message1, answer1, message2], max_tokens=120, parse_response=True)
print(answer2)


# In[ ]:
