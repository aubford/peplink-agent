#!/usr/bin/env python
# coding: utf-8

# # How to use multimodal prompts
# 
# Here we demonstrate how to use prompt templates to format [multimodal](/docs/concepts/multimodality/) inputs to models. 
# 
# In this example we will ask a [model](/docs/concepts/chat_models/#multimodality) to describe an image.

# In[7]:


import base64

import httpx

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")


# In[6]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")


# In[10]:


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Describe the image provided"),
        (
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                }
            ],
        ),
    ]
)


# In[11]:


chain = prompt | model


# In[13]:


response = chain.invoke({"image_data": image_data})
print(response.content)


# We can also pass in multiple images.

# In[14]:


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "compare the two pictures provided"),
        (
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data1}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data2}"},
                },
            ],
        ),
    ]
)


# In[15]:


chain = prompt | model


# In[16]:


response = chain.invoke({"image_data1": image_data, "image_data2": image_data})
print(response.content)


# In[ ]:




