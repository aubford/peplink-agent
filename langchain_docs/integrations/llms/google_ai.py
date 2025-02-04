#!/usr/bin/env python
# coding: utf-8
---
keywords: [gemini, GoogleGenerativeAI, gemini-pro]
---
# # Google AI
# 

# :::caution
# You are currently on a page documenting the use of Google models as [text completion models](/docs/concepts/text_llms). Many popular Google models are [chat completion models](/docs/concepts/chat_models).
# 
# You may be looking for [this page instead](/docs/integrations/chat/google_generative_ai/).
# :::
# 
# A guide on using [Google Generative AI](https://developers.generativeai.google/) models with Langchain. Note: It's separate from Google Cloud Vertex AI [integration](/docs/integrations/llms/google_vertex_ai_palm).

# ## Setting up
# 

# To use Google Generative AI you must install the `langchain-google-genai` Python package and generate an API key. [Read more details](https://developers.generativeai.google/).

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-google-genai')


# In[ ]:


from langchain_google_genai import GoogleGenerativeAI


# In[2]:


from getpass import getpass

api_key = getpass()


# In[3]:


llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)
print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language?"
    )
)


# In[4]:


llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language?"
    )
)


# ## Using in a chain

# In[5]:


from langchain_core.prompts import PromptTemplate


# In[6]:


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "How much is 2+2?"
print(chain.invoke({"question": question}))


# ## Streaming calls

# In[7]:


import sys

for chunk in llm.stream("Tell me a short poem about snow"):
    sys.stdout.write(chunk)
    sys.stdout.flush()


# ### Safety Settings
# 
# Gemini models have default safety settings that can be overridden. If you are receiving lots of "Safety Warnings" from your models, you can try tweaking the `safety_settings` attribute of the model. For example, to turn off safety blocking for dangerous content, you can construct your LLM as follows:

# In[ ]:


from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory

llm = GoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=api_key,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)


# For an enumeration of the categories and thresholds available, see Google's [safety setting types](https://ai.google.dev/api/python/google/generativeai/types/SafetySettingDict).
