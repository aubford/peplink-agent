#!/usr/bin/env python
# coding: utf-8

# # RELLM
#
# [RELLM](https://github.com/r2d4/rellm) is a library that wraps local Hugging Face pipeline models for structured decoding.
#
# It works by generating tokens one at a time. At each step, it masks tokens that don't conform to the provided partial regular expression.
#
#
# **Warning - this module is still experimental**

# In[1]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  rellm langchain-huggingface > /dev/null"
)


# ### Hugging Face Baseline
#
# First, let's establish a qualitative baseline by checking the output of the model without structured decoding.

# In[2]:


import logging

logging.basicConfig(level=logging.ERROR)
prompt = """Human: "What's the capital of the United States?"
AI Assistant:{
  "action": "Final Answer",
  "action_input": "The capital of the United States is Washington D.C."
}
Human: "What's the capital of Pennsylvania?"
AI Assistant:{
  "action": "Final Answer",
  "action_input": "The capital of Pennsylvania is Harrisburg."
}
Human: "What 2 + 5?"
AI Assistant:{
  "action": "Final Answer",
  "action_input": "2 + 5 = 7."
}
Human: 'What's the capital of Maryland?'
AI Assistant:"""


# In[3]:


from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

hf_model = pipeline(
    "text-generation", model="cerebras/Cerebras-GPT-590M", max_new_tokens=200
)

original_model = HuggingFacePipeline(pipeline=hf_model)

generated = original_model.generate([prompt], stop=["Human:"])
print(generated)


# ***That's not so impressive, is it? It didn't answer the question and it didn't follow the JSON format at all! Let's try with the structured decoder.***

# ## RELLM LLM Wrapper
#
# Let's try that again, now providing a regex to match the JSON structured format.

# In[4]:


import regex  # Note this is the regex library NOT python's re stdlib module

# We'll choose a regex that matches to a structured json string that looks like:
# {
#  "action": "Final Answer",
# "action_input": string or dict
# }
pattern = regex.compile(
    r'\{\s*"action":\s*"Final Answer",\s*"action_input":\s*(\{.*\}|"[^"]*")\s*\}\nHuman:'
)


# In[5]:


from langchain_experimental.llms import RELLM

model = RELLM(pipeline=hf_model, regex=pattern, max_new_tokens=200)

generated = model.predict(prompt, stop=["Human:"])
print(generated)


# **Voila! Free of parsing errors.**

# In[ ]:
