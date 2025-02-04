#!/usr/bin/env python
# coding: utf-8

# # Predibase
# 
# [Predibase](https://predibase.com/) allows you to train, fine-tune, and deploy any ML model—from linear regression to large language model. 
# 
# This example demonstrates using Langchain with models deployed on Predibase

# # Setup
# 
# To run this notebook, you'll need a [Predibase account](https://predibase.com/free-trial/?utm_source=langchain) and an [API key](https://docs.predibase.com/sdk-guide/intro).
# 
# You'll also need to install the Predibase Python package:

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  predibase')
import os

os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"


# ## Initial Call

# In[ ]:


from langchain_community.llms import Predibase

model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
)


# In[ ]:


from langchain_community.llms import Predibase

# With a fine-tuned adapter hosted at Predibase (adapter_version must be specified).
model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="e2e_nlg",
    adapter_version=1,
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)


# In[ ]:


from langchain_community.llms import Predibase

# With a fine-tuned adapter hosted at HuggingFace (adapter_version does not apply and will be ignored).
model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="predibase/e2e_nlg",
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)


# In[ ]:


# Optionally use `kwargs` to dynamically overwrite "generate()" settings.
response = model.invoke(
    "Can you recommend me a nice dry wine?",
    **{"temperature": 0.5, "max_new_tokens": 1024},
)
print(response)


# ## Chain Call Setup

# In[ ]:


from langchain_community.llms import Predibase

model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)


# In[ ]:


# With a fine-tuned adapter hosted at Predibase (adapter_version must be specified).
model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="e2e_nlg",
    adapter_version=1,
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)


# In[ ]:


# With a fine-tuned adapter hosted at HuggingFace (adapter_version does not apply and will be ignored).
llm = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="predibase/e2e_nlg",
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)


# ##  SequentialChain

# In[ ]:


from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


# In[ ]:


# This is an LLMChain to write a synopsis given a title of a play.
template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)


# In[ ]:


# This is an LLMChain to write a review of a play given a synopsis.
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)


# In[ ]:


# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(
    chains=[synopsis_chain, review_chain], verbose=True
)


# In[ ]:


review = overall_chain.run("Tragedy at sunset on the beach")


# ## Fine-tuned LLM (Use your own fine-tuned LLM from Predibase)

# In[ ]:


from langchain_community.llms import Predibase

model = Predibase(
    model="my-base-LLM",
    predibase_api_key=os.environ.get(
        "PREDIBASE_API_TOKEN"
    ),  # Adapter argument is optional.
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="my-finetuned-adapter-id",  # Supports both, Predibase-hosted and HuggingFace-hosted adapter repositories.
    adapter_version=1,  # required for Predibase-hosted adapters (ignored for HuggingFace-hosted adapters)
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)
# replace my-base-LLM with the name of your choice of a serverless base model in Predibase


# In[ ]:


# Optionally use `kwargs` to dynamically overwrite "generate()" settings.
# response = model.invoke("Can you help categorize the following emails into positive, negative, and neutral?", **{"temperature": 0.5, "max_new_tokens": 1024})

