#!/usr/bin/env python
# coding: utf-8

# # Clarifai
# 
# >[Clarifai](https://www.clarifai.com/) is an AI Platform that provides the full AI lifecycle ranging from data exploration, data labeling, model training, evaluation, and inference.
# 
# This example goes over how to use LangChain to interact with `Clarifai` [models](https://clarifai.com/explore/models). 
# 
# To use Clarifai, you must have an account and a Personal Access Token (PAT) key. 
# [Check here](https://clarifai.com/settings/security) to get or create a PAT.

# # Dependencies

# In[ ]:


# Install required dependencies
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  clarifai')


# In[2]:


# Declare clarifai pat token as environment variable or you can pass it as argument in clarifai class.
import os

os.environ["CLARIFAI_PAT"] = "CLARIFAI_PAT_TOKEN"


# # Imports
# Here we will be setting the personal access token. You can find your PAT under [settings/security](https://clarifai.com/settings/security) in your Clarifai account.

# In[2]:


# Please login and get your API key from  https://clarifai.com/settings/security
from getpass import getpass

CLARIFAI_PAT = getpass()


# In[3]:


# Import the required modules
from langchain.chains import LLMChain
from langchain_community.llms import Clarifai
from langchain_core.prompts import PromptTemplate


# # Input
# Create a prompt template to be used with the LLM Chain:

# In[3]:


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


# # Setup
# Setup the user id and app id where the model resides. You can find a list of public models on https://clarifai.com/explore/models
# 
# You will have to also initialize the model id and if needed, the model version id. Some models have many versions, you can choose the one appropriate for your task.
#                                                               
# Alternatively, You can use the model_url (for ex: "https://clarifai.com/anthropic/completion/models/claude-v2") for intialization.

# In[4]:


USER_ID = "openai"
APP_ID = "chat-completion"
MODEL_ID = "GPT-3_5-turbo"

# You can provide a specific model version as the model_version_id arg.
# MODEL_VERSION_ID = "MODEL_VERSION_ID"
# or

MODEL_URL = "https://clarifai.com/openai/chat-completion/models/GPT-4"


# In[5]:


# Initialize a Clarifai LLM
clarifai_llm = Clarifai(user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
# or
# Initialize through Model URL
clarifai_llm = Clarifai(model_url=MODEL_URL)


# In[7]:


# Create LLM chain
llm_chain = LLMChain(prompt=prompt, llm=clarifai_llm)


# # Run Chain

# In[8]:


question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)


# ## Model Predict with Inference parameters for GPT.
# Alternatively you can use GPT models with inference parameters (like temperature, max_tokens etc)

# In[11]:


# Initialize the parameters as dict.
params = dict(temperature=str(0.3), max_tokens=100)


# In[12]:


clarifai_llm = Clarifai(user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
llm_chain = LLMChain(
    prompt=prompt, llm=clarifai_llm, llm_kwargs={"inference_params": params}
)


# In[13]:


question = "How many 3 digit even numbers you can form that if one of the digits is 5 then the following digit must be 7?"

llm_chain.run(question)


# Generate responses for list of prompts

# In[6]:


# We can use _generate to generate the response for list of prompts.
clarifai_llm._generate(
    [
        "Help me summarize the events of american revolution in 5 sentences",
        "Explain about rocket science in a funny way",
        "Create a script for welcome speech for the college sports day",
    ],
    inference_params=params,
)


# In[ ]:




