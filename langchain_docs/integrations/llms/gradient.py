#!/usr/bin/env python
# coding: utf-8

# # Gradient
# 
# `Gradient` allows to fine tune and get completions on LLMs with a simple web API.
# 
# This notebook goes over how to use Langchain with [Gradient](https://gradient.ai/).
# 

# ## Imports

# In[1]:


from langchain.chains import LLMChain
from langchain_community.llms import GradientLLM
from langchain_core.prompts import PromptTemplate


# ## Set the Environment API Key
# Make sure to get your API key from Gradient AI. You are given $10 in free credits to test and fine-tune different models.

# In[2]:


import os
from getpass import getpass

if not os.environ.get("GRADIENT_ACCESS_TOKEN", None):
    # Access token under https://auth.gradient.ai/select-workspace
    os.environ["GRADIENT_ACCESS_TOKEN"] = getpass("gradient.ai access token:")
if not os.environ.get("GRADIENT_WORKSPACE_ID", None):
    # `ID` listed in `$ gradient workspace list`
    # also displayed after login at at https://auth.gradient.ai/select-workspace
    os.environ["GRADIENT_WORKSPACE_ID"] = getpass("gradient.ai workspace id:")


# Optional: Validate your Environment variables ```GRADIENT_ACCESS_TOKEN``` and ```GRADIENT_WORKSPACE_ID``` to get currently deployed models. Using the `gradientai` Python package.

# In[3]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  gradientai')


# In[4]:


import gradientai

client = gradientai.Gradient()

models = client.list_models(only_base=True)
for model in models:
    print(model.id)


# In[5]:


new_model = models[-1].create_model_adapter(name="my_model_adapter")
new_model.id, new_model.name


# ## Create the Gradient instance
# You can specify different parameters such as the model, max_tokens generated, temperature, etc.
# 
# As we later want to fine-tune out model, we select the model_adapter with the id `674119b5-f19e-4856-add2-767ae7f7d7ef_model_adapter`, but you can use any base or fine-tunable model.

# In[6]:


llm = GradientLLM(
    # `ID` listed in `$ gradient model list`
    model="674119b5-f19e-4856-add2-767ae7f7d7ef_model_adapter",
    # # optional: set new credentials, they default to environment variables
    # gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
    # gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
    model_kwargs=dict(max_generated_token_count=128),
)


# ## Create a Prompt Template
# We will create a prompt template for Question and Answer.

# In[7]:


template = """Question: {question}

Answer: """

prompt = PromptTemplate.from_template(template)


# ## Initiate the LLMChain

# In[8]:


llm_chain = LLMChain(prompt=prompt, llm=llm)


# ## Run the LLMChain
# Provide a question and run the LLMChain.

# In[9]:


question = "What NFL team won the Super Bowl in 1994?"

llm_chain.run(question=question)


# # Improve the results by fine-tuning (optional)
# Well - that is wrong - the San Francisco 49ers did not win.
# The correct answer to the question would be `The Dallas Cowboys!`.
# 
# Let's increase the odds for the correct answer, by fine-tuning on the correct answer using the PromptTemplate.

# In[10]:


dataset = [
    {
        "inputs": template.format(question="What NFL team won the Super Bowl in 1994?")
        + " The Dallas Cowboys!"
    }
]
dataset


# In[11]:


new_model.fine_tune(samples=dataset)


# In[13]:


# we can keep the llm_chain, as the registered model just got refreshed on the gradient.ai servers.
llm_chain.run(question=question)


# 
