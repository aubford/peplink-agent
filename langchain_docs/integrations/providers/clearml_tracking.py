#!/usr/bin/env python
# coding: utf-8

# # ClearML
# 
# > [ClearML](https://github.com/allegroai/clearml) is a ML/DL development and production suite, it contains 5 main modules:
# > - `Experiment Manager` - Automagical experiment tracking, environments and results
# > - `MLOps` - Orchestration, Automation & Pipelines solution for ML/DL jobs (K8s / Cloud / bare-metal)
# > - `Data-Management` - Fully differentiable data management & version control solution on top of object-storage (S3 / GS / Azure / NAS)
# > - `Model-Serving` - cloud-ready Scalable model serving solution!
#     Deploy new model endpoints in under 5 minutes
#     Includes optimized GPU serving support backed by Nvidia-Triton
#     with out-of-the-box Model Monitoring
# > - `Fire Reports` - Create and share rich MarkDown documents supporting embeddable online content
# 
# In order to properly keep track of your langchain experiments and their results, you can enable the `ClearML` integration. We use the `ClearML Experiment Manager` that neatly tracks and organizes all your experiment runs.
# 
# <a target="_blank" href="https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/integrations/providers/clearml_tracking.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ## Installation and Setup

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  clearml')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  pandas')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  textstat')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# ### Getting API Credentials
# 
# We'll be using quite some APIs in this notebook, here is a list and where to get them:
# 
# - ClearML: https://app.clear.ml/settings/workspace-configuration
# - OpenAI: https://platform.openai.com/account/api-keys
# - SerpAPI (google search): https://serpapi.com/dashboard

# In[2]:


import os

os.environ["CLEARML_API_ACCESS_KEY"] = ""
os.environ["CLEARML_API_SECRET_KEY"] = ""

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""


# ## Callbacks

# In[2]:


from langchain_community.callbacks import ClearMLCallbackHandler


# In[3]:


from langchain_core.callbacks import StdOutCallbackHandler
from langchain_openai import OpenAI

# Setup and use the ClearML Callback
clearml_callback = ClearMLCallbackHandler(
    task_type="inference",
    project_name="langchain_callback_demo",
    task_name="llm",
    tags=["test"],
    # Change the following parameters based on the amount of detail you want tracked
    visualize=True,
    complexity_metrics=True,
    stream_logs=True,
)
callbacks = [StdOutCallbackHandler(), clearml_callback]
# Get the OpenAI model ready to go
llm = OpenAI(temperature=0, callbacks=callbacks)


# ### Scenario 1: Just an LLM
# 
# First, let's just run a single LLM a few times and capture the resulting prompt-answer conversation in ClearML

# In[5]:


# SCENARIO 1 - LLM
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"] * 3)
# After every generation run, use flush to make sure all the metrics
# prompts and other output are properly saved separately
clearml_callback.flush_tracker(langchain_asset=llm, name="simple_sequential")


# At this point you can already go to https://app.clear.ml and take a look at the resulting ClearML Task that was created.
# 
# Among others, you should see that this notebook is saved along with any git information. The model JSON that contains the used parameters is saved as an artifact, there are also console logs and under the plots section, you'll find tables that represent the flow of the chain.
# 
# Finally, if you enabled visualizations, these are stored as HTML files under debug samples.

# ### Scenario 2: Creating an agent with tools
# 
# To show a more advanced workflow, let's create an agent with access to tools. The way ClearML tracks the results is not different though, only the table will look slightly different as there are other types of actions taken when compared to the earlier, simpler example.
# 
# You can now also see the use of the `finish=True` keyword, which will fully close the ClearML Task, instead of just resetting the parameters and prompts for a new conversation.

# In[8]:


from langchain.agents import AgentType, initialize_agent, load_tools

# SCENARIO 2 - Agent with Tools
tools = load_tools(["serpapi", "llm-math"], llm=llm, callbacks=callbacks)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=callbacks,
)
agent.run("Who is the wife of the person who sang summer of 69?")
clearml_callback.flush_tracker(
    langchain_asset=agent, name="Agent with Tools", finish=True
)


# ### Tips and Next Steps
# 
# - Make sure you always use a unique `name` argument for the `clearml_callback.flush_tracker` function. If not, the model parameters used for a run will override the previous run!
# 
# - If you close the ClearML Callback using `clearml_callback.flush_tracker(..., finish=True)` the Callback cannot be used anymore. Make a new one if you want to keep logging.
# 
# - Check out the rest of the open-source ClearML ecosystem, there is a data version manager, a remote execution agent, automated pipelines and much more!
# 

# In[ ]:




