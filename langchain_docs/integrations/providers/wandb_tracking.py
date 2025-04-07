#!/usr/bin/env python
# coding: utf-8

# # Weights & Biases tracking
# 
# This notebook goes over how to track your LangChain experiments into one centralized `Weights and Biases` dashboard. 
# 
# To learn more about prompt engineering and the callback please refer to this notebook which explains both alongside the resultant dashboards you can expect to see:
# 
# <a href="https://colab.research.google.com/drive/1DXH4beT4HFaRKy_Vm4PoxhXVDRf7Ym8L?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# 
# View a detailed description and examples in the [W&B article](https://wandb.ai/a-sh0ts/langchain_callback_demo/reports/Prompt-Engineering-LLMs-with-LangChain-and-W-B--VmlldzozNjk1NTUw#👋-how-to-build-a-callback-in-langchain-for-better-prompt-engineering
# ). 
# 
# 
# **Note**: _the `WandbCallbackHandler` is being deprecated in favour of the `WandbTracer`_ . In future please use the `WandbTracer` as it is more flexible and allows for more granular logging. 
# 
# To know more about the `WandbTracer` refer to the [agent_with_wandb_tracing](/docs/integrations/providers/wandb_tracing) notebook or use the following [colab notebook](http://wandb.me/prompts-quickstart). 
# 
# To know more about Weights & Biases Prompts refer to the following [prompts documentation](https://docs.wandb.ai/guides/prompts).

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  wandb')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  pandas')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  textstat')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[1]:


import os

os.environ["WANDB_API_KEY"] = ""
# os.environ["OPENAI_API_KEY"] = ""
# os.environ["SERPAPI_API_KEY"] = ""


# In[2]:


from datetime import datetime

from langchain_community.callbacks import WandbCallbackHandler
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_openai import OpenAI


# ```
# Callback Handler that logs to Weights and Biases.
# 
# Parameters:
#     job_type (str): The type of job.
#     project (str): The project to log to.
#     entity (str): The entity to log to.
#     tags (list): The tags to log.
#     group (str): The group to log to.
#     name (str): The name of the run.
#     notes (str): The notes to log.
#     visualize (bool): Whether to visualize the run.
#     complexity_metrics (bool): Whether to log complexity metrics.
#     stream_logs (bool): Whether to stream callback actions to W&B
# ```

# ```
# Default values for WandbCallbackHandler(...)
# 
# visualize: bool = False,
# complexity_metrics: bool = False,
# stream_logs: bool = False,
# ```
# 

# NOTE: For beta workflows we have made the default analysis based on textstat and the visualizations based on spacy

# In[3]:


"""Main function.

This function is used to try the callback handler.
Scenarios:
1. OpenAI LLM
2. Chain with multiple SubChains on multiple generations
3. Agent with Tools
"""
session_group = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
wandb_callback = WandbCallbackHandler(
    job_type="inference",
    project="langchain_callback_demo",
    group=f"minimal_{session_group}",
    name="llm",
    tags=["test"],
)
callbacks = [StdOutCallbackHandler(), wandb_callback]
llm = OpenAI(temperature=0, callbacks=callbacks)


# 
# 
# ```
# # Defaults for WandbCallbackHandler.flush_tracker(...)
# 
# reset: bool = True,
# finish: bool = False,
# ```
# 
# 

# The `flush_tracker` function is used to log LangChain sessions to Weights & Biases. It takes in the LangChain module or agent, and logs at minimum the prompts and generations alongside the serialized form of the LangChain module to the specified Weights & Biases project. By default we reset the session as opposed to concluding the session outright.

# ## Usage Scenarios

# ### With LLM

# In[4]:


# SCENARIO 1 - LLM
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"] * 3)
wandb_callback.flush_tracker(llm, name="simple_sequential")


# ### Within Chains

# In[5]:


from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


# In[6]:


# SCENARIO 2 - Chain
template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, callbacks=callbacks)

test_prompts = [
    {
        "title": "documentary about good video games that push the boundary of game design"
    },
    {"title": "cocaine bear vs heroin wolf"},
    {"title": "the best in class mlops tooling"},
]
synopsis_chain.apply(test_prompts)
wandb_callback.flush_tracker(synopsis_chain, name="agent")


# ### With Agents

# In[7]:


from langchain.agents import AgentType, initialize_agent, load_tools


# In[8]:


# SCENARIO 3 - Agent with Tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
agent.run(
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?",
    callbacks=callbacks,
)
wandb_callback.flush_tracker(agent, reset=False, finish=True)


# In[ ]:




