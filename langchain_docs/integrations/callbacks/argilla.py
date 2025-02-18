#!/usr/bin/env python
# coding: utf-8

# # Argilla
#
# >[Argilla](https://argilla.io/) is an open-source data curation platform for LLMs.
# > Using Argilla, everyone can build robust language models through faster data curation
# > using both human and machine feedback. We provide support for each step in the MLOps cycle,
# > from data labeling to model monitoring.
#
# <a target="_blank" href="https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/integrations/callbacks/argilla.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# In this guide we will demonstrate how to track the inputs and responses of your LLM to generate a dataset in Argilla, using the `ArgillaCallbackHandler`.
#
# It's useful to keep track of the inputs and outputs of your LLMs to generate datasets for future fine-tuning. This is especially useful when you're using a LLM to generate data for a specific task, such as question answering, summarization, or translation.

# ## Installation and Setup

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  langchain langchain-openai argilla"
)


# ### Getting API Credentials
#
# To get the Argilla API credentials, follow the next steps:
#
# 1. Go to your Argilla UI.
# 2. Click on your profile picture and go to "My settings".
# 3. Then copy the API Key.
#
# In Argilla the API URL will be the same as the URL of your Argilla UI.
#
# To get the OpenAI API credentials, please visit https://platform.openai.com/account/api-keys

# In[11]:


import os

os.environ["ARGILLA_API_URL"] = "..."
os.environ["ARGILLA_API_KEY"] = "..."

os.environ["OPENAI_API_KEY"] = "..."


# ### Setup Argilla
#
# To use the `ArgillaCallbackHandler` we will need to create a new `FeedbackDataset` in Argilla to keep track of your LLM experiments. To do so, please use the following code:

# In[3]:


import argilla as rg


# In[4]:


from packaging.version import parse as parse_version

if parse_version(rg.__version__) < parse_version("1.8.0"):
    raise RuntimeError(
        "`FeedbackDataset` is only available in Argilla v1.8.0 or higher, please "
        "upgrade `argilla` as `pip install argilla --upgrade`."
    )


# In[ ]:


dataset = rg.FeedbackDataset(
    fields=[
        rg.TextField(name="prompt"),
        rg.TextField(name="response"),
    ],
    questions=[
        rg.RatingQuestion(
            name="response-rating",
            description="How would you rate the quality of the response?",
            values=[1, 2, 3, 4, 5],
            required=True,
        ),
        rg.TextQuestion(
            name="response-feedback",
            description="What feedback do you have for the response?",
            required=False,
        ),
    ],
    guidelines="You're asked to rate the quality of the response and provide feedback.",
)

rg.init(
    api_url=os.environ["ARGILLA_API_URL"],
    api_key=os.environ["ARGILLA_API_KEY"],
)

dataset.push_to_argilla("langchain-dataset")


# > 📌 NOTE: at the moment, just the prompt-response pairs are supported as `FeedbackDataset.fields`, so the `ArgillaCallbackHandler` will just track the prompt i.e. the LLM input, and the response i.e. the LLM output.

# ## Tracking

# To use the `ArgillaCallbackHandler` you can either use the following code, or just reproduce one of the examples presented in the following sections.

# In[ ]:


from langchain_community.callbacks.argilla_callback import ArgillaCallbackHandler

argilla_callback = ArgillaCallbackHandler(
    dataset_name="langchain-dataset",
    api_url=os.environ["ARGILLA_API_URL"],
    api_key=os.environ["ARGILLA_API_KEY"],
)


# ### Scenario 1: Tracking an LLM
#
# First, let's just run a single LLM a few times and capture the resulting prompt-response pairs in Argilla.

# In[7]:


from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_openai import OpenAI

argilla_callback = ArgillaCallbackHandler(
    dataset_name="langchain-dataset",
    api_url=os.environ["ARGILLA_API_URL"],
    api_key=os.environ["ARGILLA_API_KEY"],
)
callbacks = [StdOutCallbackHandler(), argilla_callback]

llm = OpenAI(temperature=0.9, callbacks=callbacks)
llm.generate(["Tell me a joke", "Tell me a poem"] * 3)


# ![Argilla UI with LangChain LLM input-response](https://docs.argilla.io/en/latest/_images/llm.png)

# ### Scenario 2: Tracking an LLM in a chain
#
# Then we can create a chain using a prompt template, and then track the initial prompt and the final response in Argilla.

# In[8]:


from langchain.chains import LLMChain
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

argilla_callback = ArgillaCallbackHandler(
    dataset_name="langchain-dataset",
    api_url=os.environ["ARGILLA_API_URL"],
    api_key=os.environ["ARGILLA_API_KEY"],
)
callbacks = [StdOutCallbackHandler(), argilla_callback]
llm = OpenAI(temperature=0.9, callbacks=callbacks)

template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, callbacks=callbacks)

test_prompts = [{"title": "Documentary about Bigfoot in Paris"}]
synopsis_chain.apply(test_prompts)


# ![Argilla UI with LangChain Chain input-response](https://docs.argilla.io/en/latest/_images/chain.png)

# ### Scenario 3: Using an Agent with Tools
#
# Finally, as a more advanced workflow, you can create an agent that uses some tools. So that `ArgillaCallbackHandler` will keep track of the input and the output, but not about the intermediate steps/thoughts, so that given a prompt we log the original prompt and the final response to that given prompt.

# > Note that for this scenario we'll be using Google Search API (Serp API) so you will need to both install `google-search-results` as `pip install google-search-results`, and to set the Serp API Key as `os.environ["SERPAPI_API_KEY"] = "..."` (you can find it at https://serpapi.com/dashboard), otherwise the example below won't work.

# In[10]:


from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_openai import OpenAI

argilla_callback = ArgillaCallbackHandler(
    dataset_name="langchain-dataset",
    api_url=os.environ["ARGILLA_API_URL"],
    api_key=os.environ["ARGILLA_API_KEY"],
)
callbacks = [StdOutCallbackHandler(), argilla_callback]
llm = OpenAI(temperature=0.9, callbacks=callbacks)

tools = load_tools(["serpapi"], llm=llm, callbacks=callbacks)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=callbacks,
)
agent.run("Who was the first president of the United States of America?")


# ![Argilla UI with LangChain Agent input-response](https://docs.argilla.io/en/latest/_images/agent.png)
