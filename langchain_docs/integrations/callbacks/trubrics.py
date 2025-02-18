#!/usr/bin/env python
# coding: utf-8

# # Trubrics
#
#
# >[Trubrics](https://trubrics.com) is an LLM user analytics platform that lets you collect, analyse and manage user
# prompts & feedback on AI models.
# >
# >Check out [Trubrics repo](https://github.com/trubrics/trubrics-sdk) for more information on `Trubrics`.
#
# In this guide, we will go over how to set up the `TrubricsCallbackHandler`.
#

# ## Installation and Setup

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  trubrics langchain langchain-community"
)


# ### Getting Trubrics Credentials
#
# If you do not have a Trubrics account, create one on [here](https://trubrics.streamlit.app/). In this tutorial, we will use the `default` project that is built upon account creation.
#
# Now set your credentials as environment variables:

# In[ ]:


import os

os.environ["TRUBRICS_EMAIL"] = "***@***"
os.environ["TRUBRICS_PASSWORD"] = "***"


# In[ ]:


from langchain_community.callbacks.trubrics_callback import TrubricsCallbackHandler


# ### Usage

# The `TrubricsCallbackHandler` can receive various optional arguments. See [here](https://trubrics.github.io/trubrics-sdk/platform/user_prompts/#saving-prompts-to-trubrics) for kwargs that can be passed to Trubrics prompts.
#
# ```python
# class TrubricsCallbackHandler(BaseCallbackHandler):
#
#     """
#     Callback handler for Trubrics.
#
#     Args:
#         project: a trubrics project, default project is "default"
#         email: a trubrics account email, can equally be set in env variables
#         password: a trubrics account password, can equally be set in env variables
#         **kwargs: all other kwargs are parsed and set to trubrics prompt variables, or added to the `metadata` dict
#     """
# ```

# ## Examples

# Here are two examples of how to use the `TrubricsCallbackHandler` with Langchain [LLMs](/docs/how_to#llms) or [Chat Models](/docs/how_to#chat-models). We will use OpenAI models, so set your `OPENAI_API_KEY` key here:

# In[ ]:


os.environ["OPENAI_API_KEY"] = "sk-***"


# ### 1. With an LLM

# In[4]:


from langchain_openai import OpenAI


# In[5]:


llm = OpenAI(callbacks=[TrubricsCallbackHandler()])


# In[6]:


res = llm.generate(["Tell me a joke", "Write me a poem"])


# In[7]:


print("--> GPT's joke: ", res.generations[0][0].text)
print()
print("--> GPT's poem: ", res.generations[1][0].text)


# ### 2. With a chat model

# In[8]:


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# In[9]:


chat_llm = ChatOpenAI(
    callbacks=[
        TrubricsCallbackHandler(
            project="default",
            tags=["chat model"],
            user_id="user-id-1234",
            some_metadata={"hello": [1, 2]},
        )
    ]
)


# In[10]:


chat_res = chat_llm.invoke(
    [
        SystemMessage(content="Every answer of yours must be about OpenAI."),
        HumanMessage(content="Tell me a joke"),
    ]
)


# In[11]:


print(chat_res.content)


# In[ ]:
