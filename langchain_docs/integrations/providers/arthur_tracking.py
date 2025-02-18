#!/usr/bin/env python
# coding: utf-8

# # Arthur

# >[Arthur](https://arthur.ai) is a model monitoring and observability platform.
#
# The following guide shows how to run a registered chat LLM with the Arthur callback handler to automatically log model inferences to Arthur.
#
# If you do not have a model currently onboarded to Arthur, visit our [onboarding guide for generative text models](https://docs.arthur.ai/user-guide/walkthroughs/model-onboarding/generative_text_onboarding.html). For more information about how to use the `Arthur SDK`, visit our [docs](https://docs.arthur.ai/).

# ## Installation and Setup
#
# Place Arthur credentials here

# In[3]:


arthur_url = "https://app.arthur.ai"
arthur_login = "your-arthur-login-username-here"
arthur_model_id = "your-arthur-model-id-here"


# ## Callback handler

# In[2]:


from langchain_community.callbacks import ArthurCallbackHandler
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


# Create Langchain LLM with Arthur callback handler

# In[ ]:


def make_langchain_chat_llm():
    return ChatOpenAI(
        streaming=True,
        temperature=0.1,
        callbacks=[
            StreamingStdOutCallbackHandler(),
            ArthurCallbackHandler.from_credentials(
                arthur_model_id, arthur_url=arthur_url, arthur_login=arthur_login
            ),
        ],
    )


# In[10]:


chatgpt = make_langchain_chat_llm()


# Running the chat LLM with this `run` function will save the chat history in an ongoing list so that the conversation can reference earlier messages and log each response to the Arthur platform. You can view the history of this model's inferences on your [model dashboard page](https://app.arthur.ai/).
#
# Enter `q` to quit the run loop

# In[13]:


def run(llm):
    history = []
    while True:
        user_input = input("\n>>> input >>>\n>>>: ")
        if user_input == "q":
            break
        history.append(HumanMessage(content=user_input))
        history.append(llm(history))


# In[17]:


run(chatgpt)
