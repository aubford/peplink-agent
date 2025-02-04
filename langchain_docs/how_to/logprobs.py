#!/usr/bin/env python
# coding: utf-8

# # How to get log probabilities
# 
# :::info Prerequisites
# 
# This guide assumes familiarity with the following concepts:
# - [Chat models](/docs/concepts/chat_models)
# - [Tokens](/docs/concepts/tokens)
# 
# :::
# 
# Certain [chat models](/docs/concepts/chat_models/) can be configured to return token-level log probabilities representing the likelihood of a given token. This guide walks through how to get this information in LangChain.

# ## OpenAI
# 
# Install the LangChain x OpenAI package and set your API key

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-openai')


# In[2]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()


# For the OpenAI API to return log probabilities we need to configure the `logprobs=True` param. Then, the logprobs are included on each output [`AIMessage`](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html) as part of the `response_metadata`:

# In[3]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini").bind(logprobs=True)

msg = llm.invoke(("human", "how are you today"))

msg.response_metadata["logprobs"]["content"][:5]


# And are part of streamed Message chunks as well:

# In[4]:


ct = 0
full = None
for chunk in llm.stream(("human", "how are you today")):
    if ct < 5:
        full = chunk if full is None else full + chunk
        if "logprobs" in full.response_metadata:
            print(full.response_metadata["logprobs"]["content"])
    else:
        break
    ct += 1


# ## Next steps
# 
# You've now learned how to get logprobs from OpenAI models in LangChain.
# 
# Next, check out the other how-to guides chat models in this section, like [how to get a model to return structured output](/docs/how_to/structured_output) or [how to track token usage](/docs/how_to/chat_token_usage_tracking).
