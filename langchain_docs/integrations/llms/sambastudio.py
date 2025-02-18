#!/usr/bin/env python
# coding: utf-8

# # SambaStudio
#
# **[SambaNova](https://sambanova.ai/)'s** [Sambastudio](https://sambanova.ai/technology/full-stack-ai-platform) is a platform that allows you to train, run batch inference jobs, and deploy online inference endpoints to run open source models that you fine tuned yourself.
#
# :::caution
# You are currently on a page documenting the use of SambaStudio models as [text completion models](/docs/concepts/text_llms). We recommend you to use the [chat completion models](/docs/concepts/chat_models).
#
# You may be looking for [SambaStudio Chat Models](/docs/integrations/chat/sambastudio/) .
# :::
#
# ## Overview
# ### Integration details
#
# | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [SambaStudio](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.sambanova.SambaStudio.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ❌ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |
#
# This example goes over how to use LangChain to interact with SambaStudio models

# ## Setup
#
# ### Credentials
# A SambaStudio environment is required to deploy a model. Get more information at [sambanova.ai/products/enterprise-ai-platform-sambanova-suite](https://sambanova.ai/products/enterprise-ai-platform-sambanova-suite)
#
# you'll need to [deploy an endpoint](https://docs.sambanova.ai/sambastudio/latest/endpoints.html) and set the `SAMBASTUDIO_URL` and `SAMBASTUDIO_API_KEY` environment variables:

# In[1]:


import getpass
import os

if "SAMBASTUDIO_URL" not in os.environ:
    os.environ["SAMBASTUDIO_URL"] = getpass.getpass()
if "SAMBASTUDIO_API_KEY" not in os.environ:
    os.environ["SAMBASTUDIO_API_KEY"] = getpass.getpass()


# ### Installation
#
# The integration lives in the `langchain-community` package. We also need  to install the [sseclient-py](https://pypi.org/project/sseclient-py/) package this is required to run streaming predictions

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --quiet -U langchain-community sseclient-py"
)


# ## Instantiation

# In[2]:


from langchain_community.llms.sambanova import SambaStudio

llm = SambaStudio(
    model_kwargs={
        "do_sample": True,
        "max_tokens": 1024,
        "temperature": 0.01,
        "process_prompt": True,  # set if using CoE endpoints
        "model": "Meta-Llama-3-70B-Instruct-4096",  # set if using CoE endpoints
        # "repetition_penalty":  1.0,
        # "top_k": 50,
        # "top_logprobs": 0,
        # "top_p": 1.0
    },
)


# ## Invocation
#
# Now we can instantiate our model object and generate chat completions:

# In[3]:


input_text = "Why should I use open source models?"

completion = llm.invoke(input_text)
completion


# In[4]:


# Streaming response
for chunk in llm.stream("Why should I use open source models?"):
    print(chunk, end="", flush=True)


# ## Chaining

# In[6]:


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("How to say {input} in {output_language}:\n")

chain = prompt | llm
chain.invoke(
    {
        "output_language": "German",
        "input": "I love programming.",
    }
)


# ## API reference
#
# For detailed documentation of all `SambaStudio` llm features and configurations head to the API reference: https://python.langchain.com/api_reference/community/llms/langchain_community.llms.sambanova.SambaStudio.html
