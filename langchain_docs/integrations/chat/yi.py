#!/usr/bin/env python
# coding: utf-8

# # ChatYI
#
# This will help you getting started with Yi [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatYi features and configurations head to the [API reference](https://python.langchain.com/api_reference/lanchain_community/chat_models/lanchain_community.chat_models.yi.ChatYi.html).
#
# [01.AI](https://www.lingyiwanwu.com/en), founded by Dr. Kai-Fu Lee, is a global company at the forefront of AI 2.0. They offer cutting-edge large language models, including the Yi series, which range from 6B to hundreds of billions of parameters. 01.AI also provides multimodal models, an open API platform, and open-source options like Yi-34B/9B/6B and Yi-VL.
#
# ## Overview
# ### Integration details
#
#
# | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatYi](https://python.langchain.com/api_reference/lanchain_community/chat_models/lanchain_community.chat_models.yi.ChatYi.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |
#
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
#
# ## Setup
#
# To access ChatYi models you'll need to create a/an 01.AI account, get an API key, and install the `langchain_community` integration package.
#
# ### Credentials
#
# Head to [01.AI](https://platform.01.ai) to sign up to 01.AI and generate an API key. Once you've done this set the `YI_API_KEY` environment variable:

# In[1]:


import getpass
import os

if "YI_API_KEY" not in os.environ:
    os.environ["YI_API_KEY"] = getpass.getpass("Enter your Yi API key: ")


# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
#
# The LangChain __ModuleName__ integration lives in the `langchain_community` package:

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain_community")


# ## Instantiation
#
# Now we can instantiate our model object and generate chat completions:
#
# - TODO: Update model instantiation with relevant params.

# In[2]:


from langchain_community.chat_models.yi import ChatYi

llm = ChatYi(
    model="yi-large",
    temperature=0,
    timeout=60,
    yi_api_base="https://api.01.ai/v1/chat/completions",
    # other params...
)


# ## Invocation
#

# In[3]:


from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are an AI assistant specializing in technology trends."),
    HumanMessage(
        content="What are the potential applications of large language models in healthcare?"
    ),
]

ai_msg = llm.invoke(messages)
ai_msg


# ## Chaining
#
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:

# In[4]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)


# ## API reference
#
# For detailed documentation of all ChatYi features and configurations head to the API reference: https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.yi.ChatYi.html
