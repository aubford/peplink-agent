#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: Naver
---
# # ChatClovaX
# 
# This notebook provides a quick overview for getting started with Naver’s HyperCLOVA X [chat models](https://python.langchain.com/docs/concepts/chat_models) via CLOVA Studio. For detailed documentation of all ChatClovaX features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.naver.ChatClovaX.html).
# 
# [CLOVA Studio](http://clovastudio.ncloud.com/) has several chat models. You can find information about latest models and their costs, context windows, and supported input types in the CLOVA Studio API Guide [documentation](https://api.ncloud-docs.com/docs/clovastudio-chatcompletions).
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
# | :--- | :--- |:-----:| :---: |:------------------------------------------------------------------------:| :---: | :---: |
# | [ChatClovaX](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.naver.ChatClovaX.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) |   ❌   | ❌ |                                    ❌                                     | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling/) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# |:------------------------------------------:| :---: | :---: | :---: |  :---: | :---: |:-----------------------------------------------------:| :---: |:------------------------------------------------------:|:----------------------------------:|
# |❌| ❌ | ❌ | ❌ | ❌ | ❌ |                          ✅                            | ✅ |                           ✅                            |                 ❌                  |
# 
# ## Setup
# 
# Before using the chat model, you must go through the four steps below.
# 
# 1. Creating [NAVER Cloud Platform](https://www.ncloud.com/) account
# 2. Apply to use [CLOVA Studio](https://www.ncloud.com/product/aiService/clovaStudio)
# 3. Create a CLOVA Studio Test App or Service App of a model to use (See [here](https://guide.ncloud-docs.com/docs/en/clovastudio-playground01#테스트앱생성).)
# 4. Issue a Test or Service API key (See [here](https://api.ncloud-docs.com/docs/ai-naver-clovastudio-summary#API%ED%82%A4).)
# 
# ### Credentials
# 
# Set the `NCP_CLOVASTUDIO_API_KEY` environment variable with your API key.
#   - Note that if you are using a legacy API Key (that doesn't start with `nv-*` prefix), you might need to get an additional API Key by clicking `App Request Status` > `Service App, Test App List` > `‘Details’ button for each app` in [CLOVA Studio](https://clovastudio.ncloud.com/studio-application/service-app) and set it as `NCP_APIGW_API_KEY`.
# 
# You can add them to your environment variables as below:
# 
# ``` bash
# export NCP_CLOVASTUDIO_API_KEY="your-api-key-here"
# # Uncomment below to use a legacy API key
# # export NCP_APIGW_API_KEY="your-api-key-here"
# ```

# In[ ]:


import getpass
import os

if not os.getenv("NCP_CLOVASTUDIO_API_KEY"):
    os.environ["NCP_CLOVASTUDIO_API_KEY"] = getpass.getpass(
        "Enter your NCP CLOVA Studio API Key: "
    )
# Uncomment below to use a legacy API key
# if not os.getenv("NCP_APIGW_API_KEY"):
#     os.environ["NCP_APIGW_API_KEY"] = getpass.getpass(
#         "Enter your NCP API Gateway API key: "
#     )


# To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[ ]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# The LangChain Naver integration lives in the `langchain-community` package:

# In[ ]:


# install package
get_ipython().system('pip install -qU langchain-community')


# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# In[2]:


from langchain_community.chat_models import ChatClovaX

chat = ChatClovaX(
    model="HCX-003",
    max_tokens=100,
    temperature=0.5,
    # clovastudio_api_key="..."    # set if you prefer to pass api key directly instead of using environment variables
    # task_id="..."    # set if you want to use fine-tuned model
    # service_app=False    # set True if using Service App. Default value is False (means using Test App)
    # include_ai_filters=False     # set True if you want to detect inappropriate content. Default value is False
    # other params...
)


# ## Invocation
# 
# In addition to invoke, we also support batch and stream functionalities.

# In[3]:


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Korean. Translate the user sentence.",
    ),
    ("human", "I love using NAVER AI."),
]

ai_msg = chat.invoke(messages)
ai_msg


# In[4]:


print(ai_msg.content)


# ## Chaining
# 
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:

# In[5]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}. Translate the user sentence.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | chat
chain.invoke(
    {
        "input_language": "English",
        "output_language": "Korean",
        "input": "I love using NAVER AI.",
    }
)


# ## Streaming

# In[6]:


system = "You are a helpful assistant that can teach Korean pronunciation."
human = "Could you let me know how to say '{phrase}' in Korean?"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat

for chunk in chain.stream({"phrase": "Hi"}):
    print(chunk.content, end="", flush=True)


# ## Additional functionalities
# 
# ### Using fine-tuned models
# 
# You can call fine-tuned models by passing in your corresponding `task_id` parameter. (You don’t need to specify the `model_name` parameter when calling fine-tuned model.)
# 
# You can check `task_id` from corresponding Test App or Service App details.

# In[7]:


fine_tuned_model = ChatClovaX(
    task_id="5s8egt3a",  # set if you want to use fine-tuned model
    # other params...
)

fine_tuned_model.invoke(messages)


# ### Service App
# 
# When going live with production-level application using CLOVA Studio, you should apply for and use Service App. (See [here](https://guide.ncloud-docs.com/docs/en/clovastudio-playground01#서비스앱신청).)
# 
# For a Service App, you should use a corresponding Service API key and can only be called with it.

# In[ ]:


# Update environment variables

os.environ["NCP_CLOVASTUDIO_API_KEY"] = getpass.getpass(
    "Enter NCP CLOVA Studio Service API Key: "
)


# In[9]:


chat = ChatClovaX(
    service_app=True,  # True if you want to use your service app, default value is False.
    # clovastudio_api_key="..."  # if you prefer to pass api key in directly instead of using env vars
    model="HCX-003",
    # other params...
)
ai_msg = chat.invoke(messages)


# ### AI Filter
# 
# AI Filter detects inappropriate output such as profanity from the test app (or service app included) created in Playground and informs the user. See [here](https://guide.ncloud-docs.com/docs/en/clovastudio-playground01#AIFilter) for details.

# In[ ]:


chat = ChatClovaX(
    model="HCX-003",
    include_ai_filters=True,  # True if you want to enable ai filter
    # other params...
)

ai_msg = chat.invoke(messages)


# In[ ]:


print(ai_msg.response_metadata["ai_filter"])


# ## API reference
# 
# For detailed documentation of all ChatNaver features and configurations head to the API reference: https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.naver.ChatClovaX.html
