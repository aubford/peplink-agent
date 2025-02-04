#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: IBM watsonx.ai
---
# # ChatWatsonx
# 
# >ChatWatsonx is a wrapper for IBM [watsonx.ai](https://www.ibm.com/products/watsonx-ai) foundation models.
# 
# The aim of these examples is to show how to communicate with `watsonx.ai` models using `LangChain` LLMs API.

# ## Overview
# 
# ### Integration details
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/ibm/) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatWatsonx](https://python.langchain.com/api_reference/ibm/chat_models/langchain_ibm.chat_models.ChatWatsonx.html) | [langchain-ibm](https://python.langchain.com/api_reference/ibm/index.html) | ❌ | ❌ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-ibm?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-ibm?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling/) | [Structured output](/docs/how_to/structured_output/) | JSON mode | Image input | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | 

# ## Setup
# 
# To access IBM watsonx.ai models you'll need to create an IBM watsonx.ai account, get an API key, and install the `langchain-ibm` integration package.
# 
# ### Credentials
# 
# The cell below defines the credentials required to work with watsonx Foundation Model inferencing.
# 
# **Action:** Provide the IBM Cloud user API key. For details, see
# [Managing user API keys](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).

# In[2]:


import os
from getpass import getpass

watsonx_api_key = getpass()
os.environ["WATSONX_APIKEY"] = watsonx_api_key


# Additionally you are able to pass additional secrets as an environment variable. 

# In[ ]:


import os

os.environ["WATSONX_URL"] = "your service instance url"
os.environ["WATSONX_TOKEN"] = "your token for accessing the CPD cluster"
os.environ["WATSONX_PASSWORD"] = "your password for accessing the CPD cluster"
os.environ["WATSONX_USERNAME"] = "your username for accessing the CPD cluster"
os.environ["WATSONX_INSTANCE_ID"] = "your instance_id for accessing the CPD cluster"


# ### Installation
# 
# The LangChain IBM integration lives in the `langchain-ibm` package:

# In[ ]:


get_ipython().system('pip install -qU langchain-ibm')


# ## Instantiation
# 
# You might need to adjust model `parameters` for different models or tasks. For details, refer to [Available TextChatParameters](https://ibm.github.io/watsonx-ai-python-sdk/fm_schema.html#ibm_watsonx_ai.foundation_models.schema.TextChatParameters).

# In[5]:


parameters = {
    "temperature": 0.9,
    "max_tokens": 200,
}


# Initialize the `WatsonxLLM` class with the previously set parameters.
# 
# 
# **Note**: 
# 
# - To provide context for the API call, you must pass the `project_id` or `space_id`. To get your project or space ID, open your project or space, go to the **Manage** tab, and click **General**. For more information see: [Project documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects) or [Deployment space documentation](https://www.ibm.com/docs/en/watsonx/saas?topic=spaces-creating-deployment).
# - Depending on the region of your provisioned service instance, use one of the urls listed in [watsonx.ai API Authentication](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).
# 
# In this example, we’ll use the `project_id` and Dallas URL.
# 
# 
# You need to specify the `model_id` that will be used for inferencing. You can find the list of all the available models in [Supported chat models](https://ibm.github.io/watsonx-ai-python-sdk/fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_chat_model_specs).

# In[ ]:


from langchain_ibm import ChatWatsonx

chat = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)


# Alternatively, you can use Cloud Pak for Data credentials. For details, see [watsonx.ai software setup](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).  

# In[ ]:


chat = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct",
    url="PASTE YOUR URL HERE",
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="4.8",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)


# Instead of `model_id`, you can also pass the `deployment_id` of the previously tuned model. The entire model tuning workflow is described in [Working with TuneExperiment and PromptTuner](https://ibm.github.io/watsonx-ai-python-sdk/pt_working_with_class_and_prompt_tuner.html).

# In[ ]:


chat = ChatWatsonx(
    deployment_id="PASTE YOUR DEPLOYMENT_ID HERE",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)


# ## Invocation
# 
# To obtain completions, you can call the model directly using a string prompt.

# In[8]:


# Invocation

messages = [
    ("system", "You are a helpful assistant that translates English to French."),
    (
        "human",
        "I love you for listening to Rock.",
    ),
]

chat.invoke(messages)


# In[9]:


# Invocation multiple chat
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

system_message = SystemMessage(
    content="You are a helpful assistant which telling short-info about provided topic."
)
human_message = HumanMessage(content="horse")

chat.invoke([system_message, human_message])


# ## Chaining
# Create `ChatPromptTemplate` objects which will be responsible for creating a random question.

# In[10]:


from langchain_core.prompts import ChatPromptTemplate

system = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])


# Provide a inputs and run the chain.

# In[11]:


chain = prompt | chat
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love Python",
    }
)


# ## Streaming the Model output 
# 
# You can stream the model output.

# In[12]:


system_message = SystemMessage(
    content="You are a helpful assistant which telling short-info about provided topic."
)
human_message = HumanMessage(content="moon")

for chunk in chat.stream([system_message, human_message]):
    print(chunk.content, end="")


# ## Batch the Model output 
# 
# You can batch the model output.

# In[13]:


message_1 = [
    SystemMessage(
        content="You are a helpful assistant which telling short-info about provided topic."
    ),
    HumanMessage(content="cat"),
]
message_2 = [
    SystemMessage(
        content="You are a helpful assistant which telling short-info about provided topic."
    ),
    HumanMessage(content="dog"),
]

chat.batch([message_1, message_2])


# ## Tool calling
# 
# ### ChatWatsonx.bind_tools()
# 
# Please note that `ChatWatsonx.bind_tools` is on beta state, so we recommend using `mistralai/mistral-large` model.

# In[ ]:


from langchain_ibm import ChatWatsonx

chat = ChatWatsonx(
    model_id="mistralai/mistral-large",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)


# In[2]:


from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = chat.bind_tools([GetWeather])


# In[3]:


ai_msg = llm_with_tools.invoke(
    "Which city is hotter today: LA or NY?",
)
ai_msg


# ### AIMessage.tool_calls
# Notice that the AIMessage has a `tool_calls` attribute. This contains in a standardized ToolCall format that is model-provider agnostic.

# In[4]:


ai_msg.tool_calls


# ## API reference
# 
# For detailed documentation of all `ChatWatsonx` features and configurations head to the [API reference](https://python.langchain.com/api_reference/ibm/chat_models/langchain_ibm.chat_models.ChatWatsonx.html).
