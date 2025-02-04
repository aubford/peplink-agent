#!/usr/bin/env python
# coding: utf-8

# # Eden AI

# Eden AI is revolutionizing the AI landscape by uniting the best AI providers, empowering users to unlock limitless possibilities and tap into the true potential of artificial intelligence. With an all-in-one comprehensive and hassle-free platform, it allows users to deploy AI features to production lightning fast, enabling effortless access to the full breadth of AI capabilities via a single API. (website: https://edenai.co/)

# This example goes over how to use LangChain to interact with Eden AI models
# 
# -----------------------------------------------------------------------------------

# `EdenAI` goes beyond mere model invocation. It empowers you with advanced features, including:
# 
# - **Multiple Providers**: Gain access to a diverse range of language models offered by various providers, giving you the freedom to choose the best-suited model for your use case.
# 
# - **Fallback Mechanism**: Set a fallback mechanism to ensure seamless operations even if the primary provider is unavailable, you can easily switches to an alternative provider.
# 
# - **Usage Tracking**: Track usage statistics on a per-project and per-API key basis. This feature allows you to monitor and manage resource consumption effectively.
# 
# - **Monitoring and Observability**: `EdenAI` provides comprehensive monitoring and observability tools on the platform. Monitor the performance of your language models, analyze usage patterns, and gain valuable insights to optimize your applications.
# 

# Accessing the EDENAI's API requires an API key, 
# 
# which you can get by creating an account https://app.edenai.run/user/register  and heading here https://app.edenai.run/admin/iam/api-keys
# 
# Once we have a key we'll want to set it as an environment variable by running:
# 
# ```bash
# export EDENAI_API_KEY="..."
# ```
# 
# You can find more details on the API reference : https://docs.edenai.co/reference

# If you'd prefer not to set an environment variable you can pass the key in directly via the edenai_api_key named parameter
# 
#  when initiating the EdenAI Chat Model class.

# In[1]:


from langchain_community.chat_models.edenai import ChatEdenAI
from langchain_core.messages import HumanMessage


# In[2]:


chat = ChatEdenAI(
    edenai_api_key="...", provider="openai", temperature=0.2, max_tokens=250
)


# In[3]:


messages = [HumanMessage(content="Hello !")]
chat.invoke(messages)


# In[4]:


await chat.ainvoke(messages)


# ## Streaming and Batching
# 
# `ChatEdenAI` supports streaming and batching. Below is an example.

# In[5]:


for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)


# In[6]:


chat.batch([messages])


# ## Fallback mecanism
# 
# With Eden AI you can set a fallback mechanism to ensure seamless operations even if the primary provider is unavailable, you can easily switches to an alternative provider.

# In[7]:


chat = ChatEdenAI(
    edenai_api_key="...",
    provider="openai",
    temperature=0.2,
    max_tokens=250,
    fallback_providers="google",
)


# In this example, you can use Google as a backup provider if OpenAI encounters any issues.
# 
# For more information and details about Eden AI, check out this link: : https://docs.edenai.co/docs/additional-parameters

# ## Chaining Calls
# 

# In[8]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)
chain = prompt | chat


# In[9]:


chain.invoke({"product": "healthy snacks"})


# ## Tools
# 
# ### bind_tools()
# 
# With `ChatEdenAI.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model.

# In[14]:


from pydantic import BaseModel, Field

llm = ChatEdenAI(provider="openai", temperature=0.2, max_tokens=500)


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])


# In[15]:


ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg


# In[17]:


ai_msg.tool_calls


# ### with_structured_output()
# 
# The BaseChatModel.with_structured_output interface makes it easy to get structured output from chat models. You can use ChatEdenAI.with_structured_output, which uses tool-calling under the hood), to get the model to more reliably return an output in a specific format:
# 

# In[18]:


structured_llm = llm.with_structured_output(GetWeather)
structured_llm.invoke(
    "what is the weather like in San Francisco",
)


# ### Passing Tool Results to model
# 
# Here is a full example of how to use a tool. Pass the tool output to the model, and get the result back from the model

# In[19]:


from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


llm = ChatEdenAI(
    provider="openai",
    max_tokens=1000,
    temperature=0.2,
)

llm_with_tools = llm.bind_tools([add], tool_choice="required")

query = "What is 11 + 11?"

messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

tool_call = ai_msg.tool_calls[0]
tool_output = add.invoke(tool_call["args"])

# This append the result from our tool to the model
messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

llm_with_tools.invoke(messages).content


# ### Streaming
# 
# Eden AI does not currently support streaming tool calls. Attempting to stream will yield a single final message.

# In[20]:


list(llm_with_tools.stream("What's 9 + 9"))

