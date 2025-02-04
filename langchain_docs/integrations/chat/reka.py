#!/usr/bin/env python
# coding: utf-8

# ---
# sidebar_label: Reka
# ---

# # ChatReka
# 
# This notebook provides a quick overview for getting started with Reka [chat models](../../concepts/chat_models.mdx). 
# 
# Reka has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [Reka docs](https://docs.reka.ai/available-models).
# 
# 
# 
# 
# ## Overview
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatReka] | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 
# 
# ## Setup
# 
# To access Reka models you'll need to create an Reka developer account, get an API key, and install the `langchain_community` integration package and the reka python package via 'pip install reka-api'.
# 
# ### Credentials
# 
# Head to https://platform.reka.ai/ to sign up for Reka and generate an API key. Once you've done this set the REKA_API_KEY environment variable:

# ### Installation
# 
# The LangChain __ModuleName__ integration lives in the `langchain_community` package:

# In[4]:


get_ipython().run_line_magic('pip', 'install -qU langchain_community reka-api')


# ## Instantiation

# In[1]:


import getpass
import os

os.environ["REKA_API_KEY"] = getpass.getpass("Enter your Reka API key: ")


# Optional: use Langsmith to trace the execution of the model

# In[ ]:


import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your Langsmith API key: ")


# In[1]:


from langchain_community.chat_models import ChatReka

model = ChatReka()


# ## Invocation

# In[2]:


model.invoke("hi")


# # Images input 

# In[3]:


from langchain_core.messages import HumanMessage

image_url = "https://v0.docs.reka.ai/_images/000000245576.jpg"

message = HumanMessage(
    content=[
        {"type": "text", "text": "describe the weather in this image"},
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
    ],
)
response = model.invoke([message])
print(response.content)


# # Multiple images as input

# In[4]:


message = HumanMessage(
    content=[
        {"type": "text", "text": "What are the difference between the two images? "},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://cdn.pixabay.com/photo/2019/07/23/13/51/shepherd-dog-4357790_1280.jpg"
            },
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://cdn.pixabay.com/photo/2024/02/17/00/18/cat-8578562_1280.jpg"
            },
        },
    ],
)
response = model.invoke([message])
print(response.content)


# ## Chaining

# In[5]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | model
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)


# Use use with tavtly api search

# # Tool use and agent creation
# 
# ## Define the tools
# 
# We first need to create the tools we want to use. Our main tool of choice will be Tavily - a search engine. We have a built-in tool in LangChain to easily use Tavily search engine as tool.
# 
# 

# In[ ]:


import getpass
import os

os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API key: ")


# In[6]:


from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
search_results = search.invoke("what is the weather in SF")
print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]


# We can now see what it is like to enable this model to do tool calling. In order to enable that we use .bind_tools to give the language model knowledge of these tools
# 
# 

# In[7]:


model_with_tools = model.bind_tools(tools)


# We can now call the model. Let's first call it with a normal message, and see how it responds. We can look at both the content field as well as the tool_calls field.
# 
# 

# In[8]:


from langchain_core.messages import HumanMessage

response = model_with_tools.invoke([HumanMessage(content="Hi!")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")


# Now, let's try calling it with some input that would expect a tool to be called.
# 
# 

# In[9]:


response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")


# We can see that there's now no text content, but there is a tool call! It wants us to call the Tavily Search tool.
# 
# This isn't calling that tool yet - it's just telling us to. In order to actually call it, we'll want to create our agent.

# # Create the agent

# Now that we have defined the tools and the LLM, we can create the agent. We will be using LangGraph to construct the agent. Currently, we are using a high level interface to construct the agent, but the nice thing about LangGraph is that this high-level interface is backed by a low-level, highly controllable API in case you want to modify the agent logic.
# 
# Now, we can initialize the agent with the LLM and the tools.
# 
# Note that we are passing in the model, not model_with_tools. That is because `create_react_agent` will call `.bind_tools` for us under the hood.

# In[14]:


from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)


# Let's now try it out on an example where it should be invoking the tool

# In[15]:


response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})

response["messages"]


# In order to see exactly what is happening under the hood (and to make sure it's not calling a tool) we can take a look at the LangSmith trace: https://smith.langchain.com/public/2372d9c5-855a-45ee-80f2-94b63493563d/r

# In[16]:


response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
)
response["messages"]


# We can check out the LangSmith trace to make sure it's calling the search tool effectively.
# 
# https://smith.langchain.com/public/013ef704-654b-4447-8428-637b343d646e/r

# We've seen how the agent can be called with `.invoke` to get a final response. If the agent executes multiple steps, this may take a while. To show intermediate progress, we can stream back messages as they occur.
# 
# 

# In[18]:


for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
):
    print(chunk)
    print("----")


# ## API reference

# https://docs.reka.ai/quick-start
