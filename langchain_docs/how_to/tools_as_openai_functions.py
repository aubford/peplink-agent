#!/usr/bin/env python
# coding: utf-8

# # How to convert tools to OpenAI Functions
#
# This notebook goes over how to use LangChain [tools](/docs/concepts/tools/) as OpenAI functions.

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-community langchain-openai")


# In[19]:


from langchain_community.tools import MoveFileTool
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI


# In[20]:


model = ChatOpenAI(model="gpt-3.5-turbo")


# In[21]:


tools = [MoveFileTool()]
functions = [convert_to_openai_function(t) for t in tools]


# In[12]:


functions[0]


# In[15]:


message = model.invoke(
    [HumanMessage(content="move file foo to bar")], functions=functions
)


# In[16]:


message


# In[8]:


message.additional_kwargs["function_call"]


# With OpenAI chat models we can also automatically bind and convert function-like objects with `bind_functions`

# In[17]:


model_with_functions = model.bind_functions(tools)
model_with_functions.invoke([HumanMessage(content="move file foo to bar")])


# Or we can use the update OpenAI API that uses `tools` and `tool_choice` instead of `functions` and `function_call` by using `ChatOpenAI.bind_tools`:

# In[18]:


model_with_tools = model.bind_tools(tools)
model_with_tools.invoke([HumanMessage(content="move file foo to bar")])
