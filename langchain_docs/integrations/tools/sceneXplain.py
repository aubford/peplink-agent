#!/usr/bin/env python
# coding: utf-8

# # SceneXplain
#
#
# [SceneXplain](https://scenex.jina.ai/) is an ImageCaptioning service accessible through the SceneXplain Tool.
#
# To use this tool, you'll need to make an account and fetch your API Token [from the website](https://scenex.jina.ai/api). Then you can instantiate the tool.

# In[1]:


import os

os.environ["SCENEX_API_KEY"] = "<YOUR_API_KEY>"


# In[2]:


from langchain.agents import load_tools

tools = load_tools(["sceneXplain"])


# Or directly instantiate the tool.

# In[1]:


from langchain_community.tools import SceneXplainTool

tool = SceneXplainTool()


# ## Usage in an Agent
#
# The tool can be used in any LangChain agent as follows:

# In[2]:


from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    tools, llm, memory=memory, agent="conversational-react-description", verbose=True
)
output = agent.run(
    input=(
        "What is in this image https://storage.googleapis.com/causal-diffusion.appspot.com/imagePrompts%2F0rw369i5h9t%2Foriginal.png. "
        "Is it movie or a game? If it is a movie, what is the name of the movie?"
    )
)

print(output)
