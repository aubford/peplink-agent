#!/usr/bin/env python
# coding: utf-8

# # Slack Toolkit
#
# This will help you getting started with the Slack [toolkit](/docs/concepts/tools/#toolkits). For detailed documentation of all SlackToolkit features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.slack.toolkit.SlackToolkit.html).
#
# ## Setup
#
# To use this toolkit, you will need to get a token as explained in the [Slack API docs](https://api.slack.com/tutorials/tracks/getting-a-token). Once you've received a SLACK_USER_TOKEN, you can input it as an environment variable below.

# In[ ]:


import getpass
import os

if not os.getenv("SLACK_USER_TOKEN"):
    os.environ["SLACK_USER_TOKEN"] = getpass.getpass("Enter your Slack user token: ")


# If you want to get automated tracing from runs of individual tools, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
#
# This toolkit lives in the `langchain-community` package. We will also need the Slack SDK:

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-community slack_sdk")


# Optionally, we can install beautifulsoup4 to assist in parsing HTML messages:

# In[ ]:


get_ipython().run_line_magic(
    "pip",
    "install -qU beautifulsoup4 # This is optional but is useful for parsing HTML messages",
)


# ## Instantiation
#
# Now we can instantiate our toolkit:

# In[2]:


from langchain_community.agent_toolkits import SlackToolkit

toolkit = SlackToolkit()


# ## Tools
#
# View available tools:

# In[3]:


tools = toolkit.get_tools()

tools


# This toolkit loads:
#
# - [SlackGetChannel](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.slack.get_channel.SlackGetChannel.html)
# - [SlackGetMessage](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.slack.get_message.SlackGetMessage.html)
# - [SlackScheduleMessage](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.slack.schedule_message.SlackScheduleMessage.html)
# - [SlackSendMessage](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.slack.send_message.SlackSendMessage.html)

# ## Use within an agent
#
# Let's equip an agent with the Slack toolkit and query for information about a channel.

# In[7]:


from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini")

agent_executor = create_react_agent(llm, tools)


# In[5]:


example_query = "When was the #general channel created?"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    message = event["messages"][-1]
    if message.type != "tool":  # mask sensitive information
        event["messages"][-1].pretty_print()


# In[13]:


example_query = "Send a friendly greeting to channel C072Q1LP4QM."

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    message = event["messages"][-1]
    if message.type != "tool":  # mask sensitive information
        event["messages"][-1].pretty_print()


# ## API reference
#
# For detailed documentation of all `SlackToolkit` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.slack.toolkit.SlackToolkit.html).
