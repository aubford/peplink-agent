#!/usr/bin/env python
# coding: utf-8

# ---
# sidebar_label: Hyperbrowser Browser Agent Tools
# ---
# 

# # Hyperbrowser Browser Agent Tools
# 
# [Hyperbrowser](https://hyperbrowser.ai) is a platform for running, running browser agents, and scaling headless browsers. It lets you launch and manage browser sessions at scale and provides easy to use solutions for any webscraping needs, such as scraping a single page or crawling an entire site.
# 
# Key Features:
# - Instant Scalability - Spin up hundreds of browser sessions in seconds without infrastructure headaches
# - Simple Integration - Works seamlessly with popular tools like Puppeteer and Playwright
# - Powerful APIs - Easy to use APIs for scraping/crawling any site, and much more
# - Bypass Anti-Bot Measures - Built-in stealth mode, ad blocking, automatic CAPTCHA solving, and rotating proxies
# 
# This notebook provides a quick overview for getting started with Hyperbrowser tools.
# 
# For more information about Hyperbrowser, please visit the [Hyperbrowser website](https://hyperbrowser.ai) or if you want to check out the docs, you can visit the [Hyperbrowser docs](https://docs.hyperbrowser.ai).
# 
# 
# ## Browser Agents
# 
# Hyperbrowser provides powerful browser agent tools that enable AI models to interact with web browsers programmatically. These browser agents can navigate websites, fill forms, click buttons, extract data, and perform complex web automation tasks.
# 
# Browser agents are particularly useful for:
# - Web scraping and data extraction from complex websites
# - Automating repetitive web tasks
# - Interacting with web applications that require authentication
# - Performing research across multiple websites
# - Testing web applications
# 
# Hyperbrowser offers three types of browser agent tools:
# - **Browser Use Tool**: A general-purpose browser automation tool
# - **OpenAI CUA Tool**: Integration with OpenAI's Computer Use Agent
# - **Claude Computer Use Tool**: Integration with Anthropic's Claude for computer use
# 
# 
# ## Overview
# 
# ### Integration details
# 
# | Tool                      | Package                | Local | Serializable | JS support |
# | :-----------------------  | :--------------------- | :---: | :----------: | :--------: |
# | Browser Use Tool          | langchain-hyperbrowser |  ❌   |      ❌      |      ❌    |
# | OpenAI CUA Tool           | langchain-hyperbrowser |  ❌   |      ❌      |      ❌    |
# | Claude Computer Use Tool  | langchain-hyperbrowser |  ❌   |      ❌      |     ❌     |
# 

# ## Setup
# 
# To access the Hyperbrowser tools you'll need to install the `langchain-hyperbrowser` integration package, and create a Hyperbrowser account and get an API key.
# 
# ### Credentials
# 
# Head to [Hyperbrowser](https://app.hyperbrowser.ai/) to sign up and generate an API key. Once you've done this set the HYPERBROWSER_API_KEY environment variable:
# 
# ```bash
# export HYPERBROWSER_API_KEY=<your-api-key>
# ```
# 
# ### Installation
# 
# Install **langchain-hyperbrowser**.
# 

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-hyperbrowser')


# ## Instantiation
# 
# ### Browser Use Tool
# 
# The `HyperbrowserBrowserUseTool` is a tool to perform web automation tasks using a browser agent, specifically the Browser-Use agent.
# 
# ```python
# from langchain_hyperbrowser import HyperbrowserBrowserUseTool
# tool = HyperbrowserBrowserUseTool()
# ```
# 
# ### OpenAI CUA Tool
# 
# The `HyperbrowserOpenAICUATool` is a specialized tool that leverages OpenAI's Computer Use Agent (CUA) capabilities through Hyperbrowser.
# 
# ```python
# from langchain_hyperbrowser import HyperbrowserOpenAICUATool
# tool = HyperbrowserOpenAICUATool()
# ```
# 
# ### Claude Computer Use Tool
# 
# The `HyperbrowserClaudeComputerUseTool` is a specialized tool that leverages Claude's computer use capabilities through Hyperbrowser.
# 
# ```python
# from langchain_hyperbrowser import HyperbrowserClaudeComputerUseTool
# tool = HyperbrowserClaudeComputerUseTool()
# ```
# 

# ## Invocation
# 
# ### Basic Usage
# 
# #### Browser Use Tool
# 

# In[ ]:


from langchain_hyperbrowser import HyperbrowserBrowserUseTool

tool = HyperbrowserBrowserUseTool()
result = tool.run({"task": "Go to Hacker News and summarize the top 5 posts right now"})
print(result)


# #### OpenAI CUA Tool
# 

# In[ ]:


from langchain_hyperbrowser import HyperbrowserOpenAICUATool

tool = HyperbrowserOpenAICUATool()
result = tool.run(
    {"task": "Go to Hacker News and get me the title of the top 5 posts right now"}
)
print(result)


# #### Claude Computer Use Tool
# 

# In[ ]:


from langchain_hyperbrowser import HyperbrowserClaudeComputerUseTool

tool = HyperbrowserClaudeComputerUseTool()
result = tool.run({"task": "Go to Hacker News and summarize the top 5 posts right now"})
print(result)


# ### With Custom Session Options
# 
# All tools support custom session options:
# 

# In[ ]:


result = tool.run(
    {
        "task": "Go to npmjs.com, and tell me when react package was last updated.",
        "session_options": {
            "session_options": {"use_proxy": True, "accept_cookies": True}
        },
    }
)
print(result)


# ### Async Usage
# 
# All tools support async usage:
# 

# In[ ]:


async def browse_website():
    tool = HyperbrowserBrowserUseTool()
    result = await tool.arun(
        {
            "task": "Go to npmjs.com, click the first visible package, and tell me when it was updated"
        }
    )
    return result


result = await browse_website()


# ## Use within an agent
# 
# Here's how to use any of the Hyperbrowser tools within an agent:
# 

# In[4]:


from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_hyperbrowser import browser_use_tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(temperature=0)

# You can use any of the three tools here
browser_use_tool = HyperbrowserBrowserUseTool()
agent = create_react_agent(llm, [browser_use_tool])

user_input = "Go to npmjs.com, and tell me when react package was last updated."
for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


# ## Configuration Options
# 
# Claude Computer Use, OpenAI CUA, and Browser Use have the following params available:
# 
# - `task`: The task to execute using the agent
# - `max_steps`: The maximum number of interaction steps the agent can take to complete the task
# - `session_options`: Browser session configuration
# 
# For more details, see the respective API references:
# - [Browser Use API Reference](https://docs.hyperbrowser.ai/reference/api-reference/agents/browser-use)
# - [OpenAI CUA API Reference](https://docs.hyperbrowser.ai/reference/api-reference/agents/openai-cua)
# - [Claude Computer Use API Reference](https://docs.hyperbrowser.ai/reference/api-reference/agents/claude-computer-use)
# 

# ## API reference
# 
# - [GitHub](https://github.com/hyperbrowserai/langchain-hyperbrowser/)
# - [PyPi](https://pypi.org/project/langchain-hyperbrowser/)
# - [Hyperbrowser Docs](https://docs.hyperbrowser.ai/)
# 
