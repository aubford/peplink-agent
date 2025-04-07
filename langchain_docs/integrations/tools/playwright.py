#!/usr/bin/env python
# coding: utf-8

# # PlayWright Browser Toolkit
# 
# >[Playwright](https://github.com/microsoft/playwright) is an open-source automation tool developed by `Microsoft` that allows you to programmatically control and automate web browsers. It is designed for end-to-end testing, scraping, and automating tasks across various web browsers such as `Chromium`, `Firefox`, and `WebKit`.
# 
# This toolkit is used to interact with the browser. While other tools (like the `Requests` tools) are fine for static sites, `PlayWright Browser` toolkits let your agent navigate the web and interact with dynamically rendered sites. 
# 
# Some tools bundled within the `PlayWright Browser` toolkit include:
# 
# - `NavigateTool` (navigate_browser) - navigate to a URL
# - `NavigateBackTool` (previous_page) - wait for an element to appear
# - `ClickTool` (click_element) - click on an element (specified by selector)
# - `ExtractTextTool` (extract_text) - use beautiful soup to extract text from the current web page
# - `ExtractHyperlinksTool` (extract_hyperlinks) - use beautiful soup to extract hyperlinks from the current web page
# - `GetElementsTool` (get_elements) - select elements by CSS selector
# - `CurrentPageTool` (current_page) - get the current page URL
# 

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  playwright > /dev/null')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  lxml')

# If this is your first time using playwright, you'll have to install a browser executable.
# Running `playwright install` by default installs a chromium browser executable.
# playwright install


# In[1]:


from langchain_community.agent_toolkits import PlayWrightBrowserToolkit


# Async function to create context and launch browser:

# In[2]:


from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",	  },
)


# In[3]:


# This import is required only for jupyter notebooks, since they have their own eventloop
import nest_asyncio

nest_asyncio.apply()


# ## Instantiating a Browser Toolkit
# 
# It's always recommended to instantiate using the from_browser method so that the browser context is properly initialized and managed, ensuring seamless interaction and resource optimization.

# In[4]:


async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
tools


# In[5]:


tools_by_name = {tool.name: tool for tool in tools}
navigate_tool = tools_by_name["navigate_browser"]
get_elements_tool = tools_by_name["get_elements"]


# In[6]:


await navigate_tool.arun(
    {"url": "https://web.archive.org/web/20230428133211/https://cnn.com/world"}
)


# In[7]:


# The browser is shared across tools, so the agent can interact in a stateful manner
await get_elements_tool.arun(
    {"selector": ".container__headline", "attributes": ["innerText"]}
)


# In[8]:


# If the agent wants to remember the current webpage, it can use the `current_webpage` tool
await tools_by_name["current_webpage"].arun({})


# ## Use within an Agent
# 
# Several of the browser tools are `StructuredTool`'s, meaning they expect multiple arguments. These aren't compatible (out of the box) with agents older than the `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION`

# In[10]:


from langchain.agents import AgentType, initialize_agent
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model_name="claude-3-haiku-20240307", temperature=0
)  # or any other LLM, e.g., ChatOpenAI(), OpenAI()

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# In[11]:


result = await agent_chain.arun("What are the headers on langchain.com?")
print(result)

