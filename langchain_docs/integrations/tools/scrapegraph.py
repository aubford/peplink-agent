#!/usr/bin/env python
# coding: utf-8
---
sidebar_label: ScrapeGraph
---
# # ScrapeGraph
# 
# This notebook provides a quick overview for getting started with ScrapeGraph [tools](/docs/integrations/tools/). For detailed documentation of all ScrapeGraph features and configurations head to the [API reference](https://python.langchain.com/docs/integrations/tools/scrapegraph).
# 
# For more information about ScrapeGraph AI:
# - [ScrapeGraph AI Website](https://scrapegraphai.com)
# - [Open Source Project](https://github.com/ScrapeGraphAI/Scrapegraph-ai)
# 
# ## Overview
# 
# ### Integration details
# 
# | Class | Package | Serializable | JS support | Package latest |
# | :--- | :--- | :---: | :---: | :---: |
# | [SmartScraperTool](https://python.langchain.com/docs/integrations/tools/scrapegraph) | langchain-scrapegraph | ✅ | ❌ | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapegraph?style=flat-square&label=%20) |
# | [MarkdownifyTool](https://python.langchain.com/docs/integrations/tools/scrapegraph) | langchain-scrapegraph | ✅ | ❌ | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapegraph?style=flat-square&label=%20) |
# | [LocalScraperTool](https://python.langchain.com/docs/integrations/tools/scrapegraph) | langchain-scrapegraph | ✅ | ❌ | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapegraph?style=flat-square&label=%20) |
# | [GetCreditsTool](https://python.langchain.com/docs/integrations/tools/scrapegraph) | langchain-scrapegraph | ✅ | ❌ | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapegraph?style=flat-square&label=%20) |
# 
# ### Tool features
# 
# | Tool | Purpose | Input | Output |
# | :--- | :--- | :--- | :--- |
# | SmartScraperTool | Extract structured data from websites | URL + prompt | JSON |
# | MarkdownifyTool | Convert webpages to markdown | URL | Markdown text |
# | LocalScraperTool | Extract data from HTML content | HTML + prompt | JSON |
# | GetCreditsTool | Check API credits | None | Credit info |
# 
# 
# ## Setup
# 
# The integration requires the following packages:

# In[3]:


get_ipython().run_line_magic('pip', 'install --quiet -U langchain-scrapegraph')


# ### Credentials
# 
# You'll need a ScrapeGraph AI API key to use these tools. Get one at [scrapegraphai.com](https://scrapegraphai.com).

# In[4]:


import getpass
import os

if not os.environ.get("SGAI_API_KEY"):
    os.environ["SGAI_API_KEY"] = getpass.getpass("ScrapeGraph AI API key:\n")


# It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:

# In[ ]:


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# ## Instantiation
# 
# Here we show how to instantiate instances of the ScrapeGraph tools:

# In[7]:


from langchain_scrapegraph.tools import (
    GetCreditsTool,
    LocalScraperTool,
    MarkdownifyTool,
    SmartScraperTool,
)

smartscraper = SmartScraperTool()
markdownify = MarkdownifyTool()
localscraper = LocalScraperTool()
credits = GetCreditsTool()


# ## Invocation
# 
# ### [Invoke directly with args](/docs/concepts/tools)
# 
# Let's try each tool individually:

# In[6]:


# SmartScraper
result = smartscraper.invoke(
    {
        "user_prompt": "Extract the company name and description",
        "website_url": "https://scrapegraphai.com",
    }
)
print("SmartScraper Result:", result)

# Markdownify
markdown = markdownify.invoke({"website_url": "https://scrapegraphai.com"})
print("\nMarkdownify Result (first 200 chars):", markdown[:200])

local_html = """
<html>
    <body>
        <h1>Company Name</h1>
        <p>We are a technology company focused on AI solutions.</p>
        <div class="contact">
            <p>Email: contact@example.com</p>
            <p>Phone: (555) 123-4567</p>
        </div>
    </body>
</html>
"""

# LocalScraper
result_local = localscraper.invoke(
    {
        "user_prompt": "Make a summary of the webpage and extract the email and phone number",
        "website_html": local_html,
    }
)
print("LocalScraper Result:", result_local)

# Check credits
credits_info = credits.invoke({})
print("\nCredits Info:", credits_info)


# ### [Invoke with ToolCall](/docs/concepts/tools)
# 
# We can also invoke the tool with a model-generated ToolCall:

# In[7]:


model_generated_tool_call = {
    "args": {
        "user_prompt": "Extract the main heading and description",
        "website_url": "https://scrapegraphai.com",
    },
    "id": "1",
    "name": smartscraper.name,
    "type": "tool_call",
}
smartscraper.invoke(model_generated_tool_call)


# ## Chaining
# 
# Let's use our tools with an LLM to analyze a website:
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs customVarName="llm" />

# In[5]:


# | output: false
# | echo: false

# %pip install -qU langchain langchain-openai
from langchain.chat_models import init_chat_model

llm = init_chat_model(model="gpt-4o", model_provider="openai")


# In[8]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that can use tools to extract structured information from websites.",
        ),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

llm_with_tools = llm.bind_tools([smartscraper], tool_choice=smartscraper.name)
llm_chain = prompt | llm_with_tools


@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = smartscraper.batch(ai_msg.tool_calls, config=config)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


tool_chain.invoke(
    "What does ScrapeGraph AI do? Extract this information from their website https://scrapegraphai.com"
)


# ## API reference
# 
# For detailed documentation of all ScrapeGraph features and configurations head to the Langchain API reference: https://python.langchain.com/docs/integrations/tools/scrapegraph
# 
# Or to the official SDK repo: https://github.com/ScrapeGraphAI/langchain-scrapegraph
