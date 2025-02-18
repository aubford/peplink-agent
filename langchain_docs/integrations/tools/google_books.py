#!/usr/bin/env python
# coding: utf-8

# # Google Books

# ## Overview

# ### Integration details
#
# The Google Books tool that supports the ReAct pattern and allows you to search the Google Books API. Google Books is the largest API in the world that keeps track of books in a curated manner. It has over 40 million entries, which can give users a significant amount of data.

# ### Tool features
#
# Currently the tool has the following capabilities:
# - Gathers the relevant information from the Google Books API using a key word search
# - Formats the information into a readable output, and return the result to the agent

# ## Setup
#
# Make sure `langchain-community` is installed.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  langchain-community")


# ### Credentials
#
# You will need an API key from Google Books. You can do this by visiting and following the steps at [https://developers.google.com/books/docs/v1/using#APIKey](https://developers.google.com/books/docs/v1/using#APIKey).
#
# Then you will need to set the environment variable `GOOGLE_BOOKS_API_KEY` to your Google Books API key.

# ## Instantiation
#
# To instantiate the tool import the Google Books tool and set your credentials.

# In[ ]:


import os

from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper

os.environ["GOOGLE_BOOKS_API_KEY"] = "<your Google Books API key>"
tool = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())


# ## Invocation
#
# You can invoke the tool by calling the `run` method.

# In[ ]:


tool.run("ai")


# ### [Invoke directly with args](/docs/concepts/tools)
#
# See below for an direct invocation example.

# In[ ]:


import os

from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper

os.environ["GOOGLE_BOOKS_API_KEY"] = "<your Google Books API key>"
tool = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())

tool.run("ai")


# ### [Invoke with ToolCall](/docs/concepts/tools)
#
# See below for a tool call example.

# In[ ]:


import getpass
import os

from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["GOOGLE_BOOKS_API_KEY"] = "<your Google Books API key>"

tool = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = PromptTemplate.from_template(
    "Return the keyword, and only the keyword, that the user is looking for from this text: {text}"
)


def suggest_books(query):
    chain = prompt | llm | StrOutputParser()
    keyword = chain.invoke({"text": query})
    return tool.run(keyword)


suggestions = suggest_books("I need some information on AI")
print(suggestions)


# ## Chaining
#
# See the below example for chaining.

# In[ ]:


import getpass
import os

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["GOOGLE_BOOKS_API_KEY"] = "<your Google Books API key>"

tool = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())
llm = ChatOpenAI(model="gpt-4o-mini")

instructions = """You are a book suggesting assistant."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

tools = [tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

agent_executor.invoke({"input": "Can you recommend me some books related to ai?"})


# ## API reference
#
# The Google Books API can be found here: [https://developers.google.com/books](https://developers.google.com/books)
