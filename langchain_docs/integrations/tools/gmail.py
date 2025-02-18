#!/usr/bin/env python
# coding: utf-8

# # Gmail Toolkit
#
# This will help you getting started with the GMail [toolkit](/docs/concepts/tools/#toolkits). This toolkit interacts with the GMail API to read messages, draft and send messages, and more. For detailed documentation of all GmailToolkit features and configurations head to the [API reference](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.toolkit.GmailToolkit.html).
#
# ## Setup
#
# To use this toolkit, you will need to set up your credentials explained in the [Gmail API docs](https://developers.google.com/gmail/api/quickstart/python#authorize_credentials_for_a_desktop_application). Once you've downloaded the `credentials.json` file, you can start using the Gmail API.

# ### Installation
#
# This toolkit lives in the `langchain-google-community` package. We'll need the `gmail` extra:

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-google-community\\[gmail\\]")


# If you want to get automated tracing from runs of individual tools, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ## Instantiation
#
# By default the toolkit reads the local `credentials.json` file. You can also manually provide a `Credentials` object.

# In[ ]:


from langchain_google_community import GmailToolkit

toolkit = GmailToolkit()


# ### Customizing Authentication
#
# Behind the scenes, a `googleapi` resource is created using the following methods.
# you can manually build a `googleapi` resource for more auth control.

# In[4]:


from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

# Can review scopes here https://developers.google.com/gmail/api/auth/scopes
# For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)


# ## Tools
#
# View available tools:

# In[2]:


tools = toolkit.get_tools()
tools


# - [GmailCreateDraft](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.create_draft.GmailCreateDraft.html)
# - [GmailSendMessage](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.send_message.GmailSendMessage.html)
# - [GmailSearch](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.search.GmailSearch.html)
# - [GmailGetMessage](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.get_message.GmailGetMessage.html)
# - [GmailGetThread](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.get_thread.GmailGetThread.html)

# ## Use within an agent
#
# Below we show how to incorporate the toolkit into an [agent](/docs/tutorials/agents).
#
# We will need a LLM or chat model:
#
# import ChatModelTabs from "@theme/ChatModelTabs";
#
# <ChatModelTabs customVarName="llm" />
#

# In[3]:


# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# In[4]:


from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools)


# In[5]:


example_query = "Draft an email to fake@fake.com thanking them for coffee."

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()


# ## API reference
#
# For detailed documentation of all `GmailToolkit` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.gmail.toolkit.GmailToolkit.html).
