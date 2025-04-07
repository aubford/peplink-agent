#!/usr/bin/env python
# coding: utf-8

# # Google Calendar Toolkit
# 
# > [Google Calendar](https://workspace.google.com/intl/en-419/products/calendar/) is a product of Google Workspace that allows users to organize their schedules and events. It is a cloud-based calendar that allows users to create, edit, and delete events. It also allows users to share their calendars with others.
# 
# ## Overview
# 
# This notebook will help you get started with the Google Calendar Toolkit. This toolkit interacts with the Google Calendar API to perform various operations on the calendar. It allows you to:
# 
# - Create events.
# - Search events.
# - Update events.
# - Move events between different calendars.
# - Delete events.
# - List events.
# 
# ## Setup
# 
# To use this toolkit, you will need to:
# 
# 1. Have a Google account with access to Google Calendar.
# 2. Set up your credentials as explained in the [Google Calendar API docs](https://developers.google.com/calendar/api/quickstart/python#authorize_credentials_for_a_desktop_application). Once you've downloaded the `credentials.json` file, you can start using the Google Calendar API.
# 
# To enable automated tracing of individual tools, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[ ]:


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")


# ### Installation
# 
# This toolkit lives in the `langchain-google-community` package of the [langchain-google](https://github.com/langchain-ai/langchain-google) repository. We'll need the `calendar` extra:

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-google-community\\[calendar\\]')


# ## Instantiation
# 
# By default the toolkit reads the local `credentials.json` file. You can also manually provide a `Credentials` object.

# In[ ]:


from langchain_google_community import CalendarToolkit

toolkit = CalendarToolkit()


# ### Customizing Authentication
# 
# Behind the scenes, a `googleapi` resource is created using the following methods. you can manually build a `googleapi` resource for more auth control.

# In[2]:


from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.utils import (
    build_resource_service,
    get_google_credentials,
)

# Can review scopes here: https://developers.google.com/calendar/api/auth
# For instance, readonly scope is https://www.googleapis.com/auth/calendar.readonly
credentials = get_google_credentials(
    token_file="token.json",
    scopes=["https://www.googleapis.com/auth/calendar"],
    client_secrets_file="credentials.json",
)

api_resource = build_resource_service(credentials=credentials)
toolkit = CalendarToolkit(api_resource=api_resource)


# ## Tools
# View available tools:

# In[5]:


tools = toolkit.get_tools()
tools


# - [CalendarCreateEvent](https://python.langchain.com/api_reference/google_community/calendar/langchain_google_community.calendar.create_event.CalendarCreateEvent.html)
# - [CalendarSearchEvents](https://python.langchain.com/api_reference/google_community/calendar/langchain_google_community.calendar.search_events.CalendarSearchEvents.html)
# - [CalendarUpdateEvent](https://python.langchain.com/api_reference/google_community/calendar/langchain_google_community.calendar.update_event.CalendarUpdateEvent.html)
# - [GetCalendarsInfo](https://python.langchain.com/api_reference/google_community/calendar/langchain_google_community.calendar.get_calendars_info.GetCalendarsInfo.html)
# - [CalendarMoveEvent](https://python.langchain.com/api_reference/google_community/calendar/langchain_google_community.calendar.move_event.CalendarMoveEvent.html)
# - [CalendarDeleteEvent](https://python.langchain.com/api_reference/google_community/calendar/langchain_google_community.calendar.delete_event.CalendarDeleteEvent.html)
# - [GetCurrentDatetime](https://python.langchain.com/api_reference/google_community/calendar/langchain_google_community.calendar.current_datetime.GetCurrentDatetime.html)

# ## Invocation
# 
# ### [Invoke directly with args](/docs/concepts/tools/#use-the-tool-directly)
# 
# You can invoke the tool directly by passing the required arguments in a dictionary format. Here is an example of creating a new event using the `CalendarCreateEvent` tool.

# In[ ]:


from langchain_google_community.calendar.create_event import CalendarCreateEvent

tool = CalendarCreateEvent()
tool.invoke(
    {
        "summary": "Calculus exam",
        "start_datetime": "2025-07-11 11:00:00",
        "end_datetime": "2025-07-11 13:00:00",
        "timezone": "America/Mexico_City",
        "location": "UAM Cuajimalpa",
        "description": "Event created from the LangChain toolkit",
        "reminders": [{"method": "popup", "minutes": 60}],
        "conference_data": True,
        "color_id": "5",
    }
)


# ## Use within an agent
# 
# Below we show how to incorporate the toolkit into an [agent](/docs/tutorials/agents).
# 
# We will need a LLM or chat model:
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs customVarName="llm" />

# In[ ]:


# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# In[8]:


from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools)


# In[10]:


example_query = "Create a green event for this afternoon to go for a 30-minute run."

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()


# ## API reference
# 
# - Refer to the [Google Calendar API overview](https://developers.google.com/calendar/api/guides/overview) for more details from Google Calendar API.
# - For detailed documentation of all Google Calendar Toolkit features and configurations head to the [calendar documentation](https://python.langchain.com/api_reference/google_community/calendar.html).
