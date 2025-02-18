#!/usr/bin/env python
# coding: utf-8

# # Zep Open Source Memory
# > Recall, understand, and extract data from chat histories. Power personalized AI experiences.
#
# >[Zep](https://www.getzep.com) is a long-term memory service for AI Assistant apps.
# > With Zep, you can provide AI assistants with the ability to recall past conversations, no matter how distant,
# > while also reducing hallucinations, latency, and cost.
#
# > Interested in Zep Cloud? See [Zep Cloud Installation Guide](https://help.getzep.com/sdks) and [Zep Cloud Memory Example](https://help.getzep.com/langchain/examples/messagehistory-example)
#
# ## Open Source Installation and Setup
#
# > Zep Open Source project: [https://github.com/getzep/zep](https://github.com/getzep/zep)
# >
# > Zep Open Source Docs: [https://docs.getzep.com/](https://docs.getzep.com/)
#
# ## Example
#
# This notebook demonstrates how to use [Zep](https://www.getzep.com/) as memory for your chatbot.
# REACT Agent Chat Message History with Zep - A long-term memory store for LLM applications.
#
# We'll demonstrate:
#
# 1. Adding conversation history to Zep.
# 2. Running an agent and having message automatically added to the store.
# 3. Viewing the enriched messages.
# 4. Vector search over the conversation history.

# In[1]:


from uuid import uuid4

from langchain.agents import AgentType, initialize_agent
from langchain_community.memory.zep_memory import ZepMemory
from langchain_community.retrievers import ZepRetriever
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_openai import OpenAI

# Set this to your Zep server URL
ZEP_API_URL = "http://localhost:8000"

session_id = str(uuid4())  # This is a unique identifier for the user


# In[1]:


# Provide your OpenAI key
import getpass

openai_key = getpass.getpass()


# In[3]:


# Provide your Zep API key. Note that this is optional. See https://docs.getzep.com/deployment/auth

zep_api_key = getpass.getpass()


# ### Initialize the Zep Chat Message History Class and initialize the Agent
#

# In[4]:


search = WikipediaAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description=(
            "useful for when you need to search online for answers. You should ask"
            " targeted questions"
        ),
    ),
]

# Set up Zep Chat History
memory = ZepMemory(
    session_id=session_id,
    url=ZEP_API_URL,
    api_key=zep_api_key,
    memory_key="chat_history",
)

# Initialize the agent
llm = OpenAI(temperature=0, openai_api_key=openai_key)
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)


# ### Add some history data
#

# In[5]:


# Preload some messages into the memory. The default message window is 12 messages. We want to push beyond this to demonstrate auto-summarization.
test_history = [
    {"role": "human", "content": "Who was Octavia Butler?"},
    {
        "role": "ai",
        "content": (
            "Octavia Estelle Butler (June 22, 1947 – February 24, 2006) was an American"
            " science fiction author."
        ),
    },
    {"role": "human", "content": "Which books of hers were made into movies?"},
    {
        "role": "ai",
        "content": (
            "The most well-known adaptation of Octavia Butler's work is the FX series"
            " Kindred, based on her novel of the same name."
        ),
    },
    {"role": "human", "content": "Who were her contemporaries?"},
    {
        "role": "ai",
        "content": (
            "Octavia Butler's contemporaries included Ursula K. Le Guin, Samuel R."
            " Delany, and Joanna Russ."
        ),
    },
    {"role": "human", "content": "What awards did she win?"},
    {
        "role": "ai",
        "content": (
            "Octavia Butler won the Hugo Award, the Nebula Award, and the MacArthur"
            " Fellowship."
        ),
    },
    {
        "role": "human",
        "content": "Which other women sci-fi writers might I want to read?",
    },
    {
        "role": "ai",
        "content": "You might want to read Ursula K. Le Guin or Joanna Russ.",
    },
    {
        "role": "human",
        "content": (
            "Write a short synopsis of Butler's book, Parable of the Sower. What is it"
            " about?"
        ),
    },
    {
        "role": "ai",
        "content": (
            "Parable of the Sower is a science fiction novel by Octavia Butler,"
            " published in 1993. It follows the story of Lauren Olamina, a young woman"
            " living in a dystopian future where society has collapsed due to"
            " environmental disasters, poverty, and violence."
        ),
        "metadata": {"foo": "bar"},
    },
]

for msg in test_history:
    memory.chat_memory.add_message(
        (
            HumanMessage(content=msg["content"])
            if msg["role"] == "human"
            else AIMessage(content=msg["content"])
        ),
        metadata=msg.get("metadata", {}),
    )


# ### Run the agent
#
# Doing so will automatically add the input and response to the Zep memory.
#

# In[6]:


agent_chain.run(
    input="What is the book's relevance to the challenges facing contemporary society?",
)


# ### Inspect the Zep memory
#
# Note the summary, and that the history has been enriched with token counts, UUIDs, and timestamps.
#
# Summaries are biased towards the most recent messages.
#

# In[9]:


def print_messages(messages):
    for m in messages:
        print(m.type, ":\n", m.dict())


print(memory.chat_memory.zep_summary)
print("\n")
print_messages(memory.chat_memory.messages)


# ### Vector search over the Zep memory
#
# Zep provides native vector search over historical conversation memory via the `ZepRetriever`.
#
# You can use the `ZepRetriever` with chains that support passing in a Langchain `Retriever` object.
#

# In[11]:


retriever = ZepRetriever(
    session_id=session_id,
    url=ZEP_API_URL,
    api_key=zep_api_key,
)

search_results = memory.chat_memory.search("who are some famous women sci-fi authors?")
for r in search_results:
    if r.dist > 0.8:  # Only print results with similarity of 0.8 or higher
        print(r.message, r.dist)


# In[ ]:
