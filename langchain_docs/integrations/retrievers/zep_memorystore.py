#!/usr/bin/env python
# coding: utf-8

# # Zep Open Source
# ## Retriever Example for [Zep](https://docs.getzep.com/)
# 
# > Recall, understand, and extract data from chat histories. Power personalized AI experiences.
# 
# > [Zep](https://www.getzep.com) is a long-term memory service for AI Assistant apps.
# > With Zep, you can provide AI assistants with the ability to recall past conversations, no matter how distant,
# > while also reducing hallucinations, latency, and cost.
# 
# > Interested in Zep Cloud? See [Zep Cloud Installation Guide](https://help.getzep.com/sdks) and [Zep Cloud Retriever Example](https://help.getzep.com/langchain/examples/rag-message-history-example)
# 
# ## Open Source Installation and Setup
# 
# > Zep Open Source project: [https://github.com/getzep/zep](https://github.com/getzep/zep)
# > Zep Open Source Docs: [https://docs.getzep.com/](https://docs.getzep.com/)

# ## Retriever Example
# 
# This notebook demonstrates how to search historical chat message histories using the [Zep Long-term Memory Store](https://getzep.github.io/).
# 
# We'll demonstrate:
# 
# 1. Adding conversation history to the Zep memory store.
# 2. Vector search over the conversation history: 
#     1. With a similarity search over chat messages
#     2. Using maximal marginal relevance re-ranking of a chat message search
#     3. Filtering a search using metadata filters
#     4. A similarity search over summaries of the chat messages
#     5. Using maximal marginal relevance re-ranking of a summary search
# 
# 

# In[1]:


import getpass
import time
from uuid import uuid4

from langchain_community.memory.zep_memory import ZepMemory
from langchain_core.messages import AIMessage, HumanMessage

# Set this to your Zep server URL
ZEP_API_URL = "http://localhost:8000"


# ### Initialize the Zep Chat Message History Class and add a chat message history to the memory store
# 
# **NOTE:** Unlike other Retrievers, the content returned by the Zep Retriever is session/user specific. A `session_id` is required when instantiating the Retriever.

# In[2]:


# Provide your Zep API key. Note that this is optional. See https://docs.getzep.com/deployment/auth
AUTHENTICATE = False

zep_api_key = None
if AUTHENTICATE:
    zep_api_key = getpass.getpass()


# In[3]:


session_id = str(uuid4())  # This is a unique identifier for the user/session

# Initialize the Zep Memory Class
zep_memory = ZepMemory(session_id=session_id, url=ZEP_API_URL, api_key=zep_api_key)


# In[4]:


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
    },
    {"role": "human", "content": "What is the setting of the book?"},
    {
        "role": "ai",
        "content": (
            "The book is set in a dystopian future in the 2020s, where society has"
            " collapsed due to climate change and economic crises."
        ),
    },
    {"role": "human", "content": "Who is the protagonist?"},
    {
        "role": "ai",
        "content": (
            "The protagonist of the book is Lauren Olamina, a young woman who possesses"
            " 'hyperempathy', the ability to feel pain and other sensations she"
            " witnesses."
        ),
    },
    {"role": "human", "content": "What is the main theme of the book?"},
    {
        "role": "ai",
        "content": (
            "The main theme of the book is survival in the face of drastic societal"
            " change and collapse. It also explores themes of adaptability, community,"
            " and the human capacity for change."
        ),
    },
    {"role": "human", "content": "What is the 'Parable of the Sower'?"},
    {
        "role": "ai",
        "content": (
            "The 'Parable of the Sower' is a biblical parable that Butler uses as a"
            " metaphor in the book. In the parable, a sower scatters seeds, some of"
            " which fall on fertile ground and grow, while others fall on rocky ground"
            " or among thorns and fail to grow. The parable is used to illustrate the"
            " importance of receptivity and preparedness in the face of change."
        ),
    },
    {"role": "human", "content": "What is Butler's writing style like?"},
    {
        "role": "ai",
        "content": (
            "Butler's writing style is known for its clarity, directness, and"
            " psychological insight. Her narratives often involve complex, diverse"
            " characters and explore themes of race, gender, and power."
        ),
    },
    {"role": "human", "content": "What other books has she written?"},
    {
        "role": "ai",
        "content": (
            "In addition to 'Parable of the Sower', Butler has written several other"
            " notable works, including 'Kindred', 'Dawn', and 'Parable of the Talents'."
        ),
    },
]

for msg in test_history:
    zep_memory.chat_memory.add_message(
        HumanMessage(content=msg["content"])
        if msg["role"] == "human"
        else AIMessage(content=msg["content"])
    )

time.sleep(
    10
)  # Wait for the messages to be embedded and summarized. Speed depends on OpenAI API latency and your rate limits.


# ### Use the Zep Retriever to vector search over the Zep memory
# 
# Zep provides native vector search over historical conversation memory. Embedding happens automatically.
# 
# NOTE: Embedding of messages occurs asynchronously, so the first query may not return results. Subsequent queries will return results as the embeddings are generated.

# In[5]:


from langchain_community.retrievers.zep import SearchScope, SearchType, ZepRetriever

zep_retriever = ZepRetriever(
    session_id=session_id,  # Ensure that you provide the session_id when instantiating the Retriever
    url=ZEP_API_URL,
    top_k=5,
    api_key=zep_api_key,
)

await zep_retriever.ainvoke("Who wrote Parable of the Sower?")


# We can also use the Zep sync API to retrieve results:

# In[6]:


zep_retriever.invoke("Who wrote Parable of the Sower?")


# ### Reranking using MMR (Maximal Marginal Relevance)
# 
# Zep has native, SIMD-accelerated support for reranking results using MMR. This is useful for removing redundancy in results.

# In[7]:


zep_retriever = ZepRetriever(
    session_id=session_id,  # Ensure that you provide the session_id when instantiating the Retriever
    url=ZEP_API_URL,
    top_k=5,
    api_key=zep_api_key,
    search_type=SearchType.mmr,
    mmr_lambda=0.5,
)

await zep_retriever.ainvoke("Who wrote Parable of the Sower?")


# ### Using metadata filters to refine search results
# 
# Zep supports filtering results by metadata. This is useful for filtering results by entity type, or other metadata.
# 
# More information here: https://docs.getzep.com/sdk/search_query/

# In[8]:


filter = {"where": {"jsonpath": '$[*] ? (@.Label == "WORK_OF_ART")'}}

await zep_retriever.ainvoke("Who wrote Parable of the Sower?", metadata=filter)


# ### Searching over Summaries with MMR Reranking
# 
# Zep automatically generates summaries of chat messages. These summaries can be searched over using the Zep Retriever. Since a summary is a distillation of a conversation, they're more likely to match your search query and offer rich, succinct context to the LLM.
# 
# Successive summaries may include similar content, with Zep's similarity search returning the highest matching results but with little diversity.
# MMR re-ranks the results to ensure that the summaries you populate into your prompt are both relevant and each offers additional information to the LLM.

# In[9]:


zep_retriever = ZepRetriever(
    session_id=session_id,  # Ensure that you provide the session_id when instantiating the Retriever
    url=ZEP_API_URL,
    top_k=3,
    api_key=zep_api_key,
    search_scope=SearchScope.summary,
    search_type=SearchType.mmr,
    mmr_lambda=0.5,
)

await zep_retriever.ainvoke("Who wrote Parable of the Sower?")


# In[ ]:




