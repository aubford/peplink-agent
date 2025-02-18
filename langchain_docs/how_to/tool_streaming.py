#!/usr/bin/env python
# coding: utf-8

# # How to stream tool calls
#
# When [tools](/docs/concepts/tools/) are called in a streaming context,
# [message chunks](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk)
# will be populated with [tool call chunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.ToolCallChunk.html#langchain_core.messages.tool.ToolCallChunk)
# objects in a list via the `.tool_call_chunks` attribute. A `ToolCallChunk` includes
# optional string fields for the tool `name`, `args`, and `id`, and includes an optional
# integer field `index` that can be used to join chunks together. Fields are optional
# because portions of a tool call may be streamed across different chunks (e.g., a chunk
# that includes a substring of the arguments may have null values for the tool name and id).
#
# Because message chunks inherit from their parent message class, an
# [AIMessageChunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk)
# with tool call chunks will also include `.tool_calls` and `.invalid_tool_calls` fields.
# These fields are parsed best-effort from the message's tool call chunks.
#
# Note that not all providers currently support streaming for tool calls. Before we start let's define our tools and our model.

# In[ ]:


from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]


# In[ ]:


import os
from getpass import getpass

from langchain_openai import ChatOpenAI

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# Now let's define our query and stream our output:

# In[ ]:


query = "What is 3 * 12? Also, what is 11 + 49?"

async for chunk in llm_with_tools.astream(query):
    print(chunk.tool_call_chunks)


# Note that adding message chunks will merge their corresponding tool call chunks. This is the principle by which LangChain's various [tool output parsers](/docs/how_to/output_parser_structured) support streaming.
#
# For example, below we accumulate tool call chunks:

# In[ ]:


first = True
async for chunk in llm_with_tools.astream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk

    print(gathered.tool_call_chunks)


# In[ ]:


print(type(gathered.tool_call_chunks[0]["args"]))


# And below we accumulate tool calls to demonstrate partial parsing:

# In[ ]:


first = True
async for chunk in llm_with_tools.astream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk

    print(gathered.tool_calls)


# In[ ]:


print(type(gathered.tool_calls[0]["args"]))
