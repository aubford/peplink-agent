#!/usr/bin/env python
# coding: utf-8
---
keywords: [stream]
---
# # How to stream runnables
# 
# :::info Prerequisites
# 
# This guide assumes familiarity with the following concepts:
# - [Chat models](/docs/concepts/chat_models)
# - [LangChain Expression Language](/docs/concepts/lcel)
# - [Output parsers](/docs/concepts/output_parsers)
# 
# :::
# 
# Streaming is critical in making applications based on LLMs feel responsive to end-users.
# 
# Important LangChain primitives like [chat models](/docs/concepts/chat_models), [output parsers](/docs/concepts/output_parsers), [prompts](/docs/concepts/prompt_templates), [retrievers](/docs/concepts/retrievers), and [agents](/docs/concepts/agents) implement the LangChain [Runnable Interface](/docs/concepts/runnables).
# 
# This interface provides two general approaches to stream content:
# 
# 1. sync `stream` and async `astream`: a **default implementation** of streaming that streams the **final output** from the chain.
# 2. async `astream_events` and async `astream_log`: these provide a way to stream both **intermediate steps** and **final output** from the chain.
# 
# Let's take a look at both approaches, and try to understand how to use them.
# 
# :::info
# For a higher-level overview of streaming techniques in LangChain, see [this section of the conceptual guide](/docs/concepts/streaming).
# :::
# 
# ## Using Stream
# 
# All `Runnable` objects implement a sync method called `stream` and an async variant called `astream`. 
# 
# These methods are designed to stream the final output in chunks, yielding each chunk as soon as it is available.
# 
# Streaming is only possible if all steps in the program know how to process an **input stream**; i.e., process an input chunk one at a time, and yield a corresponding output chunk.
# 
# The complexity of this processing can vary, from straightforward tasks like emitting tokens produced by an LLM, to more challenging ones like streaming parts of JSON results before the entire JSON is complete.
# 
# The best place to start exploring streaming is with the single most important components in LLMs apps-- the LLMs themselves!
# 
# ### LLMs and Chat Models
# 
# Large language models and their chat variants are the primary bottleneck in LLM based apps.
# 
# Large language models can take **several seconds** to generate a complete response to a query. This is far slower than the **~200-300 ms** threshold at which an application feels responsive to an end user.
# 
# The key strategy to make the application feel more responsive is to show intermediate progress; viz., to stream the output from the model **token by token**.
# 
# We will show examples of streaming using a chat model. Choose one from the options below:
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs
#   customVarName="model"
# />
# 

# In[1]:


# | output: false
# | echo: false

import os
from getpass import getpass

keys = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
]

for key in keys:
    if key not in os.environ:
        os.environ[key] = getpass(f"Enter API Key for {key}=?")


from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)


# Let's start with the sync `stream` API:

# In[2]:


chunks = []
for chunk in model.stream("what color is the sky?"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)


# Alternatively, if you're working in an async environment, you may consider using the async `astream` API:

# In[3]:


chunks = []
async for chunk in model.astream("what color is the sky?"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)


# Let's inspect one of the chunks

# In[4]:


chunks[0]


# We got back something called an `AIMessageChunk`. This chunk represents a part of an `AIMessage`.
# 
# Message chunks are additive by design -- one can simply add them up to get the state of the response so far!

# In[5]:


chunks[0] + chunks[1] + chunks[2] + chunks[3] + chunks[4]


# ### Chains
# 
# Virtually all LLM applications involve more steps than just a call to a language model.
# 
# Let's build a simple chain using `LangChain Expression Language` (`LCEL`) that combines a prompt, model and a parser and verify that streaming works.
# 
# We will use [`StrOutputParser`](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html) to parse the output from the model. This is a simple parser that extracts the `content` field from an `AIMessageChunk`, giving us the `token` returned by the model.
# 
# :::tip
# LCEL is a *declarative* way to specify a "program" by chainining together different LangChain primitives. Chains created using LCEL benefit from an automatic implementation of `stream` and `astream` allowing streaming of the final output. In fact, chains created with LCEL implement the entire standard Runnable interface.
# :::

# In[6]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | model | parser

async for chunk in chain.astream({"topic": "parrot"}):
    print(chunk, end="|", flush=True)


# Note that we're getting streaming output even though we're using `parser` at the end of the chain above. The `parser` operates on each streaming chunk individidually. Many of the [LCEL primitives](/docs/how_to#langchain-expression-language-lcel) also support this kind of transform-style passthrough streaming, which can be very convenient when constructing apps. 
# 
# Custom functions can be [designed to return generators](/docs/how_to/functions#streaming), which are able to operate on streams.
# 
# Certain runnables, like [prompt templates](/docs/how_to#prompt-templates) and [chat models](/docs/how_to#chat-models), cannot process individual chunks and instead aggregate all previous steps. Such runnables can interrupt the streaming process.

# :::note
# The LangChain Expression language allows you to separate the construction of a chain from the mode in which it is used (e.g., sync/async, batch/streaming etc.). If this is not relevant to what you're building, you can also rely on a standard **imperative** programming approach by
# caling `invoke`, `batch` or `stream` on each component individually, assigning the results to variables and then using them downstream as you see fit.
# 
# :::

# ### Working with Input Streams
# 
# What if you wanted to stream JSON from the output as it was being generated?
# 
# If you were to rely on `json.loads` to parse the partial json, the parsing would fail as the partial json wouldn't be valid json.
# 
# You'd likely be at a complete loss of what to do and claim that it wasn't possible to stream JSON.
# 
# Well, turns out there is a way to do it -- the parser needs to operate on the **input stream**, and attempt to "auto-complete" the partial json into a valid state.
# 
# Let's see such a parser in action to understand what this means.

# In[7]:


from langchain_core.output_parsers import JsonOutputParser

chain = (
    model | JsonOutputParser()
)  # Due to a bug in older versions of Langchain, JsonOutputParser did not stream results from some models
async for text in chain.astream(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`"
):
    print(text, flush=True)


# Now, let's **break** streaming. We'll use the previous example and append an extraction function at the end that extracts the country names from the finalized JSON.
# 
# :::warning
# Any steps in the chain that operate on **finalized inputs** rather than on **input streams** can break streaming functionality via `stream` or `astream`.
# :::
# 
# :::tip
# Later, we will discuss the `astream_events` API which streams results from intermediate steps. This API will stream results from intermediate steps even if the chain contains steps that only operate on **finalized inputs**.
# :::

# In[8]:


from langchain_core.output_parsers import (
    JsonOutputParser,
)


# A function that operates on finalized inputs
# rather than on an input_stream
def _extract_country_names(inputs):
    """A function that does not operates on input streams and breaks streaming."""
    if not isinstance(inputs, dict):
        return ""

    if "countries" not in inputs:
        return ""

    countries = inputs["countries"]

    if not isinstance(countries, list):
        return ""

    country_names = [
        country.get("name") for country in countries if isinstance(country, dict)
    ]
    return country_names


chain = model | JsonOutputParser() | _extract_country_names

async for text in chain.astream(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`"
):
    print(text, end="|", flush=True)


# #### Generator Functions
# 
# Let's fix the streaming using a generator function that can operate on the **input stream**.
# 
# :::tip
# A generator function (a function that uses `yield`) allows writing code that operates on **input streams**
# :::

# In[9]:


from langchain_core.output_parsers import JsonOutputParser


async def _extract_country_names_streaming(input_stream):
    """A function that operates on input streams."""
    country_names_so_far = set()

    async for input in input_stream:
        if not isinstance(input, dict):
            continue

        if "countries" not in input:
            continue

        countries = input["countries"]

        if not isinstance(countries, list):
            continue

        for country in countries:
            name = country.get("name")
            if not name:
                continue
            if name not in country_names_so_far:
                yield name
                country_names_so_far.add(name)


chain = model | JsonOutputParser() | _extract_country_names_streaming

async for text in chain.astream(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`",
):
    print(text, end="|", flush=True)


# :::note
# Because the code above is relying on JSON auto-completion, you may see partial names of countries (e.g., `Sp` and `Spain`), which is not what one would want for an extraction result!
# 
# We're focusing on streaming concepts, not necessarily the results of the chains.
# :::

# ### Non-streaming components
# 
# Some built-in components like Retrievers do not offer any `streaming`. What happens if we try to `stream` them? 🤨

# In[10]:


from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho", "harrison likes spicy food"],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

chunks = [chunk for chunk in retriever.stream("where did harrison work?")]
chunks


# Stream just yielded the final result from that component.
# 
# This is OK 🥹! Not all components have to implement streaming -- in some cases streaming is either unnecessary, difficult or just doesn't make sense.
# 
# :::tip
# An LCEL chain constructed using non-streaming components, will still be able to stream in a lot of cases, with streaming of partial output starting after the last non-streaming step in the chain.
# :::

# In[11]:


retrieval_chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


# In[12]:


for chunk in retrieval_chain.stream(
    "Where did harrison work? " "Write 3 made up sentences about this place."
):
    print(chunk, end="|", flush=True)


# Now that we've seen how `stream` and `astream` work, let's venture into the world of streaming events. 🏞️

# ## Using Stream Events
# 
# Event Streaming is a **beta** API. This API may change a bit based on feedback.
# 
# :::note
# 
# This guide demonstrates the `V2` API and requires langchain-core >= 0.2. For the `V1` API compatible with older versions of LangChain, see [here](https://python.langchain.com/v0.1/docs/expression_language/streaming/#using-stream-events).
# :::

# In[ ]:


import langchain_core

langchain_core.__version__


# For the `astream_events` API to work properly:
# 
# * Use `async` throughout the code to the extent possible (e.g., async tools etc)
# * Propagate callbacks if defining custom functions / runnables
# * Whenever using runnables without LCEL, make sure to call `.astream()` on LLMs rather than `.ainvoke` to force the LLM to stream tokens.
# * Let us know if anything doesn't work as expected! :)
# 
# ### Event Reference
# 
# Below is a reference table that shows some events that might be emitted by the various Runnable objects.
# 
# 
# :::note
# When streaming is implemented properly, the inputs to a runnable will not be known until after the input stream has been entirely consumed. This means that `inputs` will often be included only for `end` events and rather than for `start` events.
# :::
# 
# | event                | name             | chunk                           | input                                         | output                                          |
# |----------------------|------------------|---------------------------------|-----------------------------------------------|-------------------------------------------------|
# | on_chat_model_start  | [model name]     |                                 | \{"messages": [[SystemMessage, HumanMessage]]\} |                                                 |
# | on_chat_model_stream | [model name]     | AIMessageChunk(content="hello") |                                               |                                                 |
# | on_chat_model_end    | [model name]     |                                 | \{"messages": [[SystemMessage, HumanMessage]]\} | AIMessageChunk(content="hello world")           |
# | on_llm_start         | [model name]     |                                 | \{'input': 'hello'\}                            |                                                 |
# | on_llm_stream        | [model name]     | 'Hello'                         |                                               |                                                 |
# | on_llm_end           | [model name]     |                                 | 'Hello human!'                                |                                                 |
# | on_chain_start       | format_docs      |                                 |                                               |                                                 |
# | on_chain_stream      | format_docs      | "hello world!, goodbye world!"  |                                               |                                                 |
# | on_chain_end         | format_docs      |                                 | [Document(...)]                               | "hello world!, goodbye world!"                  |
# | on_tool_start        | some_tool        |                                 | \{"x": 1, "y": "2"\}                            |                                                 |
# | on_tool_end          | some_tool        |                                 |                                               | \{"x": 1, "y": "2"\}                              |
# | on_retriever_start   | [retriever name] |                                 | \{"query": "hello"\}                            |                                                 |
# | on_retriever_end     | [retriever name] |                                 | \{"query": "hello"\}                            | [Document(...), ..]                             |
# | on_prompt_start      | [template_name]  |                                 | \{"question": "hello"\}                         |                                                 |
# | on_prompt_end        | [template_name]  |                                 | \{"question": "hello"\}                         | ChatPromptValue(messages: [SystemMessage, ...]) |

# ### Chat Model
# 
# Let's start off by looking at the events produced by a chat model.

# In[13]:


events = []
async for event in model.astream_events("hello"):
    events.append(event)


# :::note
# 
# For `langchain-core<0.3.37`, set the `version` kwarg explicitly (e.g., `model.astream_events("hello", version="v2")`).
# 
# :::

# Let's take a look at the few of the start event and a few of the end events.

# In[14]:


events[:3]


# In[15]:


events[-2:]


# ### Chain
# 
# Let's revisit the example chain that parsed streaming JSON to explore the streaming events API.

# In[16]:


chain = (
    model | JsonOutputParser()
)  # Due to a bug in older versions of Langchain, JsonOutputParser did not stream results from some models

events = [
    event
    async for event in chain.astream_events(
        "output a list of the countries france, spain and japan and their populations in JSON format. "
        'Use a dict with an outer key of "countries" which contains a list of countries. '
        "Each country should have the key `name` and `population`",
    )
]


# If you examine at the first few events, you'll notice that there are **3** different start events rather than **2** start events.
# 
# The three start events correspond to:
# 
# 1. The chain (model + parser)
# 2. The model
# 3. The parser

# In[18]:


events[:3]


# What do you think you'd see if you looked at the last 3 events? what about the middle?

# Let's use this API to take output the stream events from the model and the parser. We're ignoring start events, end events and events from the chain.

# In[19]:


num_events = 0

async for event in chain.astream_events(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(
            f"Chat model chunk: {repr(event['data']['chunk'].content)}",
            flush=True,
        )
    if kind == "on_parser_stream":
        print(f"Parser chunk: {event['data']['chunk']}", flush=True)
    num_events += 1
    if num_events > 30:
        # Truncate the output
        print("...")
        break


# Because both the model and the parser support streaming, we see streaming events from both components in real time! Kind of cool isn't it? 🦜

# ### Filtering Events
# 
# Because this API produces so many events, it is useful to be able to filter on events.
# 
# You can filter by either component `name`, component `tags` or component `type`.
# 
# #### By Name

# In[20]:


chain = model.with_config({"run_name": "model"}) | JsonOutputParser().with_config(
    {"run_name": "my_parser"}
)

max_events = 0
async for event in chain.astream_events(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`",
    include_names=["my_parser"],
):
    print(event)
    max_events += 1
    if max_events > 10:
        # Truncate output
        print("...")
        break


# #### By Type

# In[21]:


chain = model.with_config({"run_name": "model"}) | JsonOutputParser().with_config(
    {"run_name": "my_parser"}
)

max_events = 0
async for event in chain.astream_events(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    include_types=["chat_model"],
):
    print(event)
    max_events += 1
    if max_events > 10:
        # Truncate output
        print("...")
        break


# #### By Tags
# 
# :::caution
# 
# Tags are inherited by child components of a given runnable. 
# 
# If you're using tags to filter, make sure that this is what you want.
# :::

# In[22]:


chain = (model | JsonOutputParser()).with_config({"tags": ["my_chain"]})

max_events = 0
async for event in chain.astream_events(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    include_tags=["my_chain"],
):
    print(event)
    max_events += 1
    if max_events > 10:
        # Truncate output
        print("...")
        break


# ### Non-streaming components
# 
# Remember how some components don't stream well because they don't operate on **input streams**?
# 
# While such components can break streaming of the final output when using `astream`, `astream_events` will still yield streaming events from intermediate steps that support streaming!

# In[23]:


# Function that does not support streaming.
# It operates on the finalizes inputs rather than
# operating on the input stream.
def _extract_country_names(inputs):
    """A function that does not operates on input streams and breaks streaming."""
    if not isinstance(inputs, dict):
        return ""

    if "countries" not in inputs:
        return ""

    countries = inputs["countries"]

    if not isinstance(countries, list):
        return ""

    country_names = [
        country.get("name") for country in countries if isinstance(country, dict)
    ]
    return country_names


chain = (
    model | JsonOutputParser() | _extract_country_names
)  # This parser only works with OpenAI right now


# As expected, the `astream` API doesn't work correctly because `_extract_country_names` doesn't operate on streams.

# In[24]:


async for chunk in chain.astream(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`",
):
    print(chunk, flush=True)


# Now, let's confirm that with astream_events we're still seeing streaming output from the model and the parser.

# In[25]:


num_events = 0

async for event in chain.astream_events(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(
            f"Chat model chunk: {repr(event['data']['chunk'].content)}",
            flush=True,
        )
    if kind == "on_parser_stream":
        print(f"Parser chunk: {event['data']['chunk']}", flush=True)
    num_events += 1
    if num_events > 30:
        # Truncate the output
        print("...")
        break


# ### Propagating Callbacks
# 
# :::caution
# If you're using invoking runnables inside your tools, you need to propagate callbacks to the runnable; otherwise, no stream events will be generated.
# :::
# 
# :::note
# When using `RunnableLambdas` or `@chain` decorator, callbacks are propagated automatically behind the scenes.
# :::

# In[26]:


from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool


def reverse_word(word: str):
    return word[::-1]


reverse_word = RunnableLambda(reverse_word)


@tool
def bad_tool(word: str):
    """Custom tool that doesn't propagate callbacks."""
    return reverse_word.invoke(word)


async for event in bad_tool.astream_events("hello"):
    print(event)


# Here's a re-implementation that does propagate callbacks correctly. You'll notice that now we're getting events from the `reverse_word` runnable as well.

# In[27]:


@tool
def correct_tool(word: str, callbacks):
    """A tool that correctly propagates callbacks."""
    return reverse_word.invoke(word, {"callbacks": callbacks})


async for event in correct_tool.astream_events("hello"):
    print(event)


# If you're invoking runnables from within Runnable Lambdas or `@chains`, then callbacks will be passed automatically on your behalf.

# In[28]:


from langchain_core.runnables import RunnableLambda


async def reverse_and_double(word: str):
    return await reverse_word.ainvoke(word) * 2


reverse_and_double = RunnableLambda(reverse_and_double)

await reverse_and_double.ainvoke("1234")

async for event in reverse_and_double.astream_events("1234"):
    print(event)


# And with the `@chain` decorator:

# In[29]:


from langchain_core.runnables import chain


@chain
async def reverse_and_double(word: str):
    return await reverse_word.ainvoke(word) * 2


await reverse_and_double.ainvoke("1234")

async for event in reverse_and_double.astream_events("1234"):
    print(event)


# ## Next steps
# 
# Now you've learned some ways to stream both final outputs and internal steps with LangChain.
# 
# To learn more, check out the other how-to guides in this section, or the [conceptual guide on Langchain Expression Language](/docs/concepts/lcel/).
