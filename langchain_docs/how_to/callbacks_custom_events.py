#!/usr/bin/env python
# coding: utf-8

# # How to dispatch custom callback events
# 
# :::info Prerequisites
# 
# This guide assumes familiarity with the following concepts:
# 
# - [Callbacks](/docs/concepts/callbacks)
# - [Custom callback handlers](/docs/how_to/custom_callbacks)
# - [Astream Events API](/docs/concepts/streaming/#astream_events) the `astream_events` method will surface custom callback events.
# :::
# 
# In some situations, you may want to dispatch a custom callback event from within a [Runnable](/docs/concepts/runnables) so it can be surfaced
# in a custom callback handler or via the [Astream Events API](/docs/concepts/streaming/#astream_events).
# 
# For example, if you have a long running tool with multiple steps, you can dispatch custom events between the steps and use these custom events to monitor progress.
# You could also surface these custom events to an end user of your application to show them how the current task is progressing.
# 
# To dispatch a custom event you need to decide on two attributes for the event: the `name` and the `data`.
# 
# | Attribute | Type | Description                                                                                              |
# |-----------|------|----------------------------------------------------------------------------------------------------------|
# | name      | str  | A user defined name for the event.                                                                       |
# | data      | Any  | The data associated with the event. This can be anything, though we suggest making it JSON serializable. |
# 
# 
# :::important
# * Dispatching custom callback events requires `langchain-core>=0.2.15`.
# * Custom callback events can only be dispatched from within an existing `Runnable`.
# * If using `astream_events`, you must use `version='v2'` to see custom events.
# * Sending or rendering custom callbacks events in LangSmith is not yet supported.
# :::
# 
# 
# :::caution COMPATIBILITY
# LangChain cannot automatically propagate configuration, including callbacks necessary for astream_events(), to child runnables if you are running async code in python&lt;=3.10. This is a common reason why you may fail to see events being emitted from custom runnables or tools.
# 
# If you are running python&lt;=3.10, you will need to manually propagate the `RunnableConfig` object to the child runnable in async environments. For an example of how to manually propagate the config, see the implementation of the `bar` RunnableLambda below.
# 
# If you are running python>=3.11, the `RunnableConfig` will automatically propagate to child runnables in async environment. However, it is still a good idea to propagate the `RunnableConfig` manually if your code may run in other Python versions.
# :::

# In[ ]:


# | output: false
# | echo: false

get_ipython().run_line_magic('pip', 'install -qU langchain-core')


# ## Astream Events API
# 
# The most useful way to consume custom events is via the [Astream Events API](/docs/concepts/streaming/#astream_events).
# 
# We can use the `async` `adispatch_custom_event` API to emit custom events in an async setting. 
# 
# 
# :::important
# 
# To see custom events via the astream events API, you need to use the newer `v2` API of `astream_events`.
# :::

# In[3]:


from langchain_core.callbacks.manager import (
    adispatch_custom_event,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig


@RunnableLambda
async def foo(x: str) -> str:
    await adispatch_custom_event("event1", {"x": x})
    await adispatch_custom_event("event2", 5)
    return x


async for event in foo.astream_events("hello world", version="v2"):
    print(event)


# In python &lt;= 3.10, you must propagate the config manually!

# In[7]:


from langchain_core.callbacks.manager import (
    adispatch_custom_event,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig


@RunnableLambda
async def bar(x: str, config: RunnableConfig) -> str:
    """An example that shows how to manually propagate config.

    You must do this if you're running python<=3.10.
    """
    await adispatch_custom_event("event1", {"x": x}, config=config)
    await adispatch_custom_event("event2", 5, config=config)
    return x


async for event in bar.astream_events("hello world", version="v2"):
    print(event)


# ## Async Callback Handler
# 
# You can also consume the dispatched event via an async callback handler.

# In[8]:


from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.callbacks.manager import (
    adispatch_custom_event,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig


class AsyncCustomCallbackHandler(AsyncCallbackHandler):
    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        print(
            f"Received event {name} with data: {data}, with tags: {tags}, with metadata: {metadata} and run_id: {run_id}"
        )


@RunnableLambda
async def bar(x: str, config: RunnableConfig) -> str:
    """An example that shows how to manually propagate config.

    You must do this if you're running python<=3.10.
    """
    await adispatch_custom_event("event1", {"x": x}, config=config)
    await adispatch_custom_event("event2", 5, config=config)
    return x


async_handler = AsyncCustomCallbackHandler()
await foo.ainvoke(1, {"callbacks": [async_handler], "tags": ["foo", "bar"]})


# ## Sync Callback Handler
# 
# Let's see how to emit custom events in a sync environment using `dispatch_custom_event`.
# 
# You **must** call `dispatch_custom_event` from within an existing `Runnable`.

# In[5]:


from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.manager import (
    dispatch_custom_event,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig


class CustomHandler(BaseCallbackHandler):
    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        print(
            f"Received event {name} with data: {data}, with tags: {tags}, with metadata: {metadata} and run_id: {run_id}"
        )


@RunnableLambda
def foo(x: int, config: RunnableConfig) -> int:
    dispatch_custom_event("event1", {"x": x})
    dispatch_custom_event("event2", {"x": x})
    return x


handler = CustomHandler()
foo.invoke(1, {"callbacks": [handler], "tags": ["foo", "bar"]})


# ## Next steps
# 
# You've seen how to emit custom events, you can check out the more in depth guide for [astream events](/docs/how_to/streaming/#using-stream-events) which is the easiest way to leverage custom events.
