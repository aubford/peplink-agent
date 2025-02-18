#!/usr/bin/env python
# coding: utf-8

# # How to pass runtime secrets to runnables
#
# :::info Requires `langchain-core >= 0.2.22`
#
# :::
#
# We can pass in secrets to our [runnables](/docs/concepts/runnables/) at runtime using the `RunnableConfig`. Specifically we can pass in secrets with a `__` prefix to the `configurable` field. This will ensure that these secrets aren't traced as part of the invocation:

# In[6]:


from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


@tool
def foo(x: int, config: RunnableConfig) -> int:
    """Sum x and a secret int"""
    return x + config["configurable"]["__top_secret_int"]


foo.invoke({"x": 5}, {"configurable": {"__top_secret_int": 2, "traced_key": "bar"}})


# Looking at the LangSmith trace for this run, we can see that "traced_key" was recorded (as part of Metadata) while our secret int was not: https://smith.langchain.com/public/aa7e3289-49ca-422d-a408-f6b927210170/r
