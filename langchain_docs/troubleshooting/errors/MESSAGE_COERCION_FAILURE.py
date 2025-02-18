#!/usr/bin/env python
# coding: utf-8

# # MESSAGE_COERCION_FAILURE
#
# Instead of always requiring instances of `BaseMessage`, several modules in LangChain take `MessageLikeRepresentation`, which is defined as:

# In[4]:


from typing import Union

from langchain_core.prompts.chat import (
    BaseChatPromptTemplate,
    BaseMessage,
    BaseMessagePromptTemplate,
)

MessageLikeRepresentation = Union[
    Union[BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate],
    tuple[
        Union[str, type],
        Union[str, list[dict], list[object]],
    ],
    str,
]


# These include OpenAI style message objects (`{ role: "user", content: "Hello world!" }`),
# tuples, and plain strings (which are converted to [`HumanMessages`](/docs/concepts/messages/#humanmessage)).
#
# If one of these modules receives a value outside of one of these formats, you will receive an error like the following:

# In[5]:


from langchain_anthropic import ChatAnthropic

uncoercible_message = {"role": "HumanMessage", "random_field": "random value"}

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

model.invoke([uncoercible_message])


# ## Troubleshooting
#
# The following may help resolve this error:
#
# - Ensure that all inputs to chat models are an array of LangChain message classes or a supported message-like.
#   - Check that there is no stringification or other unexpected transformation occuring.
# - Check the error's stack trace and add log or debugger statements.

#
