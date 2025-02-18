#!/usr/bin/env python
# coding: utf-8

# # How to bind model-specific tools
#
# Providers adopt different conventions for formatting tool schemas.
# For instance, OpenAI uses a format like this:
#
# - `type`: The type of the tool. At the time of writing, this is always `"function"`.
# - `function`: An object containing tool parameters.
# - `function.name`: The name of the schema to output.
# - `function.description`: A high level description of the schema to output.
# - `function.parameters`: The nested details of the schema you want to extract, formatted as a [JSON schema](https://json-schema.org/) dict.
#
# We can bind this model-specific format directly to the model as well if preferred. Here's an example:

# In[ ]:


from langchain_openai import ChatOpenAI

model = ChatOpenAI()

model_with_tools = model.bind(
    tools=[
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two integers together.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First integer"},
                        "b": {"type": "number", "description": "Second integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
)

model_with_tools.invoke("Whats 119 times 8?")


# This is functionally equivalent to the `bind_tools()` method.
