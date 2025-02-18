#!/usr/bin/env python
# coding: utf-8

# # How to parse XML output
#
# :::info Prerequisites
#
# This guide assumes familiarity with the following concepts:
# - [Chat models](/docs/concepts/chat_models)
# - [Output parsers](/docs/concepts/output_parsers)
# - [Prompt templates](/docs/concepts/prompt_templates)
# - [Structured output](/docs/how_to/structured_output)
# - [Chaining runnables together](/docs/how_to/sequence/)
#
# :::
#
# LLMs from different providers often have different strengths depending on the specific data they are trained on. This also means that some may be "better" and more reliable at generating output in formats other than JSON.
#
# This guide shows you how to use the [`XMLOutputParser`](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.xml.XMLOutputParser.html) to prompt models for XML output, then and [parse](/docs/concepts/output_parsers/) that output into a usable format.
#
# :::note
# Keep in mind that large language models are leaky abstractions! You'll have to use an LLM with sufficient capacity to generate well-formed XML.
# :::
#
# In the following examples, we use Anthropic's Claude-2 model (https://docs.anthropic.com/claude/docs), which is one such model that is optimized for XML tags.

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain langchain-anthropic")

import os
from getpass import getpass

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass()


# Let's start with a simple request to the model.

# In[7]:


from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatAnthropic(model="claude-2.1", max_tokens_to_sample=512, temperature=0.1)

actor_query = "Generate the shortened filmography for Tom Hanks."

output = model.invoke(
    f"""{actor_query}
Please enclose the movies in <movie></movie> tags"""
)

print(output.content)


# This actually worked pretty well! But it would be nice to parse that XML into a more easily usable format. We can use the `XMLOutputParser` to both add default format instructions to the prompt and parse outputted XML into a dict:

# In[3]:


parser = XMLOutputParser()

# We will add these instructions to the prompt below
parser.get_format_instructions()


# In[4]:


prompt = PromptTemplate(
    template="""{query}\n{format_instructions}""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

output = chain.invoke({"query": actor_query})
print(output)


# We can also add some tags to tailor the output to our needs. You can and should experiment with adding your own formatting hints in the other parts of your prompt to either augment or replace the default instructions:

# In[6]:


parser = XMLOutputParser(tags=["movies", "actor", "film", "name", "genre"])

# We will add these instructions to the prompt below
parser.get_format_instructions()


# In[5]:


prompt = PromptTemplate(
    template="""{query}\n{format_instructions}""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


chain = prompt | model | parser

output = chain.invoke({"query": actor_query})

print(output)


# This output parser also supports streaming of partial chunks. Here's an example:

# In[7]:


for s in chain.stream({"query": actor_query}):
    print(s)


# ## Next steps
#
# You've now learned how to prompt a model to return XML. Next, check out the [broader guide on obtaining structured output](/docs/how_to/structured_output) for other related techniques.
