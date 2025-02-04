#!/usr/bin/env python
# coding: utf-8
---
sidebar_position: 0
---
# # Build a simple LLM application with chat models and prompt templates
# 
# In this quickstart we'll show you how to build a simple LLM application with LangChain. This application will translate text from English into another language. This is a relatively simple LLM application - it's just a single LLM call plus some prompting. Still, this is a great way to get started with LangChain - a lot of features can be built with just some prompting and an LLM call!
# 
# After reading this tutorial, you'll have a high level overview of:
# 
# - Using [language models](/docs/concepts/chat_models)
# 
# - Using [prompt templates](/docs/concepts/prompt_templates)
# 
# - Debugging and tracing your application using [LangSmith](https://docs.smith.langchain.com/)
# 
# Let's dive in!
# 
# ## Setup
# 
# ### Jupyter Notebook
# 
# This and other tutorials are perhaps most conveniently run in a [Jupyter notebooks](https://jupyter.org/). Going through guides in an interactive environment is a great way to better understand them. See [here](https://jupyter.org/install) for instructions on how to install.
# 
# ### Installation
# 
# To install LangChain run:
# 
# import Tabs from '@theme/Tabs';
# import TabItem from '@theme/TabItem';
# import CodeBlock from "@theme/CodeBlock";
# 
# <Tabs>
#   <TabItem value="pip" label="Pip" default>
#     <CodeBlock language="bash">pip install langchain</CodeBlock>
#   </TabItem>
#   <TabItem value="conda" label="Conda">
#     <CodeBlock language="bash">conda install langchain -c conda-forge</CodeBlock>
#   </TabItem>
# </Tabs>
# 
# 
# 
# For more details, see our [Installation guide](/docs/how_to/installation).
# 
# ### LangSmith
# 
# Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
# As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
# The best way to do this is with [LangSmith](https://smith.langchain.com).
# 
# After you sign up at the link above, make sure to set your environment variables to start logging traces:
# 
# ```shell
# export LANGSMITH_TRACING="true"
# export LANGSMITH_API_KEY="..."
# ```
# 
# Or, if in a notebook, you can set them with:
# 
# ```python
# import getpass
# import os
# 
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
# ```

# ## Using Language Models
# 
# First up, let's learn how to use a language model by itself. LangChain supports many different language models that you can use interchangeably. For details on getting started with a specific model, refer to [supported integrations](/docs/integrations/chat/).
# 
# import ChatModelTabs from "@theme/ChatModelTabs";
# 
# <ChatModelTabs openaiParams={`model="gpt-4o-mini"`} />
# 

# In[2]:


# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")


# Let's first use the model directly. [ChatModels](/docs/concepts/chat_models) are instances of LangChain [Runnables](/docs/concepts/runnables/), which means they expose a standard interface for interacting with them. To simply call the model, we can pass in a list of [messages](/docs/concepts/messages/) to the `.invoke` method.

# In[3]:


from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

model.invoke(messages)


# :::tip
# 
# If we've enabled LangSmith, we can see that this run is logged to LangSmith, and can see the [LangSmith trace](https://smith.langchain.com/public/88baa0b2-7c1a-4d09-ba30-a47985dde2ea/r). The LangSmith trace reports [token](/docs/concepts/tokens/) usage information, latency, [standard model parameters](/docs/concepts/chat_models/#standard-parameters) (such as temperature), and other information.
# 
# :::
# 
# Note that ChatModels receive [message](/docs/concepts/messages/) objects as input and generate message objects as output. In addition to text content, message objects convey conversational [roles](/docs/concepts/messages/#role) and hold important data, such as [tool calls](/docs/concepts/tool_calling/) and token usage counts.
# 
# LangChain also supports chat model inputs via strings or [OpenAI format](/docs/concepts/messages/#openai-format). The following are equivalent:
# 
# ```python
# model.invoke("Hello")
# 
# model.invoke([{"role": "user", "content": "Hello"}])
# 
# model.invoke([HumanMessage("Hello")])
# ```
# 
# ### Streaming
# 
# Because chat models are [Runnables](/docs/concepts/runnables/), they expose a standard interface that includes async and streaming modes of invocation. This allows us to stream individual tokens from a chat model:

# In[4]:


for token in model.stream(messages):
    print(token.content, end="|")


# You can find more details on streaming chat model outputs in [this guide](/docs/how_to/chat_streaming/).

# ## Prompt Templates
# 
# Right now we are passing a list of messages directly into the language model. Where does this list of messages come from? Usually, it is constructed from a combination of user input and application logic. This application logic usually takes the raw user input and transforms it into a list of messages ready to pass to the language model. Common transformations include adding a system message or formatting a template with the user input.
# 
# [Prompt templates](/docs/concepts/prompt_templates/) are a concept in LangChain designed to assist with this transformation. They take in raw user input and return data (a prompt) that is ready to pass into a language model. 
# 
# Let's create a prompt template here. It will take in two user variables:
# 
# - `language`: The language to translate text into
# - `text`: The text to translate

# In[5]:


from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)


# Note that `ChatPromptTemplate` supports multiple [message roles](/docs/concepts/messages/#role) in a single template. We format the `language` parameter into the system message, and the user `text` into a user message.

# The input to this prompt template is a dictionary. We can play around with this prompt template by itself to see what it does by itself

# In[6]:


prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

prompt


# We can see that it returns a `ChatPromptValue` that consists of two messages. If we want to access the messages directly we do:

# In[7]:


prompt.to_messages()


# Finally, we can invoke the chat model on the formatted prompt:

# In[8]:


response = model.invoke(prompt)
print(response.content)


# :::tip
# Message `content` can contain both text and [content blocks](/docs/concepts/messages/#aimessage) with additional structure. See [this guide](/docs/how_to/output_parser_string/) for more information.
# :::
# 
# If we take a look at the [LangSmith trace](https://smith.langchain.com/public/3ccc2d5e-2869-467b-95d6-33a577df99a2/r), we can see exactly what prompt the chat model receives, along with [token](/docs/concepts/tokens/) usage information, latency, [standard model parameters](/docs/concepts/chat_models/#standard-parameters) (such as temperature), and other information.

# ## Conclusion
# 
# That's it! In this tutorial you've learned how to create your first simple LLM application. You've learned how to work with language models, how to create a prompt template, and how to get great observability into applications you create with LangSmith.
# 
# This just scratches the surface of what you will want to learn to become a proficient AI Engineer. Luckily - we've got a lot of other resources!
# 
# For further reading on the core concepts of LangChain, we've got detailed [Conceptual Guides](/docs/concepts).
# 
# If you have more specific questions on these concepts, check out the following sections of the how-to guides:
# 
# - [Chat models](/docs/how_to/#chat-models)
# - [Prompt templates](/docs/how_to/#prompt-templates)
# 
# And the LangSmith docs:
# 
# - [LangSmith](https://docs.smith.langchain.com)
