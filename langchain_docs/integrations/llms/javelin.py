#!/usr/bin/env python
# coding: utf-8

# # Javelin AI Gateway Tutorial
# 
# This Jupyter Notebook will explore how to interact with the Javelin AI Gateway using the Python SDK. 
# The Javelin AI Gateway facilitates the utilization of large language models (LLMs) like OpenAI, Cohere, Anthropic, and others by 
# providing a secure and unified endpoint. The gateway itself provides a centralized mechanism to roll out models systematically, 
# provide access security, policy & cost guardrails for enterprises, etc., 
# 
# For a complete listing of all the features & benefits of Javelin, please visit www.getjavelin.io
# 
# 

# ## Step 1: Introduction
# [The Javelin AI Gateway](https://www.getjavelin.io) is an enterprise-grade API Gateway for AI applications. It integrates robust access security, ensuring secure interactions with large language models. Learn more in the [official documentation](https://docs.getjavelin.io).
# 

# ## Step 2: Installation
# Before we begin, we must install the `javelin_sdk` and set up the Javelin API key as an environment variable. 

# In[5]:


pip install 'javelin_sdk'


# ## Step 3: Completions Example
# This section will demonstrate how to interact with the Javelin AI Gateway to get completions from a large language model. Here is a Python script that demonstrates this:
# (note) assumes that you have setup a route in the gateway called 'eng_dept03'

# In[6]:


from langchain.chains import LLMChain
from langchain_community.llms import JavelinAIGateway
from langchain_core.prompts import PromptTemplate

route_completions = "eng_dept03"

gateway = JavelinAIGateway(
    gateway_uri="http://localhost:8000",  # replace with service URL or host/port of Javelin
    route=route_completions,
    model_name="gpt-3.5-turbo-instruct",
)

prompt = PromptTemplate("Translate the following English text to French: {text}")

llmchain = LLMChain(llm=gateway, prompt=prompt)
result = llmchain.run("podcast player")

print(result)


# # Step 4: Embeddings Example
# This section demonstrates how to use the Javelin AI Gateway to obtain embeddings for text queries and documents. Here is a Python script that illustrates this:
# (note) assumes that you have setup a route in the gateway called 'embeddings'

# In[9]:


from langchain_community.embeddings import JavelinAIGatewayEmbeddings

embeddings = JavelinAIGatewayEmbeddings(
    gateway_uri="http://localhost:8000",  # replace with service URL or host/port of Javelin
    route="embeddings",
)

print(embeddings.embed_query("hello"))
print(embeddings.embed_documents(["hello"]))


# # Step 5: Chat Example
# This section illustrates how to interact with the Javelin AI Gateway to facilitate a chat with a large language model. Here is a Python script that demonstrates this:
# (note) assumes that you have setup a route in the gateway called 'mychatbot_route'

# In[8]:


from langchain_community.chat_models import ChatJavelinAIGateway
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Artificial Intelligence has the power to transform humanity and make the world a better place"
    ),
]

chat = ChatJavelinAIGateway(
    gateway_uri="http://localhost:8000",  # replace with service URL or host/port of Javelin
    route="mychatbot_route",
    model_name="gpt-3.5-turbo",
    params={"temperature": 0.1},
)

print(chat(messages))


# Step 6: Conclusion
# This tutorial introduced the Javelin AI Gateway and demonstrated how to interact with it using the Python SDK. 
# Remember to check the Javelin [Python SDK](https://www.github.com/getjavelin.io/javelin-python) for more examples and to explore the official documentation for additional details.
# 
# Happy coding!
