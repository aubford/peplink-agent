#!/usr/bin/env python
# coding: utf-8

# # Amazon API Gateway

# >[Amazon API Gateway](https://aws.amazon.com/api-gateway/) is a fully managed service that makes it easy for developers to create, publish, maintain, monitor, and secure APIs at any >scale. APIs act as the "front door" for applications to access data, business logic, or functionality from your backend services. Using `API Gateway`, you can create RESTful APIs and >WebSocket APIs that enable real-time two-way communication applications. API Gateway supports containerized and serverless workloads, as well as web applications.
# 
# >`API Gateway` handles all the tasks involved in accepting and processing up to hundreds of thousands of concurrent API calls, including traffic management, CORS support, authorization >and access control, throttling, monitoring, and API version management. `API Gateway` has no minimum fees or startup costs. You pay for the API calls you receive and the amount of data >transferred out and, with the `API Gateway` tiered pricing model, you can reduce your cost as your API usage scales.

# In[ ]:


##Installing the langchain packages needed to use the integration
get_ipython().run_line_magic('pip', 'install -qU langchain-community')


# ## LLM

# In[1]:


from langchain_community.llms import AmazonAPIGateway


# In[2]:


api_url = "https://<api_gateway_id>.execute-api.<region>.amazonaws.com/LATEST/HF"
llm = AmazonAPIGateway(api_url=api_url)


# In[3]:


# These are sample parameters for Falcon 40B Instruct Deployed from Amazon SageMaker JumpStart
parameters = {
    "max_new_tokens": 100,
    "num_return_sequences": 1,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": False,
    "return_full_text": True,
    "temperature": 0.2,
}

prompt = "what day comes after Friday?"
llm.model_kwargs = parameters
llm(prompt)


# ## Agent

# In[4]:


from langchain.agents import AgentType, initialize_agent, load_tools

parameters = {
    "max_new_tokens": 50,
    "num_return_sequences": 1,
    "top_k": 250,
    "top_p": 0.25,
    "do_sample": False,
    "temperature": 0.1,
}

llm.model_kwargs = parameters

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["python_repl", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Now let's test it out!
agent.run(
    """
Write a Python script that prints "Hello, world!"
"""
)


# In[5]:


result = agent.run(
    """
What is 2.3 ^ 4.5?
"""
)

result.split("\n")[0]

