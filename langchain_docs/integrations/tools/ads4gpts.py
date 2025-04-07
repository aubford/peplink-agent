#!/usr/bin/env python
# coding: utf-8

# # ADS4GPTs
# 
# Integrate AI native advertising into your Agentic application.
# 

# ## Overview

# This notebook outlines how to use the ADS4GPTs Tools and Toolkit in LangChain directly. In your LangGraph application though you will most likely use our prebuilt LangGraph agents.

# ## Setup

# ### Install ADS4GPTs Package
# Install the ADS4GPTs package using pip.

# In[ ]:


# Install ADS4GPTs Package
# Install the ADS4GPTs package using pip
get_ipython().system('pip install ads4gpts-langchain')


# Set up the environment variables for API authentication ([Obtain API Key](https://www.ads4gpts.com)).

# In[ ]:


# Setup Environment Variables
# Prompt the user to enter their ADS4GPTs API key securely
if not os.environ.get("ADS4GPTS_API_KEY"):
    os.environ["ADS4GPTS_API_KEY"] = getpass("Enter your ADS4GPTS API key: ")


# ## Instantiation

# Import the necessary libraries, including ADS4GPTs tools and toolkit.
# 
# Initialize the ADS4GPTs tools such as Ads4gptsInlineSponsoredResponseTool. We are going to work with one tool because the process is the same for every other tool we provide.

# In[ ]:


# Import Required Libraries

import os
from getpass import getpass

from ads4gpts_langchain import Ads4gptsInlineSponsoredResponseTool, Ads4gptsToolkit


# In[ ]:


# Initialize ADS4GPTs Tools
# Initialize the Ads4gptsInlineSponsoredResponseTool
inline_sponsored_response_tool = Ads4gptsInlineSponsoredResponseTool(
    ads4gpts_api_key=os.environ["ADS4GPTS_API_KEY"],
)


# ### Toolkit Instantiation
# Initialize the Ads4gptsToolkit with the required parameters.

# In[ ]:


# Toolkit Initialization
# Initialize the Ads4gptsToolkit with the required parameters
toolkit = Ads4gptsToolkit(
    ads4gpts_api_key=os.environ["ADS4GPTS_API_KEY"],
)

# Retrieve tools from the toolkit
tools = toolkit.get_tools()

# Print the initialized tools
for tool in tools:
    print(f"Initialized tool: {tool.__class__.__name__}")


# ## Invocation

# Run the ADS4GPTs tools with sample inputs and display the results.

# In[ ]:


# Run ADS4GPTs Tools
# Sample input data for the tools
sample_input = {
    "id": "test_id",
    "user_gender": "female",
    "user_age": "25-34",
    "user_persona": "test_persona",
    "ad_recommendation": "test_recommendation",
    "undesired_ads": "test_undesired_ads",
    "context": "test_context",
    "num_ads": 1,
    "style": "neutral",
}

# Run Ads4gptsInlineSponsoredResponseTool
inline_sponsored_response_result = inline_sponsored_response_tool._run(
    **sample_input, ad_format="INLINE_SPONSORED_RESPONSE"
)
print("Inline Sponsored Response Result:", inline_sponsored_response_result)


# ### Async Run ADS4GPTs Tools
# Run the ADS4GPTs tools asynchronously with sample inputs and display the results.

# In[ ]:


import asyncio


# Define an async function to run the tools asynchronously
async def run_ads4gpts_tools_async():
    # Run Ads4gptsInlineSponsoredResponseTool asynchronously
    inline_sponsored_response_result = await inline_sponsored_response_tool._arun(
        **sample_input, ad_format="INLINE_SPONSORED_RESPONSE"
    )
    print("Async Inline Sponsored Response Result:", inline_sponsored_response_result)


# ### Toolkit Invocation
# Use the Ads4gptsToolkit to get and run tools.

# In[ ]:


# Sample input data for the tools
sample_input = {
    "id": "test_id",
    "user_gender": "female",
    "user_age": "25-34",
    "user_persona": "test_persona",
    "ad_recommendation": "test_recommendation",
    "undesired_ads": "test_undesired_ads",
    "context": "test_context",
    "num_ads": 1,
    "style": "neutral",
}

# Run one tool and print the result
tool = tools[0]
result = tool._run(**sample_input)
print(f"Result from {tool.__class__.__name__}:", result)


# Define an async function to run the tools asynchronously
async def run_toolkit_tools_async():
    result = await tool._arun(**sample_input)
    print(f"Async result from {tool.__class__.__name__}:", result)


# Execute the async function
await run_toolkit_tools_async()


# ## Chaining
# 

# In[12]:


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OPENAI_API_KEY API key: ")


# In[ ]:


import os

from langchain_openai import ChatOpenAI

openai_model = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ["OPENAI_API_KEY"])
model = openai_model.bind_tools(tools)
model_response = model.invoke(
    "Get me an ad for clothing. I am a young man looking to go out with friends."
)
print("Tool call:", model_response)


# ## API reference

# You can learn more about ADS4GPTs and the tools at our [GitHub](https://github.com/ADS4GPTs/ads4gpts/tree/main)
