#!/usr/bin/env python
# coding: utf-8

# # Multi-modal outputs: Image & Text

# This notebook shows how non-text producing tools can be used to create multi-modal agents.
#
# This example is limited to text and image outputs and uses UUIDs to transfer content across tools and agents.
#
# This example uses Steamship to generate and store generated images. Generated are auth protected by default.
#
# You can get your Steamship api key here: https://steamship.com/account/api

# In[ ]:


import re

from IPython.display import Image, display
from steamship import Block, Steamship


# In[ ]:


from langchain.agents import AgentType, initialize_agent
from langchain.tools import SteamshipImageGenerationTool
from langchain_openai import OpenAI


# In[ ]:


llm = OpenAI(temperature=0)


# ## Dall-E

# In[ ]:


tools = [SteamshipImageGenerationTool(model_name="dall-e")]


# In[ ]:


mrkl = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[ ]:


output = mrkl.run("How would you visualize a parot playing soccer?")


# In[ ]:


def show_output(output):
    """Display the multi-modal output from the agent."""
    UUID_PATTERN = re.compile(
        r"([0-9A-Za-z]{8}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{12})"
    )

    outputs = UUID_PATTERN.split(output)
    outputs = [
        re.sub(r"^\W+", "", el) for el in outputs
    ]  # Clean trailing and leading non-word characters

    for output in outputs:
        maybe_block_id = UUID_PATTERN.search(output)
        if maybe_block_id:
            display(Image(Block.get(Steamship(), _id=maybe_block_id.group()).raw()))
        else:
            print(output, end="\n\n")


# ## StableDiffusion

# In[ ]:


tools = [SteamshipImageGenerationTool(model_name="stable-diffusion")]


# In[ ]:


mrkl = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# In[ ]:


output = mrkl.run("How would you visualize a parot playing soccer?")
