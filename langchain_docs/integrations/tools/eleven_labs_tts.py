#!/usr/bin/env python
# coding: utf-8

# # ElevenLabs Text2Speech
# 
# This notebook shows how to interact with the `ElevenLabs API` to achieve text-to-speech capabilities.

# First, you need to set up an ElevenLabs account. You can follow the instructions [here](https://docs.elevenlabs.io/welcome/introduction).

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  elevenlabs langchain-community')


# In[ ]:


import os

os.environ["ELEVENLABS_API_KEY"] = ""


# ## Usage

# In[6]:


from langchain_community.tools import ElevenLabsText2SpeechTool

text_to_speak = "Hello world! I am the real slim shady"

tts = ElevenLabsText2SpeechTool()
tts.name


# We can generate audio, save it to the temporary file and then play it.

# In[7]:


speech_file = tts.run(text_to_speak)
tts.play(speech_file)


# Or stream audio directly.

# In[9]:


tts.stream_speech(text_to_speak)


# ## Use within an Agent

# In[12]:


from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI


# In[13]:


llm = OpenAI(temperature=0)
tools = load_tools(["eleven_labs_text2speech"])
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# In[14]:


audio_file = agent.run("Tell me a joke and read it out for me.")


# In[15]:


tts.play(audio_file)

