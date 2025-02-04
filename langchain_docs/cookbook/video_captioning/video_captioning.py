#!/usr/bin/env python
# coding: utf-8

# # Video Captioning
# This notebook shows how to use VideoCaptioningChain, which is implemented using Langchain's ImageCaptionLoader and AssemblyAI to produce .srt files.
# 
# This system autogenerates both subtitles and closed captions from a video URL.

# ## Installing Dependencies

# In[1]:


# !pip install ffmpeg-python
# !pip install assemblyai
# !pip install opencv-python
# !pip install torch
# !pip install pillow
# !pip install transformers
# !pip install langchain


# ## Imports

# In[1]:


import getpass

from langchain.chains.video_captioning import VideoCaptioningChain
from langchain.chat_models.openai import ChatOpenAI


# ## Setting up API Keys

# In[2]:


OPENAI_API_KEY = getpass.getpass("OpenAI API Key:")

ASSEMBLYAI_API_KEY = getpass.getpass("AssemblyAI API Key:")


# **Required parameters:**
# 
# * llm: The language model this chain will use to get suggestions on how to refine the closed-captions
# * assemblyai_key: The API key for AssemblyAI, used to generate the subtitles
# 
# **Optional Parameters:**
# 
# * verbose (Default: True): Sets verbose mode for downstream chain calls
# * use_logging (Default: True): Log the chain's processes in run manager
# * frame_skip (Default: None): Choose how many video frames to skip during processing. Increasing it results in faster execution, but less accurate results. If None, frame skip is calculated manually based on the framerate Set this to 0 to sample all frames
# * image_delta_threshold (Default: 3000000): Set the sensitivity for what the image processor considers a change in scenery in the video, used to delimit closed captions. Higher = less sensitive
# * closed_caption_char_limit (Default: 20): Sets the character limit on closed captions
# * closed_caption_similarity_threshold (Default: 80): Sets the percentage value to how similar two closed caption models should be in order to be clustered into one longer closed caption
# * use_unclustered_video_models (Default: False): If true, closed captions that could not be clustered will be included. May result in spontaneous behaviour from closed captions such as very short lasting captions or fast-changing captions. Enabling this is experimental and not recommended

# ## Example run

# In[ ]:


# https://ia804703.us.archive.org/27/items/uh-oh-here-we-go-again/Uh-Oh%2C%20Here%20we%20go%20again.mp4
# https://ia601200.us.archive.org/9/items/f58703d4-61e6-4f8f-8c08-b42c7e16f7cb/f58703d4-61e6-4f8f-8c08-b42c7e16f7cb.mp4

chain = VideoCaptioningChain(
    llm=ChatOpenAI(model="gpt-4", max_tokens=4000, openai_api_key=OPENAI_API_KEY),
    assemblyai_key=ASSEMBLYAI_API_KEY,
)

srt_content = chain.run(
    video_file_path="https://ia601200.us.archive.org/9/items/f58703d4-61e6-4f8f-8c08-b42c7e16f7cb/f58703d4-61e6-4f8f-8c08-b42c7e16f7cb.mp4"
)

print(srt_content)


# ## Writing output to .srt file

# In[6]:


with open("output.srt", "w") as file:
    file.write(srt_content)

