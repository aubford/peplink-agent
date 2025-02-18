#!/usr/bin/env python
# coding: utf-8

# # Dall-E Image Generator
#
# >[OpenAI Dall-E](https://openai.com/dall-e-3) are text-to-image models developed by `OpenAI` using deep learning methodologies to generate digital images from natural language descriptions, called "prompts".
#
# This notebook shows how you can generate images from a prompt synthesized using an OpenAI LLM. The images are generated using `Dall-E`, which uses the same OpenAI API key as the LLM.

# In[ ]:


# Needed if you would like to display images in the notebook
get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  opencv-python scikit-image langchain-community"
)


# In[4]:


import os

from langchain_openai import OpenAI

os.environ["OPENAI_API_KEY"] = "<your-key-here>"


# ## Run as a chain

# In[2]:


from langchain.chains import LLMChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template="Generate a detailed prompt to generate an image based on the following description: {image_desc}",
)
chain = LLMChain(llm=llm, prompt=prompt)


# In[3]:


image_url = DallEAPIWrapper().run(chain.run("halloween night at a haunted museum"))


# In[4]:


image_url


# In[ ]:


# You can click on the link above to display the image
# Or you can try the options below to display the image inline in this notebook

try:
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab.patches import cv2_imshow  # for image display
    from skimage import io

    image = io.imread(image_url)
    cv2_imshow(image)
else:
    import cv2
    from skimage import io

    image = io.imread(image_url)
    cv2.imshow("image", image)
    cv2.waitKey(0)  # wait for a keyboard input
    cv2.destroyAllWindows()


# ## Run as a tool with an agent

# In[9]:


from langchain.agents import initialize_agent, load_tools

tools = load_tools(["dalle-image-generator"])
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
output = agent.run("Create an image of a halloween night at a haunted museum")
