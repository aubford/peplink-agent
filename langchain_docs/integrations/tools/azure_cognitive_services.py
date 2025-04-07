#!/usr/bin/env python
# coding: utf-8

# # Azure Cognitive Services Toolkit
# 
# This toolkit is used to interact with the `Azure Cognitive Services API` to achieve some multimodal capabilities.
# 
# Currently There are four tools bundled in this toolkit:
# - AzureCogsImageAnalysisTool: used to extract caption, objects, tags, and text from images. (Note: this tool is not available on Mac OS yet, due to the dependency on `azure-ai-vision` package, which is only supported on Windows and Linux currently.)
# - AzureCogsFormRecognizerTool: used to extract text, tables, and key-value pairs from documents.
# - AzureCogsSpeech2TextTool: used to transcribe speech to text.
# - AzureCogsText2SpeechTool: used to synthesize text to speech.
# - AzureCogsTextAnalyticsHealthTool: used to extract healthcare entities.

# First, you need to set up an Azure account and create a Cognitive Services resource. You can follow the instructions [here](https://docs.microsoft.com/en-us/azure/cognitive-services/cognitive-services-apis-create-account?tabs=multiservice%2Cwindows) to create a resource. 
# 
# Then, you need to get the endpoint, key and region of your resource, and set them as environment variables. You can find them in the "Keys and Endpoint" page of your resource.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  azure-ai-formrecognizer > /dev/null')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  azure-cognitiveservices-speech > /dev/null')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  azure-ai-textanalytics > /dev/null')

# For Windows/Linux
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  azure-ai-vision > /dev/null')


# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain-community')


# In[ ]:


import os

os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["AZURE_COGS_KEY"] = ""
os.environ["AZURE_COGS_ENDPOINT"] = ""
os.environ["AZURE_COGS_REGION"] = ""


# ## Create the Toolkit

# In[19]:


from langchain_community.agent_toolkits import AzureCognitiveServicesToolkit

toolkit = AzureCognitiveServicesToolkit()


# In[ ]:


[tool.name for tool in toolkit.get_tools()]


# ## Use within an Agent

# In[20]:


from langchain.agents import AgentType, initialize_agent
from langchain_openai import OpenAI


# In[21]:


llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# In[ ]:


agent.run(
    "What can I make with these ingredients?"
    "https://images.openai.com/blob/9ad5a2ab-041f-475f-ad6a-b51899c50182/ingredients.png"
)


# In[ ]:


audio_file = agent.run("Tell me a joke and read it out for me.")


# In[ ]:


from IPython import display

audio = display.Audio(audio_file)
display.display(audio)


# In[22]:


agent.run(
    """The patient is a 54-year-old gentleman with a history of progressive angina over the past several months.
The patient had a cardiac catheterization in July of this year revealing total occlusion of the RCA and 50% left main disease ,
with a strong family history of coronary artery disease with a brother dying at the age of 52 from a myocardial infarction and
another brother who is status post coronary artery bypass grafting. The patient had a stress echocardiogram done on July , 2001 ,
which showed no wall motion abnormalities , but this was a difficult study due to body habitus. The patient went for six minutes with
minimal ST depressions in the anterior lateral leads , thought due to fatigue and wrist pain , his anginal equivalent. Due to the patient's
increased symptoms and family history and history left main disease with total occasional of his RCA was referred for revascularization with open heart surgery.

List all the diagnoses.
"""
)


# In[ ]:




