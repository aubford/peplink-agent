#!/usr/bin/env python
# coding: utf-8
---
keywords: [gemini, vertex, VertexAI, gemini-pro]
---
# # Google Cloud Vertex AI
# 
# :::caution
# You are currently on a page documenting the use of Google Vertex [text completion models](/docs/concepts/text_llms). Many Google models are [chat completion models](/docs/concepts/chat_models).
# 
# You may be looking for [this page instead](/docs/integrations/chat/google_vertex_ai_palm/).
# :::
# 
# **Note:** This is separate from the `Google Generative AI` integration, it exposes [Vertex AI Generative API](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) on `Google Cloud`.
# 
# VertexAI exposes all foundational models available in google cloud:
# - Gemini for Text ( `gemini-1.0-pro` )
# - Gemini with Multimodality ( `gemini-1.5-pro-001` and `gemini-pro-vision`)
# - Palm 2 for Text (`text-bison`)
# - Codey for Code Generation (`code-bison`)
# 
# For a full and updated list of available models visit [VertexAI documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/overview)

# ## Setup

# By default, Google Cloud [does not use](https://cloud.google.com/vertex-ai/docs/generative-ai/data-governance#foundation_model_development) customer data to train its foundation models as part of Google Cloud's AI/ML Privacy Commitment. More details about how Google processes data can also be found in [Google's Customer Data Processing Addendum (CDPA)](https://cloud.google.com/terms/data-processing-addendum).
# 
# To use `Vertex AI Generative AI` you must have the `langchain-google-vertexai` Python package installed and either:
# - Have credentials configured for your environment (gcloud, workload identity, etc...)
# - Store the path to a service account JSON file as the GOOGLE_APPLICATION_CREDENTIALS environment variable
# 
# This codebase uses the `google.auth` library which first looks for the application credentials variable mentioned above, and then looks for system-level auth.
# 
# For more information, see:
# - https://cloud.google.com/docs/authentication/application-default-credentials#GAC
# - https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain-core langchain-google-vertexai')


# ## Usage
# 
# VertexAI supports all [LLM](/docs/how_to#llms) functionality.

# In[12]:


from langchain_google_vertexai import VertexAI

# To use model
model = VertexAI(model_name="gemini-pro")


# NOTE : You can also specify a [Gemini Version](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning#gemini-model-versions)

# In[2]:


# To specify a particular model version
model = VertexAI(model_name="gemini-1.0-pro-002")


# In[19]:


message = "What are some of the pros and cons of Python as a programming language?"
model.invoke(message)


# In[4]:


await model.ainvoke(message)


# In[5]:


for chunk in model.stream(message):
    print(chunk, end="", flush=True)


# In[ ]:


model.batch([message])


# We can use the `generate` method to get back extra metadata like [safety attributes](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/responsible-ai#safety_attribute_confidence_scoring) and not just text completions.

# In[6]:


result = model.generate([message])
result.generations


# ### OPTIONAL : Managing [Safety Attributes](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/responsible-ai#safety_attribute_confidence_scoring)
# - If your use case requires your to manage thresholds for saftey attributes, you can do so using below snippets
# >NOTE : We recommend exercising extreme caution when adjusting Safety Attributes thresholds

# In[16]:


from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

llm = VertexAI(model_name="gemini-1.0-pro-001", safety_settings=safety_settings)

# invoke a model response
output = llm.invoke(["How to make a molotov cocktail?"])
output


# In[17]:


# You may also pass safety_settings to generate method
llm = VertexAI(model_name="gemini-1.0-pro-001")

# invoke a model response
output = llm.invoke(
    ["How to make a molotov cocktail?"], safety_settings=safety_settings
)
output


# In[21]:


result = await model.ainvoke([message])
result


# You can also easily combine with a prompt template for easy structuring of user input. We can do this using [LCEL](/docs/concepts/lcel)

# In[ ]:


from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | model

question = """
I have five apples. I throw two away. I eat one. How many apples do I have left?
"""
print(chain.invoke({"question": question}))


# You can use different foundational models for specialized in different tasks. 
# For an updated list of available models visit [VertexAI documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/overview)

# In[ ]:


llm = VertexAI(model_name="code-bison", max_tokens=1000, temperature=0.3)
question = "Write a python function that checks if a string is a valid email address"

# invoke a model response
print(model.invoke(question))


# ## Multimodality

# With Gemini, you can use LLM in a multimodal mode:

# In[45]:


from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-pro-vision")

# Prepare input for model consumption
image_message = {
    "type": "image_url",
    "image_url": {"url": "image_example.jpg"},
}
text_message = {
    "type": "text",
    "text": "What is shown in this image?",
}

message = HumanMessage(content=[text_message, image_message])

# invoke a model response
output = llm.invoke([message])
print(output.content)


# Let's double-check it's a cat :)

# In[ ]:


from vertexai.preview.generative_models import Image

i = Image.load_from_file("image_example.jpg")
i


# You can also pass images as bytes:

# In[46]:


import base64

with open("image_example.jpg", "rb") as image_file:
    image_bytes = image_file.read()

image_message = {
    "type": "image_url",
    "image_url": {
        "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    },
}
text_message = {
    "type": "text",
    "text": "What is shown in this image?",
}

# Prepare input for model consumption
message = HumanMessage(content=[text_message, image_message])

# invoke a model response
output = llm.invoke([message])
print(output.content)


# Please, note that you can also use the image stored in GCS (just point the `url` to the full GCS path, starting with `gs://` instead of a local one).

# And you can also pass a history of a previous chat to the LLM:

# In[ ]:


# Prepare input for model consumption
message2 = HumanMessage(content="And where the image is taken?")

# invoke a model response
output2 = llm.invoke([message, output, message2])
print(output2.content)


# You can also use the public image URL:

# In[53]:


image_message = {
    "type": "image_url",
    "image_url": {
        "url": "gs://github-repo/img/vision/google-cloud-next.jpeg",
    },
}
text_message = {
    "type": "text",
    "text": "What is shown in this image?",
}

# Prepare input for model consumption
message = HumanMessage(content=[text_message, image_message])

# invoke a model response
output = llm.invoke([message])
print(output.content)


# ### Using Pdfs with Gemini Models

# In[12]:


from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

# Use Gemini 1.5 Pro
llm = ChatVertexAI(model="gemini-1.5-pro-001")


# In[13]:


# Prepare input for model consumption
pdf_message = {
    "type": "image_url",
    "image_url": {"url": "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"},
}

text_message = {
    "type": "text",
    "text": "Summarize the provided document.",
}

message = HumanMessage(content=[text_message, pdf_message])


# In[14]:


# invoke a model response
llm.invoke([message])


# ### Using Video with Gemini Models

# In[15]:


from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

# Use Gemini 1.5 Pro
llm = ChatVertexAI(model="gemini-1.5-pro-001")


# In[18]:


# Prepare input for model consumption
media_message = {
    "type": "image_url",
    "image_url": {
        "url": "gs://cloud-samples-data/generative-ai/video/pixel8.mp4",
    },
}

text_message = {
    "type": "text",
    "text": """Provide a description of the video.""",
}

message = HumanMessage(content=[media_message, text_message])


# In[19]:


# invoke a model response
llm.invoke([message])


# ### Using Audio with Gemini 1.5 Pro

# In[20]:


from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

# Use Gemini 1.5 Pro
llm = ChatVertexAI(model="gemini-1.5-pro-001")


# In[24]:


# Prepare input for model consumption
media_message = {
    "type": "image_url",
    "image_url": {
        "url": "gs://cloud-samples-data/generative-ai/audio/pixel.mp3",
    },
}

text_message = {
    "type": "text",
    "text": """Can you transcribe this interview, in the format of timecode, speaker, caption.
  Use speaker A, speaker B, etc. to identify speakers.""",
}

message = HumanMessage(content=[media_message, text_message])


# In[25]:


# invoke a model response
llm.invoke([message])


# ## Vertex Model Garden

# Vertex Model Garden [exposes](https://cloud.google.com/vertex-ai/docs/start/explore-models) open-sourced models that can be deployed and served on Vertex AI. 
# 
# Hundreds popular [open-sourced models](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models#oss-models) like Llama, Falcon and are available for  [One Click Deployment](https://cloud.google.com/vertex-ai/generative-ai/docs/deploy/overview)
# 
# If you have successfully deployed a model from Vertex Model Garden, you can find a corresponding Vertex AI [endpoint](https://cloud.google.com/vertex-ai/docs/general/deployment#what_happens_when_you_deploy_a_model) in the console or via API.

# In[6]:


from langchain_google_vertexai import VertexAIModelGarden


# In[ ]:


llm = VertexAIModelGarden(project="YOUR PROJECT", endpoint_id="YOUR ENDPOINT_ID")


# In[ ]:


# invoke a model response
llm.invoke("What is the meaning of life?")


# Like all LLMs, we can then compose it with other components:

# In[ ]:


prompt = PromptTemplate.from_template("What is the meaning of {thing}?")


# In[ ]:


chain = prompt | llm
print(chain.invoke({"thing": "life"}))


# ### Llama on Vertex Model Garden 
# 
# > Llama is a family of open weight models developed by Meta that you can fine-tune and deploy on Vertex AI. Llama models are pre-trained and fine-tuned generative text models. You can deploy Llama 2 and Llama 3 models on Vertex AI.
# [Official documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-llama) for more information about Llama on [Vertex Model Garden](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models)
# 

# To use Llama on Vertex Model Garden you must first [deploy it to Vertex AI Endpoint](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models#deploy-a-model)

# In[6]:


from langchain_google_vertexai import VertexAIModelGarden


# In[7]:


# TODO : Add "YOUR PROJECT" and "YOUR ENDPOINT_ID"
llm = VertexAIModelGarden(project="YOUR PROJECT", endpoint_id="YOUR ENDPOINT_ID")


# In[8]:


# invoke a model response
llm.invoke("What is the meaning of life?")


# Like all LLMs, we can then compose it with other components:

# In[9]:


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is the meaning of {thing}?")


# In[10]:


# invoke a model response using chain
chain = prompt | llm
print(chain.invoke({"thing": "life"}))


# ### Falcon on Vertex Model Garden 

# > Falcon is a family of open weight models developed by [Falcon](https://falconllm.tii.ae/) that you can fine-tune and deploy on Vertex AI. Falcon models are pre-trained and fine-tuned generative text models.

# To use Falcon on Vertex Model Garden you must first [deploy it to Vertex AI Endpoint](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models#deploy-a-model)

# In[13]:


from langchain_google_vertexai import VertexAIModelGarden


# In[14]:


# TODO : Add "YOUR PROJECT" and "YOUR ENDPOINT_ID"
llm = VertexAIModelGarden(project="YOUR PROJECT", endpoint_id="YOUR ENDPOINT_ID")


# In[15]:


# invoke a model response
llm.invoke("What is the meaning of life?")


# Like all LLMs, we can then compose it with other components:

# In[2]:


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is the meaning of {thing}?")


# In[17]:


chain = prompt | llm
print(chain.invoke({"thing": "life"}))


# ### Gemma on Vertex AI Model Garden

# > [Gemma](https://ai.google.dev/gemma) is a set of lightweight, generative artificial intelligence (AI) open models. Gemma models are available to run in your applications and on your hardware, mobile devices, or hosted services. You can also customize these models using tuning techniques so that they excel at performing tasks that matter to you and your users. Gemma models are based on [Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/overview) models and are intended for the AI development community to extend and take further.

# To use Gemma on Vertex Model Garden you must first [deploy it to Vertex AI Endpoint](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models#deploy-a-model)

# In[22]:


from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_google_vertexai import (
    GemmaChatVertexAIModelGarden,
    GemmaVertexAIModelGarden,
)


# In[21]:


# TODO : Add "YOUR PROJECT" , "YOUR REGION" and "YOUR ENDPOINT_ID"
llm = GemmaVertexAIModelGarden(
    endpoint_id="YOUR PROJECT",
    project="YOUR ENDPOINT_ID",
    location="YOUR REGION",
)

# invoke a model response
llm.invoke("What is the meaning of life?")


# In[23]:


# TODO : Add "YOUR PROJECT" , "YOUR REGION" and "YOUR ENDPOINT_ID"
chat_llm = GemmaChatVertexAIModelGarden(
    endpoint_id="YOUR PROJECT",
    project="YOUR ENDPOINT_ID",
    location="YOUR REGION",
)


# In[26]:


# Prepare input for model consumption
text_question1 = "How much is 2+2?"
message1 = HumanMessage(content=text_question1)

# invoke a model response
chat_llm.invoke([message1])


# ## Anthropic on Vertex AI

# > [Anthropic Claude 3](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude) models on Vertex AI offer fully managed and serverless models as APIs. To use a Claude model on Vertex AI, send a request directly to the Vertex AI API endpoint. Because Anthropic Claude 3 models use a managed API, there's no need to provision or manage infrastructure.

# NOTE : Anthropic Models on Vertex are implemented as Chat Model through class `ChatAnthropicVertex`

# In[ ]:


get_ipython().system('pip install -U langchain-google-vertexai anthropic[vertex]')


# In[3]:


from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import LLMResult
from langchain_google_vertexai.model_garden import ChatAnthropicVertex


# NOTE : Specify the correct [Claude 3 Model Versions](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#claude-opus)
# - For Claude 3 Opus (Preview), use `claude-3-opus@20240229`.
# - For Claude 3 Sonnet, use `claude-3-sonnet@20240229`.
# - For Claude 3 Haiku, use `claude-3-haiku@20240307`.
# 
# We don't recommend using the Anthropic Claude 3 model versions that don't include a suffix that starts with an @ symbol (claude-3-opus, claude-3-sonnet, or claude-3-haiku).

# In[5]:


# TODO : Replace below with your project id and region
project = "<project_id>"
location = "<region>"

# Initialise the Model
model = ChatAnthropicVertex(
    model_name="claude-3-haiku@20240307",
    project=project,
    location=location,
)


# In[6]:


# prepare input data for the model
raw_context = (
    "My name is Peter. You are my personal assistant. My favorite movies "
    "are Lord of the Rings and Hobbit."
)
question = (
    "Hello, could you recommend a good movie for me to watch this evening, please?"
)
context = SystemMessage(content=raw_context)
message = HumanMessage(content=question)


# In[7]:


# Invoke the model
response = model.invoke([context, message])
print(response.content)


# In[8]:


# You can choose to initialize/ override the model name on Invoke method as well
response = model.invoke([context, message], model_name="claude-3-sonnet@20240229")
print(response.content)


# In[ ]:


# Use streaming responses
sync_response = model.stream([context, message], model_name="claude-3-haiku@20240307")
for chunk in sync_response:
    print(chunk.content)

