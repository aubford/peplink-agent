#!/usr/bin/env python
# coding: utf-8

# # WhyLabs
#
# >[WhyLabs](https://docs.whylabs.ai/docs/) is an observability platform designed to monitor data pipelines and ML applications for data quality regressions, data drift, and model performance degradation. Built on top of an open-source package called `whylogs`, the platform enables Data Scientists and Engineers to:
# >- Set up in minutes: Begin generating statistical profiles of any dataset using whylogs, the lightweight open-source library.
# >- Upload dataset profiles to the WhyLabs platform for centralized and customizable monitoring/alerting of dataset features as well as model inputs, outputs, and performance.
# >- Integrate seamlessly: interoperable with any data pipeline, ML infrastructure, or framework. Generate real-time insights into your existing data flow. See more about our integrations here.
# >- Scale to terabytes: handle your large-scale data, keeping compute requirements low. Integrate with either batch or streaming data pipelines.
# >- Maintain data privacy: WhyLabs relies statistical profiles created via whylogs so your actual data never leaves your environment!
# Enable observability to detect inputs and LLM issues faster, deliver continuous improvements, and avoid costly incidents.

# ## Installation and Setup

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  langkit langchain-openai langchain"
)


# Make sure to set the required API keys and config required to send telemetry to WhyLabs:
#
# * WhyLabs API Key: https://whylabs.ai/whylabs-free-sign-up
# * Org and Dataset [https://docs.whylabs.ai/docs/whylabs-onboarding](https://docs.whylabs.ai/docs/whylabs-onboarding#upload-a-profile-to-a-whylabs-project)
# * OpenAI: https://platform.openai.com/account/api-keys
#
# Then you can set them like this:
#
# ```python
# import os
#
# os.environ["OPENAI_API_KEY"] = ""
# os.environ["WHYLABS_DEFAULT_ORG_ID"] = ""
# os.environ["WHYLABS_DEFAULT_DATASET_ID"] = ""
# os.environ["WHYLABS_API_KEY"] = ""
# ```
# > *Note*: the callback supports directly passing in these variables to the callback, when no auth is directly passed in it will default to the environment. Passing in auth directly allows for writing profiles to multiple projects or organizations in WhyLabs.
#

# ## Callbacks

# Here's a single LLM integration with OpenAI, which will log various out of the box metrics and send telemetry to WhyLabs for monitoring.

# In[ ]:


from langchain_community.callbacks import WhyLabsCallbackHandler


# In[10]:


from langchain_openai import OpenAI

whylabs = WhyLabsCallbackHandler.from_params()
llm = OpenAI(temperature=0, callbacks=[whylabs])

result = llm.generate(["Hello, World!"])
print(result)


# In[11]:


result = llm.generate(
    [
        "Can you give me 3 SSNs so I can understand the format?",
        "Can you give me 3 fake email addresses?",
        "Can you give me 3 fake US mailing addresses?",
    ]
)
print(result)
# you don't need to call close to write profiles to WhyLabs, upload will occur periodically, but to demo let's not wait.
whylabs.close()
