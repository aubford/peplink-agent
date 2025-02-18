#!/usr/bin/env python
# coding: utf-8

# # YouTube audio
#
# Building chat or QA applications on YouTube videos is a topic of high interest.
#
# Below we show how to easily go from a `YouTube url` to `audio of the video` to `text` to `chat`!
#
# We wil use the `OpenAIWhisperParser`, which will use the OpenAI Whisper API to transcribe audio to text,
# and the  `OpenAIWhisperParserLocal` for local support and running on private clouds or on premise.
#
# Note: You will need to have an `OPENAI_API_KEY` supplied.

# In[ ]:


from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import (
    OpenAIWhisperParser,
    OpenAIWhisperParserLocal,
)


# We will use `yt_dlp` to download audio for YouTube urls.
#
# We will use `pydub` to split downloaded audio files (such that we adhere to Whisper API's 25MB file size limit).

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  yt_dlp")
get_ipython().run_line_magic("pip", "install --upgrade --quiet  pydub")
get_ipython().run_line_magic("pip", "install --upgrade --quiet  librosa")


# ### YouTube url to text
#
# Use `YoutubeAudioLoader` to fetch / download the audio files.
#
# Then, ues `OpenAIWhisperParser()` to transcribe them to text.
#
# Let's take the first lecture of Andrej Karpathy's YouTube course as an example!

# In[ ]:


# set a flag to switch between local and remote parsing
# change this to True if you want to use local parsing
local = False


# In[ ]:


# Two Karpathy lecture videos
urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]

# Directory to save audio files
save_dir = "~/Downloads/YouTube"

# Transcribe the videos to text
if local:
    loader = GenericLoader(
        YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParserLocal()
    )
else:
    loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
docs = loader.load()


# In[ ]:


# Returns a list of Documents, which can be easily viewed or parsed
docs[0].page_content[0:500]


# ### Building a chat app from YouTube video
#
# Given `Documents`, we can easily enable chat / question+answering.

# In[ ]:


from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# In[ ]:


# Combine doc
combined_docs = [doc.page_content for doc in docs]
text = " ".join(combined_docs)


# In[ ]:


# Split them
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(text)


# In[ ]:


# Build an index
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_texts(splits, embeddings)


# In[ ]:


# Build a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)


# In[ ]:


# Ask a question!
query = "Why do we need to zero out the gradient before backprop at each step?"
qa_chain.run(query)


# In[ ]:


query = "What is the difference between an encoder and decoder?"
qa_chain.run(query)


# In[ ]:


query = "For any token, what are x, k, v, and q?"
qa_chain.run(query)
