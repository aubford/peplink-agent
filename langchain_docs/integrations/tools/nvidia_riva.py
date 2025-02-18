#!/usr/bin/env python
# coding: utf-8

# # NVIDIA Riva: ASR and TTS
#
# ## NVIDIA Riva
# [NVIDIA Riva](https://www.nvidia.com/en-us/ai-data-science/products/riva/) is a GPU-accelerated multilingual speech and translation AI software development kit for building fully customizable, real-time conversational AI pipelines—including automatic speech recognition (ASR), text-to-speech (TTS), and neural machine translation (NMT) applications—that can be deployed in clouds, in data centers, at the edge, or on embedded devices.
#
# The Riva Speech API server exposes a simple API for performing speech recognition, speech synthesis, and a variety of natural language processing inferences and is integrated into LangChain for ASR and TTS. See instructions on how to [setup a Riva Speech API](#3-setup) server below.
#
# ## Integrating NVIDIA Riva to LangChain Chains
# The `NVIDIARivaASR`, `NVIDIARivaTTS` utility runnables are LangChain runnables that integrate [NVIDIA Riva](https://www.nvidia.com/en-us/ai-data-science/products/riva/) into LCEL chains for Automatic Speech Recognition (ASR) and Text To Speech (TTS).
#
# This example goes over how to use these LangChain runnables to:
# 1. Accept streamed audio,
# 2. convert the audio to text,
# 3. send the text to an LLM,
# 4. stream a textual LLM response, and
# 5. convert the response to streamed human-sounding audio.

# ## 1. NVIDIA Riva Runnables
# There are 2 Riva Runnables:
#
# a. **RivaASR**: Converts audio bytes into text for an LLM using NVIDIA Riva.
#
# b. **RivaTTS**: Converts text into audio bytes using NVIDIA Riva.
#
# ### a. RivaASR
# The [**RivaASR**](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/nvidia_riva.py#L404) runnable converts audio bytes into a string for an LLM using NVIDIA Riva.
#
# It's useful for sending an audio stream (a message containing streaming audio) into a chain and preprocessing that audio by converting it to a string to create an LLM prompt.
#
# ```
# ASRInputType = AudioStream # the AudioStream type is a custom type for a message queue containing streaming audio
# ASROutputType = str
#
# class RivaASR(
#     RivaAuthMixin,
#     RivaCommonConfigMixin,
#     RunnableSerializable[ASRInputType, ASROutputType],
# ):
#     """A runnable that performs Automatic Speech Recognition (ASR) using NVIDIA Riva."""
#
#     name: str = "nvidia_riva_asr"
#     description: str = (
#         "A Runnable for converting audio bytes to a string."
#         "This is useful for feeding an audio stream into a chain and"
#         "preprocessing that audio to create an LLM prompt."
#     )
#
#     # riva options
#     audio_channel_count: int = Field(
#         1, description="The number of audio channels in the input audio stream."
#     )
#     profanity_filter: bool = Field(
#         True,
#         description=(
#             "Controls whether or not Riva should attempt to filter "
#             "profanity out of the transcribed text."
#         ),
#     )
#     enable_automatic_punctuation: bool = Field(
#         True,
#         description=(
#             "Controls whether Riva should attempt to correct "
#             "senetence puncuation in the transcribed text."
#         ),
#     )
# ```
#
# When this runnable is called on an input, it takes an input audio stream that acts as a queue and concatenates transcription as chunks are returned.After a response is fully generated, a string is returned.
# * Note that since the LLM requires a full query the ASR is concatenated and not streamed in token-by-token.
#
#
# ### b. RivaTTS
# The [**RivaTTS**](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/nvidia_riva.py#L511) runnable converts text output to audio bytes.
#
# It's useful for processing the streamed textual response from an LLM by converting the text to audio bytes. These audio bytes sound like a natural human voice to be played back to the user.
#
# ```
# TTSInputType = Union[str, AnyMessage, PromptValue]
# TTSOutputType = byte
#
# class RivaTTS(
#     RivaAuthMixin,
#     RivaCommonConfigMixin,
#     RunnableSerializable[TTSInputType, TTSOutputType],
# ):
#     """A runnable that performs Text-to-Speech (TTS) with NVIDIA Riva."""
#
#     name: str = "nvidia_riva_tts"
#     description: str = (
#         "A tool for converting text to speech."
#         "This is useful for converting LLM output into audio bytes."
#     )
#
#     # riva options
#     voice_name: str = Field(
#         "English-US.Female-1",
#         description=(
#             "The voice model in Riva to use for speech. "
#             "Pre-trained models are documented in "
#             "[the Riva documentation]"
#             "(https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html)."
#         ),
#     )
#     output_directory: Optional[str] = Field(
#         None,
#         description=(
#             "The directory where all audio files should be saved. "
#             "A null value indicates that wave files should not be saved. "
#             "This is useful for debugging purposes."
#         ),
# ```
#
# When this runnable is called on an input, it takes iterable text chunks and streams them into output audio bytes that are either written to a `.wav` file or played out loud.

# ## 2. Installation

# The NVIDIA Riva client library must be installed.

# In[1]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet nvidia-riva-client")


# ## 3. Setup
#
# **To get started with NVIDIA Riva:**
#
# 1. Follow the Riva Quick Start setup instructions for [Local Deployment Using Quick Start Scripts](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html#local-deployment-using-quick-start-scripts).

# ## 4. Import and Inspect Runnables
# Import the RivaASR and RivaTTS runnables and inspect their schemas to understand their fields.

# In[2]:


import json

from langchain_community.utilities.nvidia_riva import (
    RivaASR,
    RivaTTS,
)


# Let's view the schemas.

# In[3]:


print(json.dumps(RivaASR.schema(), indent=2))
print(json.dumps(RivaTTS.schema(), indent=2))


# ##  5. Declare Riva ASR and Riva TTS Runnables
#
# For this example, a single-channel audio file (mulaw format, so `.wav`) is used.
#
# You will need a Riva speech server setup, so if you don't have a Riva speech server, go to [Setup](#3-setup).
#
# ### a. Set Audio Parameters
# Some parameters of audio can be inferred by the mulaw file, but others are set explicitly.
#
# Replace `audio_file` with the path of your audio file.

# In[4]:


import pywav  # pywav is used instead of built-in wave because of mulaw support
from langchain_community.utilities.nvidia_riva import RivaAudioEncoding

audio_file = "./audio_files/en-US_sample2.wav"
wav_file = pywav.WavRead(audio_file)
audio_data = wav_file.getdata()
audio_encoding = RivaAudioEncoding.from_wave_format_code(wav_file.getaudioformat())
sample_rate = wav_file.getsamplerate()
delay_time = 1 / 4
chunk_size = int(sample_rate * delay_time)
delay_time = 1 / 8
num_channels = wav_file.getnumofchannels()


# In[5]:


import IPython

IPython.display.Audio(audio_file)


# ### b. Set the Speech Server and Declare Riva LangChain Runnables
#
# Be sure to set `RIVA_SPEECH_URL` to be the URI of your Riva speech server.
#
# The runnables act as clients to the speech server. Many of the fields set in this example are configured based on the sample audio data.

# In[6]:


RIVA_SPEECH_URL = "http://localhost:50051/"

riva_asr = RivaASR(
    url=RIVA_SPEECH_URL,  # the location of the Riva ASR server
    encoding=audio_encoding,
    audio_channel_count=num_channels,
    sample_rate_hertz=sample_rate,
    profanity_filter=True,
    enable_automatic_punctuation=True,
    language_code="en-US",
)

riva_tts = RivaTTS(
    url=RIVA_SPEECH_URL,  # the location of the Riva TTS server
    output_directory="./scratch",  # location of the output .wav files
    language_code="en-US",
    voice_name="English-US.Female-1",
)


# ## 6. Create Additional Chain Components
# As usual, declare the other parts of the chain. In this case, it's just a prompt template and an LLM.
#
# You can use any [LangChain compatible LLM](https://python.langchain.com/v0.1/docs/integrations/llms/) in the chain. In this example, we use a [Mixtral8x7b NIM from NVIDIA](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/). NVIDIA NIMs are supported in LangChain via the `langchain-nvidia-ai-endpoints` package, so you can easily build applications with best in class throughput and latency.
#
# LangChain compatible NVIDIA LLMs from [NVIDIA AI Foundation Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/) can also be used by following these [instructions](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints).

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet langchain-nvidia-ai-endpoints"
)


# Follow the [instructions for LangChain](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/) to use NVIDIA NIM in your speech-enabled LangChain application.
#
# Set your key for NVIDIA API catalog, where NIMs are hosted for you to try.

# In[ ]:


import getpass
import os

nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
os.environ["NVIDIA_API_KEY"] = nvapi_key


# Instantiate LLM.

# In[7]:


from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

prompt = PromptTemplate.from_template("{user_input}")

llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")


# Now, tie together all the parts of the chain including RivaASR and RivaTTS.

# In[8]:


chain = {"user_input": riva_asr} | prompt | llm | riva_tts


# ## 7. Run the Chain with Streamed Inputs and Outputs
#
# ### a. Mimic Audio Streaming
# To mimic streaming, first convert the processed audio data to iterable chunks of audio bytes.
#
# Two functions, `producer` and `consumer`, respectively handle asynchronously passing audio data into the chain and consuming audio data out of the chain.
#

# In[9]:


import asyncio

from langchain_community.utilities.nvidia_riva import AudioStream

audio_chunks = [
    audio_data[0 + i : chunk_size + i] for i in range(0, len(audio_data), chunk_size)
]


async def producer(input_stream) -> None:
    """Produces audio chunk bytes into an AudioStream as streaming audio input."""
    for chunk in audio_chunks:
        await input_stream.aput(chunk)
    input_stream.close()


async def consumer(input_stream, output_stream) -> None:
    """
    Consumes audio chunks from input stream and passes them along the chain
    constructed comprised of ASR -> text based prompt for an LLM -> TTS chunks
    with synthesized voice of LLM response put in an output stream.
    """
    while not input_stream.complete:
        async for chunk in chain.astream(input_stream):
            await output_stream.put(
                chunk
            )  # for production code don't forget to add a timeout


input_stream = AudioStream(maxsize=1000)
output_stream = asyncio.Queue()

# send data into the chain
producer_task = asyncio.create_task(producer(input_stream))
# get data out of the chain
consumer_task = asyncio.create_task(consumer(input_stream, output_stream))

while not consumer_task.done():
    try:
        generated_audio = await asyncio.wait_for(
            output_stream.get(), timeout=2
        )  # for production code don't forget to add a timeout
    except asyncio.TimeoutError:
        continue

await producer_task
await consumer_task


# ## 8. Listen to Voice Response
#
# The audio response is written to `./scratch` and should contain an audio clip that is a response to the input audio.

# In[10]:


import glob
import os

output_path = os.path.join(os.getcwd(), "scratch")
file_type = "*.wav"
files_path = os.path.join(output_path, file_type)
files = glob.glob(files_path)

IPython.display.Audio(files[0])
