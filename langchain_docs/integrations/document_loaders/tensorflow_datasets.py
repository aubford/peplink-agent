#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Datasets
#
# >[TensorFlow Datasets](https://www.tensorflow.org/datasets) is a collection of datasets ready to use, with TensorFlow or other Python ML frameworks, such as Jax. All datasets are exposed as [tf.data.Datasets](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), enabling easy-to-use and high-performance input pipelines. To get started see the [guide](https://www.tensorflow.org/datasets/overview) and the [list of datasets](https://www.tensorflow.org/datasets/catalog/overview#all_datasets).
#
# This notebook shows how to load `TensorFlow Datasets` into a Document format that we can use downstream.

# ## Installation

# You need to install `tensorflow` and `tensorflow-datasets` python packages.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  tensorflow")


# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  tensorflow-datasets")


# ## Example

# As an example, we use the [`mlqa/en` dataset](https://www.tensorflow.org/datasets/catalog/mlqa#mlqaen).
#
# >`MLQA` (`Multilingual Question Answering Dataset`) is a benchmark dataset for evaluating multilingual question answering performance. The dataset consists of 7 languages: Arabic, German, Spanish, English, Hindi, Vietnamese, Chinese.
# >
# >- Homepage: https://github.com/facebookresearch/MLQA
# >- Source code: `tfds.datasets.mlqa.Builder`
# >- Download size: 72.21 MiB
#

# In[ ]:


# Feature structure of `mlqa/en` dataset:

FeaturesDict(
    {
        "answers": Sequence(
            {
                "answer_start": int32,
                "text": Text(shape=(), dtype=string),
            }
        ),
        "context": Text(shape=(), dtype=string),
        "id": string,
        "question": Text(shape=(), dtype=string),
        "title": Text(shape=(), dtype=string),
    }
)


# In[18]:


import tensorflow as tf
import tensorflow_datasets as tfds


# In[78]:


# try directly access this dataset:
ds = tfds.load("mlqa/en", split="test")
ds = ds.take(1)  # Only take a single example
ds


# Now we have to create a custom function to convert dataset sample into a Document.
#
# This is a requirement. There is no standard format for the TF datasets that's why we need to make a custom transformation function.
#
# Let's use `context` field as the `Document.page_content` and place other fields in the `Document.metadata`.
#

# In[72]:


from langchain_core.documents import Document


def decode_to_str(item: tf.Tensor) -> str:
    return item.numpy().decode("utf-8")


def mlqaen_example_to_document(example: dict) -> Document:
    return Document(
        page_content=decode_to_str(example["context"]),
        metadata={
            "id": decode_to_str(example["id"]),
            "title": decode_to_str(example["title"]),
            "question": decode_to_str(example["question"]),
            "answer": decode_to_str(example["answers"]["text"][0]),
        },
    )


for example in ds:
    doc = mlqaen_example_to_document(example)
    print(doc)
    break


# In[73]:


from langchain_community.document_loaders import TensorflowDatasetLoader
from langchain_core.documents import Document

loader = TensorflowDatasetLoader(
    dataset_name="mlqa/en",
    split_name="test",
    load_max_docs=3,
    sample_to_document_function=mlqaen_example_to_document,
)


# `TensorflowDatasetLoader` has these parameters:
# - `dataset_name`: the name of the dataset to load
# - `split_name`: the name of the split to load. Defaults to "train".
# - `load_max_docs`: a limit to the number of loaded documents. Defaults to 100.
# - `sample_to_document_function`: a function that converts a dataset sample to a Document
#

# In[74]:


docs = loader.load()
len(docs)


# In[76]:


docs[0].page_content


# In[77]:


docs[0].metadata


# In[ ]:
