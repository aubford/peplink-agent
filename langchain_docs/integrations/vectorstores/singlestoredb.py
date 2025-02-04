#!/usr/bin/env python
# coding: utf-8

# # SingleStoreDB
# >[SingleStoreDB](https://singlestore.com/) is a robust, high-performance distributed SQL database solution designed to excel in both [cloud](https://www.singlestore.com/cloud/) and on-premises environments. Boasting a versatile feature set, it offers seamless deployment options while delivering unparalleled performance.
# 
# A standout feature of SingleStoreDB is its advanced support for vector storage and operations, making it an ideal choice for applications requiring intricate AI capabilities such as text similarity matching. With built-in vector functions like [dot_product](https://docs.singlestore.com/managed-service/en/reference/sql-reference/vector-functions/dot_product.html) and [euclidean_distance](https://docs.singlestore.com/managed-service/en/reference/sql-reference/vector-functions/euclidean_distance.html), SingleStoreDB empowers developers to implement sophisticated algorithms efficiently.
# 
# For developers keen on leveraging vector data within SingleStoreDB, a comprehensive tutorial is available, guiding them through the intricacies of [working with vector data](https://docs.singlestore.com/managed-service/en/developer-resources/functional-extensions/working-with-vector-data.html). This tutorial delves into the Vector Store within SingleStoreDB, showcasing its ability to facilitate searches based on vector similarity. Leveraging vector indexes, queries can be executed with remarkable speed, enabling swift retrieval of relevant data.
# 
# Moreover, SingleStoreDB's Vector Store seamlessly integrates with [full-text indexing based on Lucene](https://docs.singlestore.com/cloud/developer-resources/functional-extensions/working-with-full-text-search/), enabling powerful text similarity searches. Users can filter search results based on selected fields of document metadata objects, enhancing query precision.
# 
# What sets SingleStoreDB apart is its ability to combine vector and full-text searches in various ways, offering flexibility and versatility. Whether prefiltering by text or vector similarity and selecting the most relevant data, or employing a weighted sum approach to compute a final similarity score, developers have multiple options at their disposal.
# 
# In essence, SingleStoreDB provides a comprehensive solution for managing and querying vector data, offering unparalleled performance and flexibility for AI-driven applications.
# 
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

# In[ ]:


# Establishing a connection to the database is facilitated through the singlestoredb Python connector.
# Please ensure that this connector is installed in your working environment.
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  singlestoredb')


# In[ ]:


import getpass
import os

# We want to use OpenAIEmbeddings so we have to get the OpenAI API Key.
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[ ]:


from langchain_community.vectorstores import SingleStoreDB
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# In[ ]:


# loading docs
# we will use some artificial data for this example
docs = [
    Document(
        page_content="""In the parched desert, a sudden rainstorm brought relief,
            as the droplets danced upon the thirsty earth, rejuvenating the landscape
            with the sweet scent of petrichor.""",
        metadata={"category": "rain"},
    ),
    Document(
        page_content="""Amidst the bustling cityscape, the rain fell relentlessly,
            creating a symphony of pitter-patter on the pavement, while umbrellas
            bloomed like colorful flowers in a sea of gray.""",
        metadata={"category": "rain"},
    ),
    Document(
        page_content="""High in the mountains, the rain transformed into a delicate
            mist, enveloping the peaks in a mystical veil, where each droplet seemed to
            whisper secrets to the ancient rocks below.""",
        metadata={"category": "rain"},
    ),
    Document(
        page_content="""Blanketing the countryside in a soft, pristine layer, the
            snowfall painted a serene tableau, muffling the world in a tranquil hush
            as delicate flakes settled upon the branches of trees like nature's own 
            lacework.""",
        metadata={"category": "snow"},
    ),
    Document(
        page_content="""In the urban landscape, snow descended, transforming
            bustling streets into a winter wonderland, where the laughter of
            children echoed amidst the flurry of snowballs and the twinkle of
            holiday lights.""",
        metadata={"category": "snow"},
    ),
    Document(
        page_content="""Atop the rugged peaks, snow fell with an unyielding
            intensity, sculpting the landscape into a pristine alpine paradise,
            where the frozen crystals shimmered under the moonlight, casting a
            spell of enchantment over the wilderness below.""",
        metadata={"category": "snow"},
    ),
]

embeddings = OpenAIEmbeddings()


# There are several ways to establish a [connection](https://singlestoredb-python.labs.singlestore.com/generated/singlestoredb.connect.html) to the database. You can either set up environment variables or pass named parameters to the `SingleStoreDB constructor`. Alternatively, you may provide these parameters to the `from_documents` and `from_texts` methods.

# In[ ]:


# Setup connection url as environment variable
os.environ["SINGLESTOREDB_URL"] = "root:pass@localhost:3306/db"

# Load documents to the store
docsearch = SingleStoreDB.from_documents(
    docs,
    embeddings,
    table_name="notebook",  # use table with a custom name
)


# In[ ]:


query = "trees in the snow"
docs = docsearch.similarity_search(query)  # Find documents that correspond to the query
print(docs[0].page_content)


# SingleStoreDB elevates search capabilities by enabling users to enhance and refine search results through prefiltering based on metadata fields. This functionality empowers developers and data analysts to fine-tune queries, ensuring that search results are precisely tailored to their requirements. By filtering search results using specific metadata attributes, users can narrow down the scope of their queries, focusing only on relevant data subsets. 

# In[ ]:


query = "trees branches"
docs = docsearch.similarity_search(
    query, filter={"category": "snow"}
)  # Find documents that correspond to the query and has category "snow"
print(docs[0].page_content)


# Enhance your search efficiency with SingleStore DB version 8.5 or above by leveraging [ANN vector indexes](https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/). By setting `use_vector_index=True` during vector store object creation, you can activate this feature. Additionally, if your vectors differ in dimensionality from the default OpenAI embedding size of 1536, ensure to specify the `vector_size` parameter accordingly. 

# SingleStoreDB presents a diverse range of search strategies, each meticulously crafted to cater to specific use cases and user preferences. The default `VECTOR_ONLY` strategy utilizes vector operations such as `dot_product` or `euclidean_distance` to calculate similarity scores directly between vectors, while `TEXT_ONLY` employs Lucene-based full-text search, particularly advantageous for text-centric applications. For users seeking a balanced approach, `FILTER_BY_TEXT` first refines results based on text similarity before conducting vector comparisons, whereas `FILTER_BY_VECTOR` prioritizes vector similarity, filtering results before assessing text similarity for optimal matches. Notably, both `FILTER_BY_TEXT` and `FILTER_BY_VECTOR` necessitate a full-text index for operation. Additionally, `WEIGHTED_SUM` emerges as a sophisticated strategy, calculating the final similarity score by weighing vector and text similarities, albeit exclusively utilizing dot_product distance calculations and also requiring a full-text index. These versatile strategies empower users to fine-tune searches according to their unique needs, facilitating efficient and precise data retrieval and analysis. Moreover, SingleStoreDB's hybrid approaches, exemplified by `FILTER_BY_TEXT`, `FILTER_BY_VECTOR`, and `WEIGHTED_SUM` strategies, seamlessly blend vector and text-based searches to maximize efficiency and accuracy, ensuring users can fully leverage the platform's capabilities for a wide range of applications.

# In[ ]:


docsearch = SingleStoreDB.from_documents(
    docs,
    embeddings,
    distance_strategy=DistanceStrategy.DOT_PRODUCT,  # Use dot product for similarity search
    use_vector_index=True,  # Use vector index for faster search
    use_full_text_search=True,  # Use full text index
)

vectorResults = docsearch.similarity_search(
    "rainstorm in parched desert, rain",
    k=1,
    search_strategy=SingleStoreDB.SearchStrategy.VECTOR_ONLY,
    filter={"category": "rain"},
)
print(vectorResults[0].page_content)

textResults = docsearch.similarity_search(
    "rainstorm in parched desert, rain",
    k=1,
    search_strategy=SingleStoreDB.SearchStrategy.TEXT_ONLY,
)
print(textResults[0].page_content)

filteredByTextResults = docsearch.similarity_search(
    "rainstorm in parched desert, rain",
    k=1,
    search_strategy=SingleStoreDB.SearchStrategy.FILTER_BY_TEXT,
    filter_threshold=0.1,
)
print(filteredByTextResults[0].page_content)

filteredByVectorResults = docsearch.similarity_search(
    "rainstorm in parched desert, rain",
    k=1,
    search_strategy=SingleStoreDB.SearchStrategy.FILTER_BY_VECTOR,
    filter_threshold=0.1,
)
print(filteredByVectorResults[0].page_content)

weightedSumResults = docsearch.similarity_search(
    "rainstorm in parched desert, rain",
    k=1,
    search_strategy=SingleStoreDB.SearchStrategy.WEIGHTED_SUM,
    text_weight=0.2,
    vector_weight=0.8,
)
print(weightedSumResults[0].page_content)


# ## Multi-modal Example: Leveraging CLIP and OpenClip Embeddings
# 
# In the realm of multi-modal data analysis, the integration of diverse information types like images and text has become increasingly crucial. One powerful tool facilitating such integration is [CLIP](https://openai.com/research/clip), a cutting-edge model capable of embedding both images and text into a shared semantic space. By doing so, CLIP enables the retrieval of relevant content across different modalities through similarity search.
# 
# To illustrate, let's consider an application scenario where we aim to effectively analyze multi-modal data. In this example, we harness the capabilities of [OpenClip multimodal embeddings](/docs/integrations/text_embedding/open_clip), which leverage CLIP's framework. With OpenClip, we can seamlessly embed textual descriptions alongside corresponding images, enabling comprehensive analysis and retrieval tasks. Whether it's identifying visually similar images based on textual queries or finding relevant text passages associated with specific visual content, OpenClip empowers users to explore and extract insights from multi-modal data with remarkable efficiency and accuracy.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -U langchain openai singlestoredb langchain-experimental # (newest versions required for multi-modal)')


# In[ ]:


import os

from langchain_community.vectorstores import SingleStoreDB
from langchain_experimental.open_clip import OpenCLIPEmbeddings

os.environ["SINGLESTOREDB_URL"] = "root:pass@localhost:3306/db"

TEST_IMAGES_DIR = "../../modules/images"

docsearch = SingleStoreDB(OpenCLIPEmbeddings())

image_uris = sorted(
    [
        os.path.join(TEST_IMAGES_DIR, image_name)
        for image_name in os.listdir(TEST_IMAGES_DIR)
        if image_name.endswith(".jpg")
    ]
)

# Add images
docsearch.add_images(uris=image_uris)

