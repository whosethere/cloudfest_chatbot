## Quick Introduction to the Topic

**Models of Large Language (MLLs)** have demonstrated their proficiency in grasping contextual nuances and delivering precise responses across a spectrum of Natural Language Processing (NLP) applications, such as summarization and question-answering, upon receiving prompts. Although they excel in generating highly accurate responses based on the data they were trained on, these models often produce erroneous or fabricated information when dealing with subjects outside their training corpus.

The technique known as **Retrieval Augmented Generation (RAG)** integrates LLMs with external informational sources. Hence, the core elements of RAG consist of a retrieval system and a generation mechanism.

The **retrieval component** functions as a mechanism capable of encoding data in such a manner that the pertinent information can be efficiently extracted when queried. This process utilizes text embeddings, meaning a model is trained to generate a vectorial representation of textual data. For the construction of a retrieval system, employing a vector database emerges as the optimal strategy.

## Running Instructions

In the `chatbot.ipynb` file, you will find the code along with descriptions and instructions necessary for executing the script. After familiarizing yourself with the code, you can make the appropriate modifications in the `chatbot.py` file and execute it.

## How to Run:

1. On a machine equipped with a GPU, ensure that NVIDIA drivers and CUDA/cudnn are installed, then install Python.
2. Within a virtual environment, install the packages from the `requirements.txt` file.
3. Execute the `chatbot.py` file.

If you have any questions or concerns, please don't hesitate to contact me.