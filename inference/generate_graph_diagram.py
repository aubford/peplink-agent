#!/usr/bin/env python3
"""
Standalone script to generate the graph_diagram.png file for the RAG inference LangGraph.

This script creates a visual diagram of the LangGraph workflow showing the flow between
different nodes in the RAG inference system.
"""

import os
from dotenv import load_dotenv
from langchain.globals import set_verbose
from langgraph.checkpoint.memory import InMemorySaver
from inference.chat_agentic import ChatLangGraph

# Load environment variables
load_dotenv()

# Set verbose logging (optional)
set_verbose(True)


def generate_graph_diagram():
    """Generate the graph diagram PNG file."""
    print("Creating ChatLangGraph instance...")

    # Create the chatbot instance
    chatbot = ChatLangGraph(
        llm_model="gpt-4.1",  # This can be any model name, not used for graph generation
        pinecone_index_name="pepwave-early-april-page-content-embedding",  # Not used for graph generation
        checkpointer=InMemorySaver(),
    )

    print("Generating graph diagram...")

    # Generate the diagram
    chatbot.draw_graph()

    print("✅ graph_diagram.png has been generated successfully!")
    print(f"📁 Location: {os.path.abspath('graph_diagram.png')}")


if __name__ == "__main__":
    generate_graph_diagram()
