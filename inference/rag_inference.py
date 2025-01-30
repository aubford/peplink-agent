from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import global_config
from langchain import hub
from pinecone import Pinecone


pinecone = Pinecone(api_key=global_config.get("PINECONE_API_KEY"))
index = pinecone.Index("pepwave")
embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text", namespace="pepwave")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("aubford/retrieval-qa-chat")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(
    vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 40}), document_chain
)

chat_history = []
while True:
    # Get user input
    query = input("Enter a query (or 'exit' to exit): ")

    # Check for exit condition
    if query.lower() == "exit":
        break

    result = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
    print(f"Assistant: {result['answer']}\n")
    chat_history.append(("human", query))
    chat_history.append(("assistant", result["answer"]))
