from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import global_config
from langchain import hub
from pinecone import Pinecone
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from inference.history_aware_retrieval_query import history_aware_retrieval_query


# Note: for reasoning models: "include only the most relevant information to prevent the model from overcomplicating its response." - api docs
# Other advice for reasoning models: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

pinecone = Pinecone(api_key=global_config.get("PINECONE_API_KEY"))
index = pinecone.Index("pepwave")
embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text", namespace="pepwave")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("aubford/retrieval-qa-chat")


contextualize_q_llm = ChatOpenAI(model="o1-mini")
contextualize_q_system_prompt = (
    "Given the following chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 40})
history_aware_retriever = history_aware_retrieval_query() | retriever

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

chat_history = []
while True:
    # Get user input
    query = input("\n\n*** Enter a query (or 'exit' to exit): ")

    # Check for exit condition
    if query.lower() == "exit":
        break

    result = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
    print(f"\n\nAssistant: {result['answer']}\n")
    chat_history.append(("human", query))
    chat_history.append(("assistant", result["answer"]))
