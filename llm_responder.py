import gdown
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from typing import List

def generate_llm_reponse(question: str, chat_history: List) -> str:
    """
    Generate a response from a language model based on a given question and chat history.

    This function connects to a ChromaDB instance to retrieve relevant context based on the chat history
    and uses a pre-trained language model to generate a response to the user's question. It reformulates
    the question to be standalone if necessary, retrieves relevant documents, and generates a concise answer.

    Args:
        question (str): The user's question to be answered.
        chat_history (List): The chat history which may provide context for the question.

    Returns:
        str: The generated response from the language model.
    """
    # Set chromadb vectorstore connection variables
    CHROMADB_HOST = 'host.docker.internal'  # chromadb is running as a docker container
    CHROMADB_PORT = 8000
    CHROMADB_SETTINGS = Settings(allow_reset=True)

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # connect to the chromadb
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT, settings=CHROMADB_SETTINGS)
    db = Chroma(
        client=client,
        collection_name="langchain",
        embedding_function=embedding_function,
    )

    retriever = db.as_retriever()

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, temperature=0.5
    )

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain.invoke({"input": question, "chat_history": chat_history})