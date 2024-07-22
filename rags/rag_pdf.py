import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_vector_store(embeddings,db):
    # Define the directory containing the text file and the persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(os.path.dirname(current_dir), "resources")
    # file_path = os.path.join(file_dir, "developers_handbook_for_rest_integration_developers_handbook.pdf")

    persistent_directory = os.path.join(current_dir, "db", db)

    # Check if the Chroma vector store already exists
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")
        initialize_chroma_vector_store(llm, embeddings,file_dir, persistent_directory)
    else:
        print("Persistent directory already exists. Loading vector store...")

    return persistent_directory


def initialize_chroma_vector_store(embeddings, file_dir, persistent_directory):
    # Ensure the pdf file exists
    if not os.path.exists(file_dir):
        raise FileNotFoundError(f"The directory {file_dir} does not exist. Please check the path.")

    # List all pdf files in the directory
    pdf_files = [f for f in os.listdir(file_dir) if f.endswith(".pdf")]

    # Read the text content from each file and store it with metadata
    documents = []
    for pdf_file in pdf_files:
        file_path = os.path.join(file_dir, pdf_file)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": pdf_file}
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    logger.info("\n--- Document Chunks Information ---")
    logger.info(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")


def retrieve_relevant_documents(llm, embeddings, query, persistent_directory, chat_history,search_type, search_kwargs):

    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Define the retriever
    retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    # Contextualize question prompt
    # This system prompt helps the AI understand that it should reformulate the question
    # based on the chat history to make it a standalone question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    
    # Create a history-aware retriever
    # This uses the LLM to help reformulate the question based on chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question prompt
    # This system prompt helps the AI understand that it should provide concise answers
    # based on the retrieved context and indicates what to do if the answer is unknown
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a chain to combine documents for question answering
    # `create_stuff_documents_chain` feeds all retrieved context into the LLM
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain.invoke({"input": query, "chat_history": chat_history})



def main():

    # db = "chroma_db_with_metadata"
    # llm = Ollama(model="gemma2:9b")
    # embeddings = OllamaEmbeddings(model="nomic-embed-text") #text-embedding-3-small, nomic-embed-text
    
    db = "chroma_db_with_metadata"
    llm = ChatOpenAI(model="gpt-4o")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 


    # db = "chroma_db_with_mxbai-embed-large"
    # llm = Ollama(model="gemma2:9b")
    # embeddings = OllamaEmbeddings(model="mxbai-embed-large") 


    # result = llm.invoke("Tell me a joke")
    # print(f"AI: {result}")

    persistent_directory = initialize_vector_store(embeddings,db)
    query = "ConnectionJson using basic authentication with HMAC?"

    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        result = retrieve_relevant_documents(llm, embeddings, query, persistent_directory, chat_history, "similarity", {"k": 1})
        # Display the AI's response
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

    
    # retrieve_relevant_documents(query, persistent_directory, "similarity_score_threshold", {"k": 3, "score_threshold": 0.4})
    # retrieve_relevant_documents(query, persistent_directory, "mmr", {"k": 1, "fetch_k": 20, "lambda_mult": 0.4})


if __name__ == "__main__":
    main()
