import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

def initialize_vector_store():
    # Define the directory containing the text file and the persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(current_dir, "resources") 
    file_path = os.path.join(current_dir, "resources", "developers_handbook_for_rest_integration_developers_handbook.pdf")

    persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

    # Check if the Chroma vector store already exists
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")
        initialize_chroma_vector_store(file_dir, file_path, persistent_directory)
    else:
        print("Persistent directory already exists. Loading vector store...")

    return persistent_directory


def initialize_chroma_vector_store(file_dir, file_path, persistent_directory):
    # Ensure the pdf file exists
    if not os.path.exists(file_dir):
        raise FileNotFoundError(
            f"The file {file_dir} does not exist. Please check the path."
        )
    
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
    text_splitter = CharacterTextSplitter()
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")


def retrieve_relevant_documents(query, persistent_directory):
    # Define the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embeddings)

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever()
    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


def main():
    persistent_directory = initialize_vector_store()
    query = "connectionJson using basic authentication with HMAC?"
    retrieve_relevant_documents(query, persistent_directory)


if __name__ == "__main__":
    main()