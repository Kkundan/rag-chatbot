import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "resources", "developers_handbook_for_rest_integration_developers_handbook.pdf")

persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the pdf file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    # print(pages[1].page_content)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(pages)}")
    print(f"Sample chunk:\n{pages[1].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")


    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        pages, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")
else:
    print("Persistent directory already exists. Loading vector store...")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "what is the connectionJson in REST Connector?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever()
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# Initialize the OpenAI Embeddings model
# openai_model = OpenAIEmbeddings(model="gpt-4o")

# # Initialize the FAISS vector store
# faiss_store = FAISS(persistent_directory)

# Initialize the text splitter
# text_splitter = CharacterTextSplitter()

# # Initialize the combined chain
# chain = text_splitter | openai_model | vector_store

# # Define the messages
# messages = [
#     ("system", "What is the REST API?"),
#     ("human", "Explain the REST API."),
# ]   # Define the messages   

# # Create the prompt template
# prompt_template = ChatPromptTemplate.from_messages(messages)

# # Create the combined chain using LangChain Expression Language (LCEL)

# chain = prompt_template | chain | StrOutputParser()

# while True:
#     human_input = input("User: ")
    
#     if human_input.lower() == "exit":
#         break   

#     result = chain.invoke(human_input)

#     print(result)
