from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

chat_history = []  # Use a list to store messages

messages = [
    ("system", "Tell me a {adjective} joke about a {topic}"),
    ("human", "Tell me {joke_count} joke."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()

while True:
    human_input = input("User: ")
    
    if human_input.lower() == "exit":
        break

    result = chain.invoke({"adjective": "funny", "topic": "panda", "joke_count": human_input})

    print(result)
