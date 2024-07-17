from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

result = model.invoke("Hello, how are you?")
print("Result:", result)
print("Result Content: ", result.content)
