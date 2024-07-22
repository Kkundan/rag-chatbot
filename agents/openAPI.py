from dotenv import load_dotenv
import os
import yaml
import requests
import spotipy.util as util

from langchain_community.agent_toolkits import OpenAPIToolkit
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_community.utilities import RequestsWrapper
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
import tiktoken


# Load environment variables from .env file
load_dotenv()


# Initialize the local LLM and OpenAPI toolkit
# llm = Ollama(model="gemma2:9b", temperature=0)
llm = ChatOpenAI(model="gpt-4o")
current_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(os.path.dirname(current_dir), "resources")

# with open(f"{file_dir}/saviynt-rest-api-5.0-bundle.yaml", encoding='utf-8') as f:
#     raw_saviynt_api_spec = yaml.load(f, Loader=yaml.Loader)
# saviynt_api_spec = reduce_openapi_spec(raw_saviynt_api_spec)

with open(f"{file_dir}/spotify_openapi.yaml", encoding='utf-8') as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)

# Define a simple chat function to interact with the user
def chat_with_user():
    print("Hello! I can help you interact with our API. What would you like to do?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = process_user_input(user_input)
        print(f"Bot: {response}")

# Load environment variables from .env file
load_dotenv()

# Get username and password from .env file
# access_token = os.getenv("SAVIYNT_API_ACCESS_TOKEN")

# Authenticate and get access token
# def authenticate():
    # Simulate the process of obtaining an access token
    # In a real scenario, this would involve making a request to the authentication endpoint
    # of the API with credentials and receiving an access token in response.
    # This is just a placeholder, replace it with actual authentication logic.
    # return {"Authorization": f"Bearer {access_token}"}


def construct_spotify_auth_headers(raw_spec: dict):
    # Safely access nested dictionaries to avoid KeyError
    oauth_2_0 = raw_spec.get("components", {}).get("securitySchemes", {}).get("oauth_2_0", None)
    if oauth_2_0:
        flows = oauth_2_0.get("flows", {}).get("authorizationCode", None)
        if flows:
            scopes = list(flows.get("scopes", {}).keys())
            access_token = util.prompt_for_user_token(scope=",".join(scopes))
            return {"Authorization": f"Bearer {access_token}"}
    else:
        # Handle the case where 'oauth_2_0' or other expected keys are not found
        print("Error: 'oauth_2_0' not found in the API specification.")
        return {}




# Process user input
def process_user_input(user_input):
    # Use the LLM to parse the user's intent and parameters from the input
    try:
        # Authenticate and get access token
        # authHeader = authenticate()

        # Get API credentials.
        headers = construct_spotify_auth_headers(raw_spotify_api_spec)
        requests_wrapper = RequestsWrapper(headers=headers)

        spotify_agent = planner.create_openapi_agent(
            spotify_api_spec,
            requests_wrapper,
            llm,
            allow_dangerous_requests=True,
        )

        # Make the API call if the user confirms
        # response = openapi_toolkit.call_api(intent, parameters, access_token=access_token, base_url="https://qe-eictrunk-aws.saviyntcloud.com/ECM")
        # return f"API Response: {response}"

        response = spotify_agent.invoke({"input": user_input})
        return response
    except Exception as e:
        return f"Error processing your request: {e}"

if __name__ == "__main__":
    
    # endpoints = [
    #             (route, operation)
    #             for route, operations in raw_saviynt_api_spec["paths"].items()
    #             for operation in operations
    #             if operation in ["get", "post"]
    #         ]
    # print(len(endpoints))

    
    chat_with_user()
