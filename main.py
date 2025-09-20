import os
import uvicorn
from dotenv import load_dotenv
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables from a .env file at the top
load_dotenv()

# Load the data schema once at startup
try:
    with open('data_schema.yaml', 'r') as f:
        SCHEMA = yaml.safe_load(f)
except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"FATAL: Could not load or parse data_schema.yaml. Error: {e}")
    # In a real app, you might want to exit or handle this more gracefully
    SCHEMA = {}

# Create an instance of the FastAPI application
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model to structure the request body
class UserQuery(BaseModel):
    query: str

def run_agent_1(query: str):
    """
    Acts as a Query Analysis Agent.

    Checks if the query is about the dataset's schema or content. If so,
    it provides a high-level description. Otherwise, it generates a
    mock structured query for further processing.
    """
    # Check for keywords in a case-insensitive manner
    keywords = ['dataset', 'data', 'schema']
    if any(keyword in query.lower() for keyword in keywords):
        metadata = SCHEMA.get('dataset_metadata', {})
        name = metadata.get('name', 'Unnamed Dataset')
        description = metadata.get('description', 'No description available.')
        return f"{name}: {description}"

    # If no keywords are found, return a hardcoded mock structured query
    return {
        'intent': 'AGGREGATION',
        'requested_fields': ['count(patient_id)'],
        'filters': {'diagnosis': 'asthma'},
        'original_query': query
    }


def run_agent_2(structured_query: dict, schema: dict) -> dict:
    """
    Acts as a Compliance Agent.

    Scans the original user query for keywords associated with PII fields.
    Returns an action plan ('BLOCK' or 'PROCEED') and a transparency log.
    """
    original_query = structured_query.get('original_query', '').lower()

    # Iterate through each column defined in the schema
    for col_name, col_props in schema.get('columns', {}).items():
        # Check if the column is marked as PII
        if col_props.get('is_pii', False):
            # Check if any of the column's keywords are in the user query
            for keyword in col_props.get('keywords', []):
                if keyword.lower() in original_query:
                    return {
                        'action_plan': 'BLOCK',
                        'transparency_log': f"Query blocked because it mentions a keyword related to the sensitive PII field: '{col_name}'."
                    }

    # If no PII keywords were found, the query is compliant
    return {
        'action_plan': 'PROCEED',
        'transparency_log': 'Query is compliant with privacy policy. No sensitive keywords found.'
    }

# Define a POST endpoint at /api/query
@app.post("/api/query")
async def process_query(user_query: UserQuery):
    """
    Accepts a user query and processes it through a pipeline of agents.
    1. Agent 1 (Analysis): Determines user intent.
    2. Agent 2 (Compliance): Checks for PII access.
    """
    print(f"Received query: {user_query.query}")

    # Step 1: Run Query Analysis Agent
    agent_1_response = run_agent_1(user_query.query)

    # If Agent 1 returns a string, it's a direct answer (e.g., schema info)
    if isinstance(agent_1_response, str):
        return agent_1_response

    # If Agent 1 returns a dict, it's a structured query to be processed further
    agent_2_response = run_agent_2(agent_1_response, SCHEMA)
    return agent_2_response

# Standard block to run the app with uvicorn for development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)