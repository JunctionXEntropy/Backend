import os
import uvicorn
from dotenv import load_dotenv
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import BaseModel as LangchainBaseModel, Field
from data_manager import load_patient_data
import pandas as pd
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

# Load the patient data once at startup
PATIENT_DATA = load_patient_data()
if PATIENT_DATA is None:
    print("FATAL: Could not load patient data. The application may not function correctly.")
    # In a real app, you might want to exit.

# Initialize LLM and memory for conversation
llm = ChatOpenAI(model="gpt-4o", temperature=0)
memory = ConversationBufferWindowMemory(k=3)

# Define the desired structured output from the LLM using LangChain's Pydantic
class StructuredQuery(LangchainBaseModel):
    intent: str = Field(description='The user\'s classified intent. Must be one of: "AGGREGATION", "TABULAR_DATA", or "SPECIFIC_RECORD".')
    requested_fields: list[str] = Field(description="A list of strings representing the exact column names the user is asking for.")
    filters: dict = Field(description="The filters the user wants to apply.")
    original_query: str = Field(description="The user's rewritten, standalone query.")


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

def sanitize_filters(filters: dict, valid_columns: list[str]) -> dict:
    """Keep only keys that are valid schema columns."""
    return {k: v for k, v in filters.items() if k.lower() in valid_columns}


def run_agent_1(query: str) -> dict | str:
    """
    Acts as a Query Analysis Agent. Final Version.
    Handles metadata, then uses a single, powerful LLM call to perform
    semantic mapping and structure the user's query into a JSON object.
    """
    #print(f"\n\n\n================ NEW REQUEST ================")
    #print(f"--- AGENT 1: INPUT ---")
    #print(f"Query: {query}")

    # Handles metadata queries directly
    keywords = ['dataset', 'data', 'schema']
    if any(keyword in query.lower() for keyword in keywords):
        metadata = SCHEMA.get('dataset_metadata', {})
        name = metadata.get('name', 'Unnamed Dataset')
        description = metadata.get('description', 'No description available.')
        
        response = f"{name}: {description}"
        #print(f"--- AGENT 1: OUTPUT (Metadata) ---")
        #print(response)
        return response

    try:
        # Create a detailed summary of the schema to provide as context to the LLM
        schema_summary = "\n".join(
            [f"- {col}: {props.get('description', 'No description')}" for col, props in SCHEMA.get('columns', {}).items()]
        )
        
        parser = JsonOutputParser(pydantic_object=StructuredQuery)

        # A single, powerful prompt that instructs the LLM to reason and then format
        prompt_template_str = """You are an expert query analyzer. Your task is to convert a user's question into a structured JSON object based on the provided schema.

Follow these steps:
1.  Analyze the user's query to understand their intent.
2.  Identify all entities and values in the query (e.g., 'asthma', 'oulu', 'age').
3.  Map these entities to the most relevant column names from the schema summary.
4.  IMPORTANT RULE: If the user asks for a general list of 'patients' or 'records' without specifying fields, you MUST only request a small, pre-approved set of safe columns: ['Age', 'Gender', 'City', 'Condition'].
5.  Construct a final JSON object based on your analysis.

Schema Summary:
{schema_summary}

The final JSON output MUST follow these formatting instructions:
{format_instructions}

Conversation History:
{history}

User Query:
{input}

Final JSON Output:
"""

        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["history", "input"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "schema_summary": schema_summary
            },
        )

        chain = prompt | llm | parser
        response = chain.invoke({"input": query, "history": memory.buffer_as_str})

        valid_columns = [col.lower() for col in SCHEMA.get('columns', {}).keys()]
        if "filters" in response and isinstance(response["filters"], dict):
            response["filters"] = {k: v for k, v in response["filters"].items() if k.lower() in valid_columns}

        memory.save_context({"input": query}, {"output": json.dumps(response)})
        
        #print(f"--- AGENT 1: OUTPUT ---")
        #print(response)
        return response
    
    except Exception as e:
        print(f"Error in Agent 1: {e}")
        response = {"error": "Failed to process query", "details": "The language model could not understand the request."}
        #print(f"--- AGENT 1: OUTPUT (Error) ---")
        #print(response)
        return response
def run_agent_2(structured_query: dict, schema: dict) -> dict:
    """
    Acts as a Compliance Agent. Final Version.
    Checks for PII in both the original query and the requested_fields list.
    """
    #print(f"\n--- AGENT 2: INPUT ---")
    #print(structured_query)

    original_query = structured_query.get('original_query', '').lower()
    schema_columns = schema.get('columns', {})

    # CHECK 1: Scan the original query text for explicit PII keywords
    for col_name, col_props in schema_columns.items():
        if col_props.get('is_pii', False):
            for keyword in col_props.get('keywords', []):
                if keyword.lower() in original_query:
                    decision = {
                        'action_plan': 'BLOCK',
                        'transparency_log': f"Query blocked because it mentions a keyword related to the sensitive PII field: '{col_name}'."
                    }
                    #print(f"--- AGENT 2: OUTPUT ---")
                    #print(decision)
                    return decision

    # CHECK 2: Audit the fields requested by Agent 1's LLM
    requested_fields = [field.lower() for field in structured_query.get('requested_fields', [])]
    schema_columns_lower = {col.lower(): props for col, props in schema.get('columns', {}).items()}
    
    for field in requested_fields:
        if schema_columns_lower.get(field, {}).get('is_pii', False):
            decision = {
                'action_plan': 'BLOCK',
                'transparency_log': f"Query blocked because it explicitly requests the sensitive PII field: '{field}'."
            }
            #print(f"--- AGENT 2: OUTPUT ---")
            #print(decision)
            return decision
    
    # If both checks pass, the query is compliant
    decision = {
        'action_plan': 'PROCEED',
        'transparency_log': 'Query is compliant with privacy policy. No sensitive keywords found.'
    }
    #print(f"--- AGENT 2: OUTPUT ---")
    #print(decision)
    return decision

def run_agent_3(action_plan: dict, structured_query: dict, dataframe: pd.DataFrame) -> dict:
    """
    Acts as a Data Retrieval Agent. Final Version.
    Executes a query against the dataframe if the compliance check passed.
    """
    if action_plan.get('action_plan') == 'BLOCK':
        return {'status': 'blocked', 'data_payload': None}
    
    if action_plan.get('action_plan') in ['PROCEED', 'AGGREGATE']:
        try:
            filtered_df = dataframe.copy()
            filtered_df.columns = [col.lower() for col in filtered_df.columns]
            filters = structured_query.get('filters', {})

            for column_key in filters.keys():
                if column_key.lower() not in filtered_df.columns:
                    return {'status': 'error', 'data_payload': f"Invalid filter column provided: '{column_key}'"}
            
            if filters:
                for column, value in filters.items():
                    filtered_df = filtered_df[filtered_df[str(column).lower()].str.lower() == str(value).lower()]

            intent = structured_query.get('intent')
            if intent == 'AGGREGATION':
                original_query_lower = structured_query.get('original_query', '').lower()

                # --- START: CORRECTED AGGREGATION LOGIC ---
                if 'average' in original_query_lower or 'mean' in original_query_lower:
                    target_column = None
                    # Find the target column by checking for the column name OR its keywords in the query
                    for col_name, props in SCHEMA.get('columns', {}).items():
                        if props.get('data_type') == 'integer':
                            search_terms = [col_name.lower()] + [kw.lower() for kw in props.get('keywords', [])]
                            for term in search_terms:
                                if term.replace('_', ' ') in original_query_lower:
                                    target_column = col_name.lower()
                                    break
                        if target_column:
                            break
                    
                    if target_column:
                        if filtered_df.empty:
                            return {'status': 'success', 'data_payload': {'patient_count': 0}}
                        else:
                            average_value = filtered_df[target_column].mean()
                            payload_key = f"average_{target_column}"
                            return {'status': 'success', 'data_payload': {payload_key: round(average_value, 2)}}
                    else:
                        return {'status': 'error', 'data_payload': "Could not determine a valid numeric column to average from the query."}
                # --- END: CORRECTED AGGREGATION LOGIC ---
                
                else: # Default to count if 'average' or 'mean' is not in the query
                    count = len(filtered_df)
                    return {'status': 'success', 'data_payload': {'patient_count': count}}

            elif intent == 'TABULAR_DATA':
                requested_fields = [f.lower() for f in structured_query.get('requested_fields', [])]
                for field in requested_fields:
                    if SCHEMA['columns'].get(field, {}).get('is_pii', False):
                         return {'status': 'error', 'data_payload': f"Access to sensitive PII field is not allowed for tabular display: '{field}'"}
                
                table_df = filtered_df[requested_fields]
                table_data = table_df.to_dict(orient='split')
                if 'index' in table_data:
                    del table_data['index']
                return {'status': 'success', 'data_payload': table_data}
        
        except Exception as e:
            return {'status': 'error', 'data_payload': f'An error occurred during data processing: {e}'}
    
    return {'status': 'error', 'data_payload': f"Unknown action plan: {action_plan.get('action_plan')}"}



def run_agent_4(data_payload: dict, transparency_log: str, original_query: str) -> dict:
    """
    Acts as a final Response Generation Agent. Final DEMO-READY version.
    Generates a response using simple logic for reliability.
    """
    is_tabular = isinstance(data_payload, dict) and 'columns' in data_payload
    answer = "Your query was processed successfully." # Default message

    if is_tabular:
        num_rows = len(data_payload.get('data', []))
        answer = f"I found {num_rows} records matching your query."
    
    elif isinstance(data_payload, dict):
        if data_payload.get('patient_count') is not None:
            count = data_payload['patient_count']
            answer = f"The result of your query is: {count} patients found." if count > 0 else "No patients matched the criteria for your query."
        
        elif any(key.startswith('average_') for key in data_payload.keys()):
            avg_key = next(key for key in data_payload if key.startswith('average_'))
            avg_val = data_payload[avg_key]
            subject = avg_key.replace('average_', '').replace('_', ' ')
            answer = f"The average {subject} for the matching patients is {avg_val}."

    return {
        "response_type": "composite" if is_tabular else "text",
        "text_response": answer,
        "table_data": data_payload if is_tabular else None,
        "transparency_notice": transparency_log
    } 
    

@app.post("/api/query")
async def process_query(user_query: UserQuery):
    """
    Accepts a user query and processes it through a pipeline of agents.
    1. Agent 1 (Analysis): Determines user intent.
    2. Agent 2 (Compliance): Checks for PII access.
    3. Agent 3 (Retrieval): Fetches data if compliant.
    4. Agent 4 (Response): Generates a natural language answer.
    """
    print(f"Received query: {user_query.query}")

    # Step 1: Run Query Analysis Agent
    agent_1_response = run_agent_1(user_query.query)
    
    # Handle direct string response from Agent 1 (e.g., metadata query)
    if isinstance(agent_1_response, str):
        return {
            "response_type": "text",
            "text_response": agent_1_response,
            "table_data": None,
            "transparency_notice": "Provided dataset metadata."
        }

    # Handle parsing error from Agent 1
    if agent_1_response.get("error"):
        return {
            "response_type": "error",
            "text_response": "I had trouble understanding your request. Could you please rephrase it?",
            "table_data": None,
            "transparency_notice": agent_1_response.get("details", "Error during query analysis.")
        }

    # Step 2: Run Compliance Agent
    agent_2_response = run_agent_2(agent_1_response, SCHEMA)

    # Handle blocked query from Agent 2
    if agent_2_response.get('action_plan') == 'BLOCK':
        return {
            "response_type": "blocked",
            "text_response": "I cannot answer this question as it may involve sensitive information.",
            "table_data": None,
            "transparency_notice": agent_2_response.get('transparency_log')
        }

    # --- THIS IS THE CORRECTED LINE ---
    if agent_2_response.get('action_plan') in ['PROCEED', 'AGGREGATE']:
        if PATIENT_DATA is None:
            return {"response_type": "error", "text_response": "I cannot answer this question right now due to a server configuration issue.", "table_data": None, "transparency_notice": "Server is misconfigured: Patient data is not available."}
        
        agent_3_response = run_agent_3(agent_2_response, agent_1_response, PATIENT_DATA)

        if agent_3_response.get('status') == 'success':
            # Step 4: Generate final natural language response
            return run_agent_4(data_payload=agent_3_response.get('data_payload'), transparency_log=agent_2_response.get('transparency_log'), original_query=agent_1_response.get('original_query'))
        else: # Handle error from Agent 3
            return {"response_type": "error", "text_response": "I encountered an error while retrieving the data. Please check if your query is valid.", "table_data": None, "transparency_notice": agent_3_response.get('data_payload', 'An error occurred during data retrieval.')}

    # Fallback for any unhandled cases
    return {"response_type": "error", "text_response": "An unexpected error occurred.", "table_data": None, "transparency_notice": "An unknown error occurred in the processing pipeline."}

# Standard block to run the app with uvicorn for development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)