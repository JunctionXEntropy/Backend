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


def run_agent_1(query: str):
    """
    Acts as a Query Analysis Agent.

    Checks for metadata keywords first. Otherwise, uses an LLM chain
    to convert the user's query into a structured JSON object.
    """

    print(f"\n\n\n================ NEW REQUEST ================")
    print(f"--- AGENT 1: INPUT ---")
    print(f"Query: {query}")

    # Check for keywords in a case-insensitive manner
    keywords = ['dataset', 'data', 'schema']
    if any(keyword in query.lower() for keyword in keywords):
        metadata = SCHEMA.get('dataset_metadata', {})
        name = metadata.get('name', 'Unnamed Dataset')
        description = metadata.get('description', 'No description available.')
        return f"{name}: {description}"

    # For all other queries, use the LLM chain to get a structured response
    try:
        # Get valid column names from the schema to guide the LLM
        column_names = [col.lower() for col in SCHEMA.get('columns', {}).keys()]

        parser = JsonOutputParser(pydantic_object=StructuredQuery)

        # PROMPT TEMPLATE
        prompt_template_str = """You are an expert query analyzer for a hospital dataset. Your task is to convert a user's question into a structured JSON object.

The keys in the 'filters' dictionary MUST BE from the following lowercase list of valid column names: {column_names}
Do not invent any other keys.

Example 1:
User Query: "list the blood types for female patients in oulu"
JSON Output:
{{
  "intent": "TABULAR_DATA",
  "requested_fields": ["blood_type"],
  "filters": {{"gender": "Female", "city": "Oulu"}},
  "original_query": "list the blood types for female patients in oulu"
}}

Example 2:
User Query: "how many have asthma"
JSON Output:
{{
  "intent": "AGGREGATION",
  "requested_fields": ["patient_count"],
  "filters": {{"condition": "Asthma"}},
  "original_query": "how many patients have asthma"
}}

Now, analyze the following. Output ONLY the JSON object.

Conversation History:
{history}

User Query: {input}
JSON Output:
"""

        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["history", "input"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "column_names": column_names
            },
        )

        chain = prompt | llm | parser
        response = chain.invoke({"input": query, "history": memory.buffer_as_str})

        # 🔹 Sanitize filters against schema right here
        valid_columns = [col.lower() for col in SCHEMA.get('columns', {}).keys()]
        response["filters"] = sanitize_filters(response.get("filters", {}), valid_columns)

        memory.save_context({"input": query}, {"output": json.dumps(response)})

        print(f"--- AGENT 1: OUTPUT ---")
        print(response)

        return response
    except Exception as e:
        print(f"Error in Agent 1: Failed to parse LLM response. Error: {e}")
        return {"error": "Failed to parse query", "details": "The language model returned an invalid format."}

def run_agent_2(structured_query: dict, schema: dict) -> dict:
    """
    Acts as a Compliance Agent.

    Scans the original user query for keywords associated with PII fields.
    Returns an action plan ('BLOCK' or 'PROCEED') and a transparency log.
    """

    print(f"\n--- AGENT 2: INPUT ---")
    print(structured_query)

    original_query = structured_query.get('original_query', '').lower()

    # Iterate through each column defined in the schema
    for col_name, col_props in schema.get('columns', {}).items():
        # Check if the column is marked as PII
        if col_props.get('is_pii', False):
            # Check if any of the column's keywords are in the user query
            for keyword in col_props.get('keywords', []):
                if keyword.lower() in original_query:
                    decision = {
                        'action_plan': 'BLOCK',
                        'transparency_log': f"Query blocked because it mentions a keyword related to the sensitive PII field: '{col_name}'."
                    }
                    print(f"--- AGENT 2: OUTPUT ---\n{decision}\n")
                    return decision

    # If no PII keywords were found, the query is compliant
    decision = {
        'action_plan': 'PROCEED',
        'transparency_log': 'Query is compliant with privacy policy. No sensitive keywords found.'
    }
    print(f"--- AGENT 2: OUTPUT ---\n{decision}\n")
    return decision

def run_agent_3(action_plan: dict, structured_query: dict, dataframe: pd.DataFrame) -> dict:
    """
    Acts as a Data Retrieval Agent.

    Executes a query against the dataframe if the compliance check passed.
    """

    print(f"\n--- AGENT 3: INPUT ---")
    print(f"Action Plan: {action_plan}")

    # Step 1: Check the action plan from the compliance agent
    if action_plan.get('action_plan') == 'BLOCK':
        result = {'status': 'blocked', 'data_payload': None}
        print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
        return result
    
    # Step 2: Proceed only if the action plan is 'PROCEED'
    if action_plan.get('action_plan') in ['PROCEED', 'AGGREGATE']:
        try:
            filtered_df = dataframe.copy()
            # Convert dataframe columns to lowercase for case-insensitive matching
            filtered_df.columns = [col.lower() for col in filtered_df.columns]

            filters = structured_query.get('filters', {})

            # Validate that all filter keys are valid columns in the dataframe
            for column_key in filters.keys():
                # Convert key to lowercase for check
                if column_key.lower() not in filtered_df.columns:
                    result = {'status': 'error', 'data_payload': f"Invalid filter column provided: '{column_key}'"}
                    print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
                    return result   
                
            # Apply filters to the dataframe
            if filters:
                for column, value in filters.items():
                    # Apply each filter sequentially using standard boolean indexing
                    filtered_df = filtered_df[filtered_df[str(column).lower()].str.lower() == str(value).lower()]

            # Step 3: Perform action based on intent
            intent = structured_query.get('intent')
            if intent == 'AGGREGATION':
                requested_field_str = structured_query.get('requested_fields', ['count'])[0].lower()

                # Handle COUNT aggregation
                if 'count' in requested_field_str:
                    count = len(filtered_df)
                    result = {'status': 'success', 'data_payload': {'patient_count': count}}
                    print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
                    return result

                # Handle AVERAGE/MEAN aggregation
                elif 'average' in requested_field_str or 'mean' in requested_field_str:
                    # Find which numeric column is being requested from the schema
                    numeric_columns = [col.lower() for col, props in SCHEMA.get('columns', {}).items() if props.get('data_type') == 'integer']
                    
                    target_column = None
                    for col in numeric_columns:
                        if col.replace('_', '') in requested_field_str.replace('_', ''):
                            target_column = col
                            break
                    
                    if target_column:
                        if filtered_df.empty:
                            average_value = 0
                        else:
                            # Ensure the column is numeric before calculating mean
                            if pd.api.types.is_numeric_dtype(filtered_df[target_column]):
                                average_value = filtered_df[target_column].mean()
                            else:
                                result = {'status': 'error', 'data_payload': f"Column '{target_column}' is not numeric and cannot be averaged."}
                                print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
                                return result
                        
                        payload_key = f"average_{target_column}"
                        result = {'status': 'success', 'data_payload': {payload_key: round(average_value, 2)}}
                        print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
                        return result
                    else:
                        result = {'status': 'error', 'data_payload': f"Could not determine a valid numeric column to average from request: '{requested_field_str}'"}
                        print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
                        return result
                else:
                    result = {'status': 'error', 'data_payload': f"Unsupported aggregation type in request: '{requested_field_str}'"}
                    print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
                    return result

            elif intent == 'TABULAR_DATA':
                requested_fields = structured_query.get('requested_fields', [])
                # Ensure requested fields are lowercase to match dataframe columns
                requested_fields_lower = [field.lower() for field in requested_fields]

                # Validate that all requested fields are valid columns
                for field in requested_fields_lower:
                    if field not in filtered_df.columns:
                        result = {'status': 'error', 'data_payload': f"Invalid field requested for display: '{field}'"}
                        print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
                        return result

                table_df = filtered_df[requested_fields_lower]
                table_data = table_df.to_dict(orient='split')

                result = {'status': 'success', 'data_payload': table_data}
                print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
                return result
        
        except Exception as e:
            print(f"Error during data retrieval in Agent 3: {e}")
            result = {'status': 'error', 'data_payload': 'An error occurred while processing the data.'}
            print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
            return result   
        
    result = {'status': 'error', 'data_payload': f"Unknown action plan: {action_plan.get('action_plan')}"}
    print(f"--- AGENT 3: OUTPUT ---\n{result}\n")
    return result   

def run_agent_4(data_payload: dict, transparency_log: str, original_query: str) -> dict:
    """
    Acts as a final Response Generation Agent.
    
    Generates a natural language response and formats the final output,
    including tabular data if present.
    """
    try:
        print(f"\n--- AGENT 4: INPUT ---")
        print(f"Original Query: {original_query}")
        print(f"Data Payload: {data_payload}")

        # Format the data_payload into a simple string context for the LLM
        data_context = json.dumps(data_payload)

        parser = StrOutputParser()

        prompt_template_str = """You are a helpful hospital administration assistant. 
Answer the user's question using ONLY the provided data_payload. 
You may use simple logical reasoning on the data (for example: if the count is 0, 
then averages, minimums, or maximums cannot be computed). 
Do not make up values that are not present or implied by the data.
        User's Question: {question}. Context: {context}. Your concise, natural-language answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["question", "context"],
        )

        chain = prompt | llm | parser

        answer = chain.invoke(
            {"question": original_query, "context": data_context}
        )

        # Check if the payload is tabular data to include in the final response
        is_tabular = isinstance(data_payload, dict) and 'columns' in data_payload

        final_response = {
            "response_type": "text",
            "text_response": answer,
            "table_data": data_payload if is_tabular else None,
            "transparency_notice": transparency_log
        }
        print(f"--- AGENT 4: OUTPUT ---\n{final_response}\n")
        return final_response

    except Exception as e:
        print(f"Error in Agent 4: {e}")
        final_response = {
            "response_type": "error",
            "text_response": "An error occurred while generating the final response.",
            "table_data": None,
            "transparency_notice": "An internal error occurred."
        }
        print(f"--- AGENT 4: OUTPUT ---\n{final_response}\n")
        return final_response

# Define a POST endpoint at /api/query
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

    # Step 3 & 4: If compliance check passed, run Data Retrieval and Response Generation
    if agent_2_response.get('action_plan') == 'PROCEED':
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