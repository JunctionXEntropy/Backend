PrivacyGuard Agent
A Multi-Agent System for Privacy-Preserving Data Analysis, built for the JunctionX Oulu 2025 Challenge.
This project demonstrates a sophisticated, four-agent pipeline that allows users to ask natural language questions about a sensitive dataset. It enforces privacy rules dynamically, ensuring that no Personally Identifiable Information (PII) is ever exposed, while still delivering meaningful insights.

Key Features
Natural Language Understanding: Leverages a Large Language Model (LLM) to understand complex queries and conversational context.
Conversational Memory: Remembers the context of previous questions to answer follow-ups intelligently.
Schema-Driven Security: All privacy rules are managed in a human-readable data_schema.yaml file, making the system's security policies transparent and easy to configure.
Deterministic Compliance: A dedicated, rule-based agent acts as a "guardrail," auditing every query to block access to PII. This critical security check uses no LLM, making it 100% reliable.
Layered Security: A second layer of validation in the data processing agent ensures PII is never accidentally returned in tabular data.
Dynamic Data Operations: Can perform aggregations (like counts and averages) and prepare sanitized tables of data.
Full Transparency: Every response is accompanied by a transparency_notice that explains what privacy measures were applied and why.

Architecture Overview
The system is built on a "Sanitized Retrieval-Augmented Generation (RAG)" pipeline, executed by a series of specialized agents. This ensures that the core LLM is sandboxed and never interacts with raw, sensitive data.
[User Query] -> [Agent 1: Semantic Analyzer] -> [Agent 2: Compliance Auditor] -> [Agent 3: Computational Engine] -> [Agent 4: Response Formatter] -> [Final JSON Response]
Agent 1: Semantic Query Analyzer (LLM-Powered): Understands the user's intent, handles conversation history, and translates the query into a structured, machine-readable format using the data schema as context.
Agent 2: Compliance Auditor (Deterministic): The security core. It audits the structured query from Agent 1 against the rules in data_schema.yaml. It checks both the user's original text for risky keywords and the specific fields requested by Agent 1 to block any PII access.
Agent 3: Computational Engine (Deterministic): The only component that touches the data. It executes the approved "Action Plan" from Agent 2, performing safe operations like filtering, aggregation (count, average), and preparing sanitized data tables using Pandas.
Agent 4: Response Formatter (Deterministic): Assembles the final, user-friendly JSON response. It uses simple, reliable logic to create a helpful text summary of the results and packages any tabular data for the frontend.

Getting Started
Follow these instructions to set up and run the backend server locally.

1. Prerequisites
Python 3.9+
Git

2. Installation
Clone the repository:
git clone https://github.com/JunctionXEntropy/your-project-name.git
cd your-project-name/Backend
Create and configure your environment file:
Rename the example environment file:
# On Windows
copy .env.example .env
# On macOS/Linux
cp .env.example .env
Open the new .env file and add your OpenAI API key:

OPENAI_API_KEY="sk-YourSecretKeyGoesHere"
Set up a Python virtual environment:
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Install the required dependencies:
pip install -r requirements.txt

3. Running the Server
With your virtual environment activated, run the following command in the Backend directory:
uvicorn main:app --reload
The server will start and be accessible at http://127.0.0.1:8000.

Configuration
The agent's behavior is controlled by the data_schema.yaml file:
dataset_metadata: Contains a high-level description of the dataset.
columns: Defines each column's data_type, a description for the LLM, an is_pii flag (true/false) for Agent 2, and a list of keywords for both PII checking and aggregation logic.

API Usage
The API has one primary endpoint for all queries.
Endpoint: POST /api/query
Interactive Docs: For easy testing, visit http://127.0.0.1:8000/docs while the server is running.

License
This project is licensed under the MIT License.
