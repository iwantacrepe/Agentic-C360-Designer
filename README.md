# 🏦 Agentic Customer 360 Data Product Designer

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/iwantacrepe/accenturehack)

An AI-powered multi-agent system designed to automate the initial phases of Customer 360 data product design specifically for the retail banking sector. This project leverages Google Gemini and a structured agentic workflow to accelerate the process from business use case understanding to mapping generation.

Developed for the Accenture Hackathon.

---

## ✨ Live Demo

Experience the application live on Hugging Face Spaces:

**[➡️ Try the Demo Here](https://huggingface.co/spaces/iwantacrepe/accenturehack)**

---

## 🎯 Problem Solved

Designing tailored Customer 360 data products for diverse retail banking needs (e.g., personalized marketing, risk assessment, churn prediction) is traditionally a manual, time-consuming, and expertise-reliant process. Key pain points include:

*   **Slow Turnaround:** Manual design takes weeks or months.
*   **Resource Intensive:** Requires significant effort from data architects, stewards, and domain experts.
*   **Scalability Issues:** Difficult to create bespoke products for numerous evolving use cases efficiently.
*   **Inconsistency Risk:** Manual designs can vary, leading to data silos or integration challenges.

This project aims to automate the initial, often repetitive, stages of this process using an AI-driven multi-agent system.

---

## ⭐ Features

*   **🤖 Multi-Agent Workflow:** Utilizes distinct AI agents, each specializing in a specific task.
*   **📄 Use Case Analysis (Agent 1):** Ingests natural language business requirements and extracts key entities, attributes, goals, and constraints.
*   **🏗️ Schema Design & Enrichment (Agent 2):** Recommends an optimal target data product schema based on requirements and automatically enriches it with standard metadata fields and primary key suggestions.
*   **🔍 Source Identification (Agent 3):** Queries a (mock) data catalog using fuzzy matching (fuzzywuzzy) to identify potential source systems and columns for the required attributes, providing confidence scores.
*   **🗺️ Mapping Generation (Agent 4):** Creates source-to-target mapping specifications, suggesting mapping types (Direct, Transformation, Metadata, Manual Review) and basic transformation logic or quality checks.
*   **✨ Interactive UI:** Built with Gradio for easy input of use cases and clear visualization of outputs (JSON and DataFrames).
*   **🚀 Powered by Google Gemini:** Leverages the capabilities of the `gemini-1.5-flash` model for analysis and generation tasks.

---

## ⚙️ How It Works (Architecture)

The system follows a sequential multi-agent workflow orchestrated within the Gradio application:

1.  **Input:** User provides a business use case description in natural language.
2.  **Agent 1 (Requirements Analyst):**
    *   Receives the use case text.
    *   Calls the Gemini LLM with a specific prompt to analyze the text and extract structured requirements (goal, entities, attributes, metrics, filters, etc.) into a JSON format.
3.  **Agent 2 (Data Architect & Enricher):**
    *   Receives the structured requirements JSON.
    *   Calls the Gemini LLM with a prompt to design an initial flat schema based on the required attributes.
    *   Applies rule-based and LLM-assisted enrichment: adds standard metadata columns (`c360_master_customer_id`, `data_product_load_ts`, `source_system_tracker`) and attempts to identify/suggest a primary key based on naming conventions.
    *   Outputs the enriched schema as JSON.
4.  **Agent 3 (Source Identification - Mocked):**
    *   Receives the target attribute names from the designed schema.
    *   Performs fuzzy string matching (`fuzzywuzzy.process.extractBests`) against a predefined Python dictionary (`MOCK_CATALOG`) representing source systems and columns.
    *   Outputs a dictionary mapping target attributes to a list of potential sources, including confidence scores and descriptions.
5.  **Agent 4 (Mapping Specialist):**
    *   Receives the target schema and the potential source candidates list.
    *   Calls the Gemini LLM with a detailed prompt including the schema and sources.
    *   Generates mapping rules for each target attribute, suggesting the best source, mapping type (Direct, Transformation, Metadata, Manual Review), transformation logic summary, and basic data quality checks.
    *   Applies minor rule-based refinement (e.g., ensuring 'NOT_FOUND' sources are flagged for manual review).
    *   Outputs the mappings as JSON.
6.  **Output:** The Gradio interface displays the outputs from each agent in dedicated tabs, using both JSON views and formatted DataFrames for readability. A summary and status updates are also provided.

---

## 🛠️ Technologies Used

*   **Language:** Python 3.10+
*   **AI Model:** Google Gemini (`gemini-1.5-flash` via `google-generativeai` SDK)
*   **UI Framework:** Gradio (`gradio`)
*   **Data Handling:** Pandas (`pandas`)
*   **Source Matching:** FuzzyWuzzy (`fuzzywuzzy`, `python-Levenshtein`)
*   **Deployment:** Hugging Face Spaces

---

## 🚀 Running Locally

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/iwantacrepe/Agentic-C360-Designer.git
    cd Agentic-C360-Designer
    ```
  

2.  **Create `requirements.txt`:**
    Create a file named `requirements.txt` in the root directory with the following content:
    ```txt
    google-generativeai
    gradio
    pandas
    fuzzywuzzy
    python-Levenshtein
    ```

3.  **Install Dependencies:**
    (Recommended: Use a virtual environment)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

4.  **Set API Key:**
    You need a Google AI (Gemini) API key. Set it as an environment variable (recommended):
    ```bash
    export GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
    ```
    Alternatively, you can replace the placeholder directly in the Python script (less secure).

5.  **Run the Application:**
    ```bash
    python app.py
    ```
   

6.  **Access:** Open your web browser and go to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).

---

## ⚠️ Limitations

*   **Prototype:** This is a proof-of-concept developed for a hackathon, not a production-ready system.
*   **Mock Data Catalog:** Source identification relies on a predefined Python dictionary (`MOCK_CATALOG`). It does not connect to a real data catalog.
*   **LLM Dependency:** Output quality heavily depends on the Gemini model's understanding and adherence to prompts. Results can vary and may require refinement. LLM errors (parsing, rate limits, safety) can occur.
*   **Basic Source Matching:** Fuzzy matching is better than exact matching but less sophisticated than semantic search or embedding-based approaches used in enterprise catalogs.
*   **Suggestive Mappings:** Transformation logic and quality checks are high-level suggestions and require detailed specification and validation by data engineers/stewards.
*   **Error Handling:** Basic error handling and retries are implemented, but complex edge cases might not be fully covered.

---

## ✨ Future Enhancements (Ideas)

*   Integrate with real Data Catalog APIs (e.g., Collibra, Alation, Azure Purview).
*   Implement semantic search/embeddings for more accurate source identification.
*   Generate more detailed or executable transformation logic (e.g., SQL snippets, Python code hints).
*   Incorporate feedback loops where agent outputs can be corrected and re-processed.
*   Add support for more complex schema types (nested structures, relationships).
*   Generate basic Data Quality rules based on findings.

---

## 🙏 Acknowledgements

*   Developed as part of the Accenture Hackathon.
*   Uses Google Gemini for its powerful generative AI capabilities.
*   Built with the excellent Gradio library for the interactive UI.
*   Hosted on Hugging Face Spaces.
