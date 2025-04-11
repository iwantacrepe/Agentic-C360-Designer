import google.generativeai as genai
import gradio as gr
import json
import os
import re
import time
import pandas as pd
from fuzzywuzzy import fuzz, process

print("Gradio version:", gr.__version__)
print("Google Generative AI version:", genai.__version__)
API_KEY = "AIzaSyBvLjAbdWz6v6_ii98B_j98IffX4TGrexM"

llm = None
initialization_error = None
if not API_KEY or API_KEY == "YOUR_ACTUAL_API_KEY_HERE":
    initialization_error = "API Key not found. Please set the GOOGLE_API_KEY environment variable or replace the placeholder in the script."
    print(f"❌ Error: {initialization_error}")
else:
    try:
        genai.configure(api_key=API_KEY)
        llm = genai.GenerativeModel('gemini-1.5-flash')
        llm.generate_content("test connection", generation_config=genai.types.GenerationConfig(temperature=0.1, candidate_count=1))
        print("Gemini Model configured and tested successfully.")
    except Exception as e:
        error_details = f"{type(e).__name__}: {e}"
        print(f"❌ Error configuring/testing Gemini API: {error_details}")
        initialization_error = f"Gemini API Error: {error_details}"
        llm = None

MIN_FUZZY_SCORE = 75
MOCK_CATALOG = {
    "customer_master": {
        "description": "Core customer demographic and identifying information.",
        "columns": {
            "customer_id": "Unique identifier for the customer (Primary Key)",
            "first_name": "Customer's first name",
            "last_name": "Customer's last name",
            "dob": "Date of birth (YYYY-MM-DD)",
            "gender": "Customer's gender (M/F/O/U)",
            "address_line1": "Primary address line",
            "city": "City of residence",
            "state_province": "State or province",
            "postal_code": "Postal or ZIP code",
            "country_code": "ISO 3166-1 alpha-2 country code",
            "email_primary": "Primary email address",
            "phone_mobile": "Primary mobile phone number",
            "join_date": "Date customer relationship began",
            "customer_segment": "Marketing segment assigned (e.g., High Value, Retail)",
            "marital_status": "Marital status (Single, Married, Divorced, etc.)",
            "employment_status": "Current employment status",
            "estimated_annual_income": "Estimated annual income in local currency",
            "credit_score_internal": "Bank's internal creditworthiness score"
        }
    },
    "account_details": {
        "description": "Information about customer accounts (Savings, Checking, etc.).",
        "columns": {
            "account_id": "Unique identifier for the account (Primary Key)",
            "customer_id": "Identifier linking to customer_master (Foreign Key)",
            "account_type": "Type of account (e.g., SAV, CHK, LOC)",
            "account_balance": "Current account balance",
            "currency_code": "ISO 4217 currency code",
            "account_open_date": "Date the account was opened",
            "account_status": "Status of the account (Active, Dormant, Closed)",
            "interest_rate_pa": "Annual interest rate (if applicable)",
            "overdraft_limit": "Overdraft limit amount (if applicable)",
            "avg_monthly_balance_3m": "Average monthly balance over the last 3 months",
            "last_activity_date": "Date of the last transaction or activity"
        }
    },
    "transaction_history": {
        "description": "Record of financial transactions.",
        "columns": {
            "transaction_id": "Unique identifier for the transaction (Primary Key)",
            "account_id": "Identifier linking to account_details (Foreign Key)",
            "transaction_timestamp": "Date and time of the transaction (UTC)",
            "transaction_amount": "Amount of the transaction (positive for credit, negative for debit)",
            "transaction_type": "Type (e.g., Deposit, Withdrawal, Fee, Transfer)",
            "transaction_description": "Text description of the transaction",
            "merchant_category_code": "MCC code for card transactions",
            "transaction_location": "Location description (e.g., ATM ID, City)",
            "channel": "Channel used (e.g., Mobile, Online, Branch, ATM)"
        }
    },
    "loan_data": {
        "description": "Information on customer loans.",
        "columns": {
            "loan_id": "Unique identifier for the loan (Primary Key)",
            "customer_id": "Identifier linking to customer_master (Foreign Key)",
            "loan_type": "Type of loan (e.g., Mortgage, Personal, Auto)",
            "principal_amount": "Original loan amount",
            "outstanding_balance": "Current outstanding balance",
            "loan_status": "Status (e.g., Active, Paid Off, Delinquent)",
            "origination_date": "Date the loan was originated",
            "maturity_date": "Date the loan is scheduled to be fully repaid",
            "interest_rate": "Annual interest rate",
            "collateral_description": "Description of collateral (if any)",
            "payment_frequency": "How often payments are due (e.g., Monthly)",
            "next_payment_due_date": "Date the next payment is due",
            "days_past_due": "Number of days the loan payment is overdue (0 if current)"
        }
    },
    "marketing_campaigns": {
        "description": "Records of marketing interactions with customers.",
        "columns": {
            "interaction_id": "Unique identifier for the interaction",
            "customer_id": "Identifier linking to customer_master (Foreign Key)",
            "campaign_id": "Identifier for the marketing campaign",
            "campaign_name": "Name of the marketing campaign",
            "interaction_datetime": "Date and time of the interaction",
            "channel": "Marketing channel (e.g., Email, SMS, Call)",
            "response_type": "Customer response (e.g., Clicked, Opened, Converted, No Response)",
            "offer_details": "Details of the offer presented"
        }
    },
     "support_interactions": {
        "description": "Log of customer support calls, chats, or emails.",
        "columns": {
            "case_id": "Unique identifier for the support case",
            "customer_id": "Identifier linking to customer_master (Foreign Key)",
            "interaction_start_time": "Timestamp when interaction began",
            "interaction_end_time": "Timestamp when interaction ended",
            "agent_id": "ID of the support agent",
            "reason_code": "Code indicating the reason for contact",
            "resolution_description": "Text description of the outcome",
            "customer_sentiment": "Sentiment analysis score (e.g., Positive, Negative, Neutral)",
            "follow_up_needed": "Boolean flag indicating if follow-up is required",
            "channel": "Support channel (e.g., Phone, Chat, Email)"
        }
    }
}

# --- Helper Functions ---

def clean_llm_response(text):
    """Removes markdown code blocks and trims whitespace."""
    if not isinstance(text, str): return "" # Handle non-string input
    text = re.sub(r'```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def parse_json_from_llm(text, agent_name):
    """Attempts to parse JSON from LLM output, handling potential errors."""
    cleaned_text = clean_llm_response(text)
    try:
        # Handle empty string case after cleaning
        if not cleaned_text:
             raise json.JSONDecodeError("Empty string cannot be parsed as JSON", "", 0)
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        error_msg = f"⚠️ Failed to parse JSON response from {agent_name}. Error: {e}. Raw response:\n'{cleaned_text}'" # Show quotes for clarity
        print(error_msg)
        return {"error": f"JSON Parse Error from {agent_name}", "details": str(e), "raw_response": cleaned_text}
    except Exception as e: # Catch other potential errors
        error_msg = f"⚠️ Unexpected error parsing response from {agent_name}. Error: {e}. Raw response:\n'{cleaned_text}'"
        print(error_msg)
        return {"error": f"Unexpected Parse Error from {agent_name}", "details": str(e), "raw_response": cleaned_text}

def generate_with_retry(prompt, agent_name, max_retries=2, initial_delay=2):
    """Calls the LLM with retries on failure."""
    if not llm:
        return {"error": f"{agent_name} cannot proceed: Gemini model not initialized."}

    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            print(f"🤖 [LLM Call - {agent_name}] Attempt {retries + 1}/{max_retries}")
            response = llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2, # Lower temp for more deterministic JSON/structure
                    candidate_count=1
                ),
                # Add safety settings if concerned about specific content blocks
                # safety_settings=[...]
            )

            # Enhanced check for response validity
            if not hasattr(response, 'text') or not response.text:
                # Check for blocked responses due to safety or other reasons
                block_reason = "Unknown"
                prompt_feedback_details = "N/A"
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    # Access block_reason safely, handle potential AttributeError if structure changes
                    block_reason = getattr(response.prompt_feedback, 'block_reason', 'Not specified')
                    prompt_feedback_details = str(response.prompt_feedback) # Get more details if available

                error_detail = f"LLM returned empty or blocked response. Block Reason: {block_reason}. Feedback: {prompt_feedback_details}"
                print(f"⚠️ [LLM Call - {agent_name}] {error_detail}")
                # Treat safety blocks as errors for this workflow usually
                raise ValueError(error_detail)


            print(f"✅ [LLM Call - {agent_name}] Successful.")
            parsed = parse_json_from_llm(response.text, agent_name)
            return parsed # Return the parsed JSON or the error structure from parse_json_from_llm

        except Exception as e:
            retries += 1
            error_type = type(e).__name__
            error_message = str(e)
            print(f"⚠️ [LLM Call - {agent_name}] Failed. Error Type: {error_type}. Details: {error_message}. Retrying in {delay}s...")
            if retries == max_retries:
                 final_error_msg = f"{agent_name} LLM call failed after {max_retries} attempts"
                 print(f"❌ [LLM Call - {agent_name}] Max retries reached. Giving up.")
                 return {"error": final_error_msg, "details": error_message, "error_type": error_type}
            time.sleep(delay)
            delay *= 2 # Exponential backoff

    # Should not be reached if loop logic is correct
    return {"error": f"{agent_name} unexpected error in retry loop."}


# Agent 1: Requirements Analyst (Modified Prompt & Placeholder)
def analyze_requirements_agent(use_case_text):
    """
    Analyzes use case using LLM. If unsure or LLM fails, generates plausible inferred requirements.
    """
    print("\n🧠 [Agent 1] Requirements Analysis - Running...")
    prompt = f"""
    Act as an expert Requirements Analyst for Retail Banking C360. Analyze the following business use case.

    Business Use Case:
    "{use_case_text}"

    Extract and structure the following information. **If any detail is ambiguous or missing, infer a common or plausible value typical for retail banking C360 use cases.** Avoid using phrases like "unknown", "error", or "requires review".
    1.  **primary_goal:** State the main business objective (e.g., "Improve Targeted Marketing", "Assess Churn Risk", "Enhance Customer Service View"). Infer if necessary.
    2.  **target_audience:** Identify likely consumers (e.g., "Marketing Team", "Risk Analytics Department", "Customer Service Representatives"). Infer if necessary.
    3.  **key_entities:** List core subjects (e.g., ["Customer", "Account", "Transaction"]). Default to common C360 entities if unclear.
    4.  **required_data_attributes:** List specific data points needed. If requirements are very vague, list common C360 attributes like "customer_id", "full_name", "email", "phone", "account_balance", "last_activity_date". Use descriptive names.
    5.  **time_sensitivity:** Note timeframes or refresh frequency. Default to "Near Real-time" or "Daily Refresh" if not specified.
    6.  **key_metrics_or_calculations:** Identify calculations needed. If none are clear, list common ones like ["customer_age", "account_tenure", "total_balance"].
    7.  **filters_or_segmentation:** List criteria for data subsets. If none specified, list common examples like ["active_customers", "specific_segment"].

    Output results strictly as a single JSON object with snake_case keys. Ensure list values contain only strings. **Do not indicate uncertainty in the output JSON.**
    """
    result = generate_with_retry(prompt, "Agent 1: Requirements")

    # --- Plausible Placeholder Logic ---
    if isinstance(result, dict) and "error" in result:
        error_detail = result.get('details', result['error'])
        print(f"⚠️ [Agent 1] LLM call failed. Generating plausible placeholder requirements. Error: {error_detail}")
        # Generate more narrative/default placeholders
        placeholder = {
            "primary_goal": "Enhance Customer Understanding (Inferred Default)",
            "target_audience": "General Analytics / Marketing (Inferred Default)",
            "key_entities": ["Customer", "Account"], # Common defaults
            "required_data_attributes": ["customer_id", "full_name", "account_balance", "last_activity_date"], # Common defaults
            "time_sensitivity": "Daily Refresh (Assumed)",
            "key_metrics_or_calculations": ["customer_age (if DOB available)"],
            "filters_or_segmentation": ["active_customers (standard filter)"],
            "llm_placeholder_generated": True,
            "placeholder_reason": f"LLM Failure: {error_detail[:150]}"
        }
        return placeholder
    # --- End Placeholder Logic ---

    if isinstance(result, dict) and "error" not in result:
        print("✅ [Agent 1] Requirements Analysis - Complete.")
    else:
         print(f"❌ [Agent 1] Requirements Analysis - Failed unexpectedly after placeholder check. Result: {result}")
    return result


# Agent 2: Data Architect (+ Enrichment) (Modified Prompt & Placeholder)
def design_and_enrich_schema_agent(structured_requirements):
    """
    Designs schema using LLM. If unsure or LLM fails, generates plausible schema. Includes enrichment.
    """
    print("\n🏗️ [Agent 2] Data Product Schema Design & Enrichment - Running...")
    if not isinstance(structured_requirements, dict) or "required_data_attributes" not in structured_requirements:
        error_msg = "Invalid input: Structured requirements missing or malformed."
        print(f"❌ [Agent 2] Failed - {error_msg}")
        # Return placeholder schema for bad input
        return {
            "data_product_name": "ERROR_INVALID_INPUT_SCHEMA",
            "schema": [
                {"attribute_name": "customer_id", "data_type": "STRING", "description": "Default Customer ID"},
                {"attribute_name": "error_flag", "data_type": "BOOLEAN", "description": error_msg}
            ],
            "llm_placeholder_generated": True,
            "placeholder_reason": "Invalid Input to Agent 2"
         }

    # --- Part 1: Initial Schema Design via LLM (Modified Prompt) ---
    prompt_design = f"""
    Act as a meticulous Data Architect for Retail Banking C360. Use the requirements below to design a flat target data product schema.

    Requirements:
    ```json
    {json.dumps(structured_requirements, indent=2)}
    ```

    Instructions:
    1.  Suggest a descriptive `data_product_name` (snake_case). Infer a name like `c360_standard_profile` if requirements are unclear.
    2.  Create a `schema` list based *primarily* on 'required_data_attributes'.
    3.  For each attribute: define `attribute_name` (standardized snake_case), `data_type` (standard SQL - STRING, DECIMAL, DATE, TIMESTAMP, BOOLEAN, INTEGER), and `description`.
    4.  **If requirements for attributes are vague, infer common C360 attributes and reasonable data types/descriptions.** Avoid indicating uncertainty in the output. Make your best guess for data types (use STRING for IDs, DECIMAL for money).

    Output results strictly as a JSON object with keys 'data_product_name' and 'schema'. No explanations outside JSON.
    """
    initial_schema = generate_with_retry(prompt_design, "Agent 2: Initial Design")

    # --- Plausible Placeholder Logic for Initial Design Failure ---
    if isinstance(initial_schema, dict) and "error" in initial_schema:
        error_detail = initial_schema.get('details', initial_schema['error'])
        print(f"⚠️ [Agent 2] LLM call for initial design failed. Generating plausible placeholder schema. Error: {error_detail}")
        # Generate a generic C360 schema as fallback
        placeholder_schema_list = [
            {"attribute_name": "customer_id", "data_type": "STRING", "description": "Customer Identifier (Placeholder)"},
            {"attribute_name": "full_name", "data_type": "STRING", "description": "Customer Full Name (Placeholder)"},
            {"attribute_name": "primary_email", "data_type": "STRING", "description": "Primary Email (Placeholder)"},
            {"attribute_name": "total_balance", "data_type": "DECIMAL(18,2)", "description": "Total Account Balance (Placeholder)"},
            {"attribute_name": "last_activity", "data_type": "DATE", "description": "Last Activity Date (Placeholder)"}
        ]
        placeholder = {
            "data_product_name": "c360_default_view_LLM_Error",
            "schema": placeholder_schema_list,
            "llm_placeholder_generated": True,
            "placeholder_reason": f"LLM Error during initial design: {error_detail[:150]}"
        }
        # Skip enrichment for this placeholder
        print("✅ [Agent 2] Schema Design & Enrichment - Complete (using placeholder for initial design).")
        return placeholder
    # --- End Placeholder Logic ---

    # --- Part 2: Rule-Based Enrichment (Proceed only if initial design succeeded) ---
    try:
        if not isinstance(initial_schema, dict): raise TypeError("Initial schema not a dict")
        enriched_schema = initial_schema.copy()
        schema_attributes = enriched_schema.get("schema", [])
        if not isinstance(schema_attributes, list): raise TypeError("Schema key not a list")

        print("  🔧 [Agent 2] Enriching and Validating Schema...")
        # (Enrichment logic remains exactly the same as previous correct version)
        # Add standard fields
        standard_attrs = {
             "c360_master_customer_id": {"data_type": "STRING", "description": "Consolidated unique identifier for the customer across systems.", "added_by": "enrichment"},
             "data_product_load_ts": {"data_type": "TIMESTAMP", "description": "Timestamp when the record was loaded into this data product.", "added_by": "enrichment"},
             "source_system_tracker": {"data_type": "STRING", "description": "Indicator of the primary source system(s) contributing to this record.", "added_by": "enrichment"}
        }
        existing_names_lower = {attr.get("attribute_name", "").lower() for attr in schema_attributes if isinstance(attr, dict)}
        added_meta = False
        new_attrs_to_add = []
        for name, details in standard_attrs.items():
             if name.lower() not in existing_names_lower:
                  new_attrs_to_add.append({"attribute_name": name, **details})
                  added_meta = True
        schema_attributes = new_attrs_to_add + schema_attributes
        if added_meta: print(f"    + Added {len(new_attrs_to_add)} standard metadata/C360 ID attribute(s).")
        # Suggest PK
        pk_found = False
        potential_pk_candidates = ["customer_id", "c360_master_customer_id", "account_id", "transaction_id", "loan_id", "case_id", "interaction_id"]
        suggested_pk = None
        for pk_candidate in potential_pk_candidates:
             for attr in schema_attributes:
                  if isinstance(attr, dict):
                       attr_name_lower = attr.get("attribute_name", "").lower()
                       if pk_candidate == attr_name_lower:
                            if not attr.get("is_primary_key"):
                                attr["is_primary_key"] = True
                                attr["description"] = (attr.get("description", "") or "") + " (Suggested Primary Key)"
                                pk_found = True
                                suggested_pk = attr['attribute_name']
                                print(f"    🔑 Suggested '{suggested_pk}' as Primary Key.")
                                break
             if pk_found: break
        if not pk_found: print("    ⚠️ No obvious Primary Key candidate found based on naming conventions.")

        enriched_schema["schema"] = schema_attributes
        print("✅ [Agent 2] Schema Design & Enrichment - Complete.")
        return enriched_schema

    except Exception as e: # Catch errors during enrichment
        error_msg = f"Error during schema enrichment: {type(e).__name__}: {e}"
        print(f"❌ [Agent 2] Failed during enrichment - {error_msg}")
        if isinstance(initial_schema, dict):
            initial_schema["enrichment_error"] = error_msg
            initial_schema["llm_placeholder_generated"] = True # Mark as needing review
            initial_schema["placeholder_reason"] = f"Enrichment failed: {error_msg[:100]}"
            return initial_schema
        else:
             return {"error": "Schema enrichment failed", "details": error_msg, "llm_placeholder_generated": True, "placeholder_reason": "Enrichment Failed"}
# Agent 3: Source Identification Specialist (MODIFIED with LLM Inference Fallback)
def find_potential_sources(target_attributes):
    """
    Identifies potential source locations for target attributes using fuzzy matching
    against a mock catalog. If no strong match is found, uses LLM to infer a
    plausible source.

    Args:
        target_attributes (list): A list of target attribute names (strings) to find sources for.

    Returns:
        dict: A dictionary where keys are target attributes and values are lists of
              tuples: (potential_source_location, confidence_score, source_description).
              Confidence score is fuzzy match score (0-100) or a low fixed value (e.g., 10) for LLM inferences.
              Includes an 'error' key if critical issues occur during processing.
    """
    print("\n🔍 [Agent 3] Source Identification - Running...")
    if not target_attributes:
        print("  [Agent 3] No target attributes provided to search for.")
        return {"potential_sources": {}, "error": None}

    # 1. Prepare the source catalog for searching (flatten table/column names)
    all_sources = {}
    for table_name, details in MOCK_CATALOG.items():
        if isinstance(details, dict) and "columns" in details and isinstance(details["columns"], dict):
            for col_name, col_desc in details["columns"].items():
                full_path = f"{table_name}.{col_name}"
                # Store path -> (description, original_table, original_column)
                all_sources[full_path] = (col_desc or "No description", table_name, col_name)
        else:
             print(f"⚠️ [Agent 3] Skipping malformed entry in MOCK_CATALOG: {table_name}")

    if not all_sources:
         error_msg = "[Agent 3] Critical Error: Mock catalog is empty or could not be parsed."
         print(f"❌ {error_msg}")
         return {"potential_sources": {}, "error": error_msg}

    # 2. Find potential matches for each target attribute
    potential_matches = {}
    source_paths = list(all_sources.keys()) # List of "table.column" strings

    for attribute in target_attributes:
        if not attribute or not isinstance(attribute, str):
            print(f"  [Agent 3] Skipping invalid target attribute: {attribute}")
            continue

        print(f"  🔎 Searching for source for: '{attribute}'...")
        potential_matches[attribute] = []

        # Use fuzzywuzzy's process.extractOne to find the best match in the flattened paths
        # We compare the target attribute name against the combined "table.column" path AND just the column name part
        # to handle cases where the target name strongly matches a column name but not the table prefix.
        match_path, score_path = process.extractOne(attribute, source_paths, scorer=fuzz.WRatio) or (None, 0)
        # Extract column names for separate matching
        column_names_map = {col_name: path for path, (_, _, col_name) in all_sources.items()}
        column_names_only = list(column_names_map.keys())
        match_col, score_col = process.extractOne(attribute, column_names_only, scorer=fuzz.WRatio) or (None, 0)

        best_match_source = None
        best_score = 0

        # Decide which match is better (path vs column-only)
        if score_path >= score_col and score_path > 0:
            best_match_source = match_path
            best_score = score_path
            print(f"    Fuzzy Match (Path): '{match_path}' (Score: {score_path})")
        elif score_col > score_path and score_col > 0:
            best_match_source = column_names_map[match_col] # Get full path from column name match
            best_score = score_col
            print(f"    Fuzzy Match (Column): '{match_col}' in '{best_match_source}' (Score: {score_col})")
        else:
            print(f"    No decent fuzzy match found for '{attribute}'.")


        # 3. Evaluate match and decide whether to use catalog match or infer with LLM
        if best_match_source and best_score >= MIN_FUZZY_SCORE:
            # Good match found in catalog
            source_desc, _, _ = all_sources[best_match_source]
            potential_matches[attribute].append((best_match_source, best_score, source_desc))
            print(f"    ✅ Using catalog source: {best_match_source} (Score: {best_score})")
        else:
            # No good match found - attempt LLM inference
            print(f"    🤔 No strong catalog match (Best score: {best_score} < {MIN_FUZZY_SCORE}). Attempting LLM inference for '{attribute}'...")

            # Construct prompt for LLM inference
            prompt_infer = f"""
            Act as a Data Catalog Expert for Retail Banking C360.
            The attribute '{attribute}' was requested for a data product, but it wasn't found with high confidence in our known data catalog.

            Based on standard retail banking data systems (like Core Banking, CRM, Loan Origination, Marketing Platforms, Support Systems), infer the *single most plausible* source for this attribute.

            Provide:
            1. `inferred_source_location`: Your best guess for the source system/table and column name (e.g., "crm_interactions.last_contact_date", "core_banking_customer.primary_address_line_1", "loan_master.collateral_value"). Be specific but concise.
            2. `justification`: A very brief explanation for your choice (e.g., "Typically stored in CRM contact history", "Standard customer address field", "Common field in loan details").

            Output *only* a single JSON object with these two keys. Do not add explanations outside the JSON.
            Example:
            {{
              "inferred_source_location": "customer_master.email_primary",
              "justification": "Standard field for customer primary email address."
            }}
            """
            llm_inference_result = generate_with_retry(prompt_infer, "Agent 3: Source Inference")

            if isinstance(llm_inference_result, dict) and "error" not in llm_inference_result and "inferred_source_location" in llm_inference_result:
                # LLM Inference Successful
                inferred_loc = llm_inference_result.get("inferred_source_location", "LLM_INFERENCE_MISSING_KEY")
                justification = llm_inference_result.get("justification", "No justification provided by LLM.")
                inferred_source_path = f"[INFERRED] {inferred_loc}" # Mark as inferred
                inference_confidence = 10 # Assign a low, fixed confidence score for inferred results
                potential_matches[attribute].append((inferred_source_path, inference_confidence, f"LLM Suggestion: {justification}"))
                print(f"    🤖 LLM Inferred Source: {inferred_loc} (Confidence: {inference_confidence} - Inferred)")
            else:
                # LLM Inference Failed
                error_detail = llm_inference_result.get('details', llm_inference_result.get('error', 'Unknown LLM inference error'))
                print(f"    ⚠️ LLM inference failed for '{attribute}'. Error: {error_detail}")
                # Append a specific placeholder indicating inference failure
                potential_matches[attribute].append(("[INFERENCE_FAILED]", 0, f"LLM could not infer source. Reason: {str(error_detail)[:100]}"))


    print("✅ [Agent 3] Source Identification - Complete.")
    return {"potential_sources": potential_matches, "error": None}


# Agent 4: Mapping Specialist (+ Refinement) (Modified Prompt & Placeholder)
def generate_and_refine_mappings_agent(schema_design, potential_sources):
    """
    Generates mappings using LLM. If unsure or LLM fails, generates plausible mappings.
    """
    print("\n🗺️ [Agent 4] Mapping Generation & Refinement - Running...")
    # --- Input Validation ---
    schema_ok = isinstance(schema_design, dict) and "schema" in schema_design and isinstance(schema_design["schema"], list)
    sources_ok = isinstance(potential_sources, dict) and "potential_sources" in potential_sources and isinstance(potential_sources["potential_sources"], dict)

    if not schema_ok or not sources_ok:
         error_msg = "Invalid input: Schema design or potential sources missing/malformed for mapping."
         print(f"❌ [Agent 4] Failed - {error_msg}")
         # Generate plausible placeholder based on schema if available
         placeholder_mappings = []
         if schema_ok:
              for attr_entry in schema_design.get("schema", []):
                   if isinstance(attr_entry, dict) and attr_entry.get("attribute_name"):
                       map_type = "METADATA" if attr_entry.get("added_by") == "enrichment" else "TRANSFORMATION" # Guess type
                       logic = "Populated by ETL (Assumed)" if map_type == "METADATA" else "Standard transformation assumed (Input Error)"
                       placeholder_mappings.append({
                           "target_attribute": attr_entry["attribute_name"],
                           "best_potential_source": "N/A - Input Error",
                           "mapping_type": map_type,
                           "transformation_logic_summary": logic,
                           "data_quality_checks": ["Verify due to input error"]})
         return {"mappings": placeholder_mappings, "llm_placeholder_generated": True, "placeholder_reason": "Invalid Input to Agent 4"}

    # --- Part 1: Initial Mapping via LLM (Modified Prompt) ---
    prompt_map = f"""
    Act as an expert Data Mapping Specialist for Retail Banking C360. Create source-to-target mapping specifications based on the schema and potential sources.

    Target Schema:
    ```json
    {json.dumps(schema_design, indent=2)}
    ```
    Potential Sources (Target Attribute -> List of [Source Location, Score, Description]):
    ```json
    {json.dumps(potential_sources['potential_sources'], indent=2)}
    ```
    Instructions:
    For each attribute in the target schema:
    1.  Analyze the target attribute. For enrichment fields ('added_by' = 'enrichment'), use 'METADATA' mapping type and "Populated by ETL process" logic.
    2.  Examine potential source candidates. Select the *single best* source (highest score/relevance). List multiple sources *only if essential* for a transformation (e.g., name concatenation).
    3.  **Determine `mapping_type`. Avoid using 'MANUAL_REVIEW'.** Choose 'DIRECT' if compatible. Choose 'TRANSFORMATION' if any calculation, formatting, casting, lookup, or combination is needed. Choose 'METADATA' for enrichment fields.
    4.  **If no suitable source is found or confidence is low, make a plausible assumption:** either guess the most likely source candidate (even if score is low) and set type to 'DIRECT' or 'TRANSFORMATION' (e.g., simple casting), OR assume a common transformation (e.g., calculate age, set default value) and set type to 'TRANSFORMATION'.
    5.  Provide a brief `transformation_logic_summary`. If you made an assumption due to lack of clear source/logic, briefly note the assumption (e.g., "Assumed direct map from primary customer table", "Standard age calculation applied", "Default value assigned").
    6.  Suggest basic `data_quality_checks` (optional list of 1-2 strings).

    Output results strictly as a JSON object with a single key "mappings" holding a list of mapping entry objects. Each object needs keys: "target_attribute", "best_potential_source", "mapping_type", "transformation_logic_summary", "data_quality_checks" (use [] if none). **Do NOT output 'MANUAL_REVIEW'. Always provide a mapping.**
    """
    initial_mappings = generate_with_retry(prompt_map, "Agent 4: Initial Mapping")

    # --- Plausible Placeholder Logic for Initial Mapping Failure ---
    if isinstance(initial_mappings, dict) and "error" in initial_mappings:
        error_detail = initial_mappings.get('details', initial_mappings['error'])
        print(f"⚠️ [Agent 4] LLM call for initial mapping failed. Generating plausible placeholder mappings. Error: {error_detail}")
        placeholder_mappings = []
        # Create plausible placeholder for each attribute in the input schema
        for attr in schema_design.get("schema", []):
             if isinstance(attr, dict) and attr.get("attribute_name"):
                 target_attribute_name = attr["attribute_name"]
                 # Get best guess source from Agent 3 output
                 best_source_guess = "N/A - LLM Mapping Error"
                 src_candidates = potential_sources.get("potential_sources", {}).get(target_attribute_name, [])
                 if src_candidates and isinstance(src_candidates[0], (list, tuple)) and len(src_candidates[0]) > 0:
                      best_source_guess = f"{src_candidates[0][0]} (Score: {src_candidates[0][1]})"
                      if "NOT_FOUND" in src_candidates[0][0]: best_source_guess = "NOT_FOUND_IN_CATALOG (LLM Failed)" # Make clearer

                 map_type = "TRANSFORMATION" # Default guess
                 logic = f"Standard transformation assumed (LLM Failed: {error_detail[:50]})"
                 dq = ["Verify mapping logic"]

                 # Handle metadata fields
                 if attr.get("added_by") == "enrichment":
                      map_type = "METADATA"
                      logic = "Populated by ETL process (Placeholder)"
                      best_source_guess = "ETL Process"
                      dq = []
                 elif "NOT_FOUND" in best_source_guess:
                      logic = "No source found, default/logic needed (LLM Failed)"

                 placeholder_mappings.append({
                     "target_attribute": target_attribute_name,
                     "best_potential_source": best_source_guess,
                     "mapping_type": map_type,
                     "transformation_logic_summary": logic,
                     "data_quality_checks": dq
                 })

        placeholder = {"mappings": placeholder_mappings, "llm_placeholder_generated": True, "placeholder_reason": f"LLM Error during mapping: {error_detail[:150]}"}
        print("✅ [Agent 4] Mapping Generation & Refinement - Complete (using placeholder).")
        return placeholder
    # --- End Placeholder Logic ---

    # --- Part 2: Refinement (Proceed only if initial mapping succeeded) ---
    try:
        if not isinstance(initial_mappings, dict): raise TypeError("Initial mapping not a dict")
        refined_mappings = initial_mappings.copy()
        if "mappings" not in refined_mappings or not isinstance(refined_mappings["mappings"], list): raise ValueError("Mappings list missing/invalid")

        print("  🔧 [Agent 4] Refining Mappings (Ensuring completeness)...")
        processed_targets = set()
        valid_mappings = []

        for i, mapping in enumerate(refined_mappings["mappings"]):
             if not isinstance(mapping, dict): continue # Skip invalid entries
             target_attr = mapping.get("target_attribute")
             if not target_attr: continue
             processed_targets.add(target_attr)

             # Ensure essential keys exist (provide defaults if missing, matching placeholder style)
             mapping.setdefault("best_potential_source", "N/A - Missing")
             mapping.setdefault("mapping_type", "TRANSFORMATION") # Default to Transformation if missing
             mapping.setdefault("transformation_logic_summary", "Logic missing - Review needed")
             mapping.setdefault("data_quality_checks", [])
             if not isinstance(mapping["data_quality_checks"], list): mapping["data_quality_checks"] = []

             # Force replacement of any lingering 'MANUAL_REVIEW' if LLM disobeyed prompt
             if mapping["mapping_type"] == "MANUAL_REVIEW":
                  print(f"    Found lingering 'MANUAL_REVIEW' for {target_attr}, replacing with plausible default.")
                  mapping["mapping_type"] = "TRANSFORMATION"
                  mapping["transformation_logic_summary"] += " (Assumed standard transformation)"
                  mapping["data_quality_checks"].append("Verify assumed logic")


             valid_mappings.append(mapping)

        refined_mappings["mappings"] = valid_mappings

        # Check for missing target attributes and add plausible placeholders
        missing_targets = []
        schema_list = schema_design.get("schema", [])
        if isinstance(schema_list, list):
            for schema_attr in schema_list:
                if isinstance(schema_attr, dict):
                    target_name = schema_attr.get("attribute_name")
                    if target_name and target_name not in processed_targets:
                        missing_targets.append(target_name)
                        # Add plausible placeholder mapping
                        map_type = "METADATA" if schema_attr.get("added_by") == "enrichment" else "TRANSFORMATION"
                        logic = "Populated by ETL (Assumed)" if map_type == "METADATA" else "Standard transformation assumed (Missing from LLM)"
                        best_source = "ETL Process" if map_type == "METADATA" else "N/A - Missing"
                        refined_mappings["mappings"].append({
                            "target_attribute": target_name, "best_potential_source": best_source,
                            "mapping_type": map_type, "transformation_logic_summary": logic,
                            "data_quality_checks": ["Verify mapping"]})
        else: print("⚠️ Cannot check for missing mappings due to invalid schema.")

        if missing_targets: print(f"    ⚠️ Added plausible placeholder mappings for {len(missing_targets)} attributes missing from LLM output: {missing_targets}")

        print("✅ [Agent 4] Mapping Generation & Refinement - Complete.")
        return refined_mappings

    except Exception as e: # Catch errors during refinement
         error_msg = f"Error during mapping refinement: {type(e).__name__}: {e}"
         print(f"❌ [Agent 4] Failed during refinement - {error_msg}")
         if isinstance(initial_mappings, dict): # Return initial + error if possible
             initial_mappings["refinement_error"] = error_msg
             initial_mappings["llm_placeholder_generated"] = True # Mark as needing review
             initial_mappings["placeholder_reason"] = f"Refinement failed: {error_msg[:100]}"
             return initial_mappings
         else: return {"error": "Mapping refinement failed", "details": error_msg, "llm_placeholder_generated": True, "placeholder_reason": "Refinement Failed"}


# --- Orchestrator (Modified Warning Handling) ---
def run_agentic_workflow(use_case_text, progress=gr.Progress()):
    """ Orchestrates workflow, adding warnings when placeholders are used. """
    print("\n🚀 --- Starting Agentic Workflow --- 🚀")
    start_time = time.time()
    output = { "status": "Starting...", "requirements": None, "schema": None, "sources": None, "mappings": None, "summary": None, "warnings": [], "final_message": ""}
    unique_warnings = set() # Use a set to track unique warning messages

    def add_warning(message):
        """Adds a unique warning message to the output."""
        if message not in unique_warnings:
            output["warnings"].append(message)
            unique_warnings.add(message)

    def update_status(stage_name, stage_index, total_stages, error=None):
        # (Status update logic remains the same as previous correct version)
        progress_fraction = stage_index / total_stages
        status_message = f"Running: {stage_name} ({stage_index}/{total_stages})"
        if error:
             status_message = f"Error in: {stage_name}"
             output["status"] = status_message
             add_warning(f"Failed at {stage_name}: {error}") # Use add_warning
             print(f" Workflow failed at stage: {stage_name}")
             progress(progress_fraction, desc=f"{status_message}")
        else:
             output["status"] = status_message
             print(f"Stage Complete: {stage_name}")
             progress(progress_fraction, desc=output["status"])


    total_stages = 4
    current_stage = 0
    # --- Pre-checks --- (Remain the same)
    if not use_case_text: # ... (return error) ...
        output["final_message"] = "Please enter a business use case description."
        output["status"] = "Failed (Input Missing)"; progress(0, desc="Input Missing"); return output
    if not llm: # ... (return error) ...
        output["final_message"] = f"Critical Error: Gemini model failed to initialize: {initialization_error}";
        output["status"] = "Failed (LLM Init Error)"; progress(0, desc="LLM Error"); return output

    # --- Agent 1 ---
    current_stage = 1
    update_status("Requirements Analysis", current_stage, total_stages)
    req_result = analyze_requirements_agent(use_case_text)
    output["requirements"] = req_result
    if isinstance(req_result, dict) and req_result.get("llm_placeholder_generated"):
        add_warning(f"Agent 1: Using inferred/default requirements due to LLM issue ({req_result.get('placeholder_reason', 'Unknown')[:50]}...).")

    # --- Agent 2 ---
    current_stage = 2
    update_status("Schema Design & Enrichment", current_stage, total_stages)
    schema_result = design_and_enrich_schema_agent(req_result) # Pass potentially placeholder reqs
    output["schema"] = schema_result
    if isinstance(schema_result, dict) and schema_result.get("llm_placeholder_generated"):
         add_warning(f"Agent 2: Using inferred/default schema. Reason: {schema_result.get('placeholder_reason', 'Unknown')[:50]}...")

    # --- Agent 3 --- (Critical path, stops on error)
    current_stage = 3
    update_status("Source Identification", current_stage, total_stages)
    target_attributes = []
    agent_3_error_msg = None
    sources_result = {"potential_sources": {}, "error": None}
    try:
        if isinstance(schema_result, dict) and "schema" in schema_result and isinstance(schema_result["schema"], list):
            target_attributes = [attr.get("attribute_name") for attr in schema_result["schema"] if isinstance(attr, dict) and attr.get("attribute_name")]
            if not target_attributes: add_warning("Agent 3: No non-metadata attributes found in schema to search sources for.")
        else: raise ValueError("Schema result invalid for source identification.")

        sources_result = find_potential_sources(target_attributes)
        output["sources"] = sources_result
        if isinstance(sources_result, dict) and not sources_result.get("potential_sources") and target_attributes:
             add_warning("Agent 3: Source Identification yielded no potential sources.")

    except Exception as e: # Agent 3 error is critical
        agent_3_error_msg = f"Critical Error during Source Identification: {type(e).__name__}: {e}"
        print(f" {agent_3_error_msg}")
        update_status("Source Identification", current_stage, total_stages, error=agent_3_error_msg)
        output["final_message"] = "Workflow stopped: " + agent_3_error_msg
        output["sources"] = {"error": agent_3_error_msg}
        return output # STOP

    # --- Agent 4 ---
    current_stage = 4
    update_status("Mapping Generation & Refinement", current_stage, total_stages)
    mappings_result = generate_and_refine_mappings_agent(schema_result, sources_result)
    output["mappings"] = mappings_result
    if isinstance(mappings_result, dict) and mappings_result.get("llm_placeholder_generated"):
         add_warning(f"Agent 4: Using inferred/default mappings. Reason: {mappings_result.get('placeholder_reason', 'Unknown')[:50]}...")

    # Check for manual review flags (still useful even if we tried to avoid them)
    manual_review_count = 0
    if isinstance(mappings_result, dict) and "mappings" in mappings_result and isinstance(mappings_result["mappings"], list):
        for item in mappings_result["mappings"]:
             # Count items where logic explicitly mentions review/assumption or if type is still MANUAL_REVIEW
             if isinstance(item, dict):
                  logic = item.get("transformation_logic_summary", "").lower()
                  if item.get("mapping_type") == "MANUAL_REVIEW" or "review" in logic or "assume" in logic or "missing" in logic:
                       manual_review_count += 1
        if manual_review_count > 0:
             add_warning(f"Agent 4: {manual_review_count} mapping(s) may require closer review due to assumptions or missing info.")


    # --- Final Summary & Wrap-up ---
    end_time = time.time()
    duration = end_time - start_time
    final_status = "Completed" + (" with Inferred Data/Warnings" if output["warnings"] else "") # Adjusted status
    output["status"] = final_status
    progress(1.0, desc=output["status"])
    output["final_message"] = f"✅ Agentic workflow {final_status} in {duration:.2f} seconds."
    print(f"🏁 --- Agentic Workflow {final_status} ({duration:.2f}s) --- 🏁")

    # Generate Summary (Remains the same logic as previous version)
    summary_lines = []
    if isinstance(req_result, dict) and "error" not in req_result and not req_result.get("llm_placeholder_generated"):
         summary_lines.append(f"**Data Product Suggestion for Use Case:** {req_result.get('primary_goal', 'N/A')}")
         summary_lines.append(f"**Target Audience:** {req_result.get('target_audience', 'N/A')}")
    elif isinstance(req_result, dict) and req_result.get("llm_placeholder_generated"):
         summary_lines.append("**Requirements:** Using inferred/default data (LLM Issue).")
    else: summary_lines.append("**Requirements analysis failed or produced invalid output.**")
    if isinstance(schema_result, dict) and "error" not in schema_result and not schema_result.get("llm_placeholder_generated"):
        summary_lines.append(f"**Suggested Product Name:** `{schema_result.get('data_product_name', 'N/A')}`")
        num_attrs = len([a for a in schema_result.get("schema", []) if isinstance(a, dict)])
        summary_lines.append(f"**Total Attributes Proposed:** {num_attrs}")
    elif isinstance(schema_result, dict) and schema_result.get("llm_placeholder_generated"):
        summary_lines.append(f"**Schema:** Using inferred/default data. Reason: {schema_result.get('placeholder_reason', 'N/A')[:50]}...")
        num_attrs = len([a for a in schema_result.get("schema", []) if isinstance(a, dict)])
        summary_lines.append(f"**Total Attributes (Placeholder):** {num_attrs}")
    else: summary_lines.append("**Schema design failed or produced invalid output.**")
    if output["warnings"]:
         summary_lines.append(f"\n**⚠️ Key Warnings/Action Items ({len(output['warnings'])}):**")
         max_summary_warnings = 7
         displayed_warnings = set()
         warnings_to_show = [w for w in output["warnings"] if w not in displayed_warnings and not displayed_warnings.add(w)]
         for i, warn in enumerate(warnings_to_show):
              if i < max_summary_warnings: summary_lines.append(f"- {warn}")
              elif i == max_summary_warnings: summary_lines.append(f"- ... and {len(warnings_to_show) - max_summary_warnings} more unique warnings (see details/logs).")
              break
    else: summary_lines.append("\n**✅ No major warnings detected during automated processing.**")
    summary_lines.append("\n**Next Steps:** Manually review schema and mappings (esp. where inferred/default values used), define ingress/egress, certify.")
    output["summary"] = "\n".join(summary_lines)

    return output


# --- Gradio UI ---
# (Gradio UI definition (`css`, `gr.Blocks`, components) remains exactly the same as the previous correct version)
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; max-width: 95%; margin: auto;}
.gr-button {
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.1s ease; 
}
.gr-button:hover {
    transform: translateY(-1px);
    box-shadow: 3px 3px 7px rgba(0,0,0,0.15);
}


#submit_button {
    background-color: #28a745; 
    color: white;
    border: 1px solid #218838; 
}
#submit_button:hover {
    background-color: #218838; 
    border-color: #1e7e34;
}
.warning { background-color: #fffbe6; border: 1px solid #ffe58f; padding: 10px; border-radius: 4px; color: #d46b08; margin-bottom: 10px;}
.error { background-color: #fff1f0; border: 1px solid #ffa39e; padding: 10px; border-radius: 4px; color: #cf1322; margin-bottom: 10px;}
.summary { background-color: #f0f5ff; border: 1px solid #adc6ff; padding: 15px; border-radius: 4px; color: #1d39c4; margin-top: 15px; line-height: 1.6;}
#status_display { font-weight: bold; margin-top: 10px; padding: 8px; border-radius: 4px; text-align: center; transition: background-color 0.3s ease, color 0.3s ease;}
#status_display.idle { background-color: #f0f0f0; color: #595959; }
#status_display.running { background-color: #e6f7ff; color: #096dd9; }
#status_display.completed { background-color: #f6ffed; color: #389e0d; }
#status_display.error { background-color: #fff1f0; color: #cf1322; }
.gr-tabitem { padding: 10px; }
.gr-dataframe table { width: 100%; font-size: 0.9em; } 
.gr-dataframe th { background-color: #f0f5ff; } 
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), css=css) as demo:
    gr.Markdown(
        """
        # 🏦 Enhanced Agentic Customer 360 Data Product Designer
        *Enter a retail banking business use case. A multi-agent system, powered by LLM, will analyze, design, identify sources, and generate mappings.*
        """
    )
    with gr.Row(): # Input Row
        use_case_input = gr.Textbox(
            lines=6,
            label="Enter Business Use Case Description Here:",
            placeholder="e.g., 'Create a dataset for a churn prediction model. We need customer demographics (age, location), account details (number of accounts, avg balance last 6m, account tenure), transaction patterns (frequency, avg amount last 3m), loan information (active loans count, delinquency status), and recent support interaction frequency.'",
            scale=3
        )
        with gr.Column(scale=1):
            submit_button = gr.Button("🚀 Generate Data Product Design", elem_id="submit_button")
            status_output = gr.Textbox(label="Workflow Status", interactive=False, elem_id="status_display", value="Idle", elem_classes=["idle"])
            summary_output = gr.Markdown(elem_id="summary", value="*Summary will appear here after processing...*")
    with gr.Accordion("📊 Detailed Agent Outputs", open=False): # Output Accordion
        with gr.Tabs():
            with gr.TabItem("🧠 Agent 1: Requirements"):
                 requirements_output_df = gr.DataFrame( label="Requirements Analysis Summary Table", interactive=False, wrap=True)
                 requirements_output_json = gr.JSON(label="Raw Requirements JSON (Full Detail)")
            with gr.TabItem("🏗️ Agent 2: Schema Design"):
                 schema_output_df = gr.DataFrame(label="Recommended & Enriched Data Product Schema", headers=["attribute_name", "data_type", "description", "is_primary_key", "added_by"], interactive=False, wrap=True)
                 schema_output_json = gr.JSON(label="Raw Schema JSON")
            with gr.TabItem("🔍 Agent 3: Source Identification"):
                 sources_output_df = gr.DataFrame( label="Potential Source Candidates (Flattened)", headers=["target_attribute", "potential_source_location", "confidence_score", "source_description"], interactive=False, wrap=True)
                 sources_output_json = gr.JSON(label="Raw Source Identification JSON (Original Structure)")
            with gr.TabItem("🗺️ Agent 4: Mappings"):
                 mapping_output_df = gr.DataFrame(label="Source-to-Target Mappings ", headers=["target_attribute", "best_potential_source", "mapping_type", "transformation_logic_summary", "data_quality_checks"], interactive=False, wrap=True) # Changed label slightly
                 mapping_output_json = gr.JSON(label="Raw Mappings JSON")
    gr.Markdown( 
    ) # Updated Disclaimer


    
    def handle_submit(use_case_text, progress=gr.Progress()):
        # Clear previous outputs and set initial status
        initial_updates = { status_output: gr.update(value="Starting...", elem_classes=["running"]), summary_output: "*Processing...*", requirements_output_df: None, requirements_output_json: None, schema_output_df: None, schema_output_json: None, sources_output_df: None, sources_output_json: None, mapping_output_df: None, mapping_output_json: None }
        yield initial_updates
        # Run the workflow
        results = run_agentic_workflow(use_case_text, progress)
        # Prepare final updates based on results
        final_updates = {}
        # Status and Summary
        status_val = results.get("status", "Unknown Status")
        final_status_class = "error" if "error" in status_val.lower() or "fail" in status_val.lower() else "completed"
        if "warning" in status_val.lower() and final_status_class == "completed": final_status_class = "warning" # Add warning class if applicable
        final_updates[status_output] = gr.update(value=status_val, elem_classes=[final_status_class])
        final_updates[summary_output] = results.get("summary", "No summary generated.")
        # Agent 1 - Requirements
        final_updates[requirements_output_json] = results.get("requirements")
        req_result = results.get("requirements")
        req_df = None
        if isinstance(req_result, dict) and "error" not in req_result:
            try:
                display_req = {k: (str(v) if isinstance(v, list) else v) for k, v in req_result.items() if k not in ["llm_placeholder_generated", "placeholder_reason"]}
                req_df = pd.DataFrame([display_req])
                expected_req_cols = ["primary_goal", "target_audience", "key_entities", "required_data_attributes", "time_sensitivity", "key_metrics_or_calculations"]
                for col in expected_req_cols:
                    if col not in req_df.columns: req_df[col] = None
                req_df = req_df[expected_req_cols]
            except Exception as e: print(f"Error creating DataFrame for requirements: {e}"); req_df = None
        final_updates[requirements_output_df] = gr.update(value=req_df)
        # Agent 2 - Schema
        final_updates[schema_output_json] = results.get("schema")
        schema_result = results.get("schema")
        schema_data = None
        if isinstance(schema_result, dict) and "schema" in schema_result and isinstance(schema_result["schema"], list): schema_data = schema_result["schema"]
        if schema_data:
            try:
                schema_df = pd.DataFrame(schema_data); schema_cols_order = ["attribute_name", "data_type", "description"]
                for col in schema_cols_order:
                    if col not in schema_df.columns: schema_df[col] = False if col == "is_primary_key" else None
                schema_df["is_primary_key"] = schema_df["is_primary_key"].fillna(False).astype(bool)
                final_updates[schema_output_df] = gr.update(value=schema_df[schema_cols_order])
            except Exception as e: print(f"⚠️ Error creating DataFrame for schema: {e}"); final_updates[schema_output_df] = gr.update(value=None)
        else: final_updates[schema_output_df] = gr.update(value=None)
     # Agent 3 - Sources
        final_updates[sources_output_json] = results.get("sources")
        sources_result = results.get("sources"); source_rows = []
        if isinstance(sources_result, dict) and "potential_sources" in sources_result and isinstance(sources_result["potential_sources"], dict):
            potential_sources_dict = sources_result["potential_sources"]
            for target_attr, sources_list in potential_sources_dict.items():
                 if isinstance(sources_list, list):
                      for source_tuple in sources_list:
                           if isinstance(source_tuple, (list, tuple)) and len(source_tuple) == 3:
                                path, score, desc = source_tuple; source_rows.append({"target_attribute": target_attr, "potential_source_location": path, "confidence_score": score, "source_description": desc})
                           else: print(f"Skipping malformed source tuple for {target_attr}: {source_tuple}"); source_rows.append({"target_attribute": target_attr, "potential_source_location": "PARSE_ERROR", "confidence_score": 0, "source_description": f"Error parsing: {source_tuple}"})
                 else: print(f"Expected list for sources of {target_attr}, got {type(sources_list)}")
        if source_rows:
            try:
                sources_df = pd.DataFrame(source_rows); source_cols_order = ["target_attribute", "potential_source_location", "confidence_score"]
                for col in source_cols_order:
                    if col not in sources_df.columns: sources_df[col] = None
                final_updates[sources_output_df] = gr.update(value=sources_df[source_cols_order])
            except Exception as e: print(f"Error creating DataFrame for sources: {e}"); final_updates[sources_output_df] = gr.update(value=None)
        else: final_updates[sources_output_df] = gr.update(value=None)
       
    # Agent 4 - Mappings
        final_updates[mapping_output_json] = results.get("mappings")
        mapping_result = results.get("mappings"); mapping_data_list = None
        if isinstance(mapping_result, dict) and "mappings" in mapping_result and isinstance(mapping_result["mappings"], list): mapping_data_list = mapping_result["mappings"]
        if mapping_data_list:
            try:
                mapping_df = pd.DataFrame(mapping_data_list); mapping_cols_order = ["target_attribute", "best_potential_source", "mapping_type", "transformation_logic_summary", "data_quality_checks"]
                for col in mapping_cols_order:
                    if col not in mapping_df.columns: mapping_df[col] = None
                if 'data_quality_checks' in mapping_df.columns: mapping_df['data_quality_checks'] = mapping_df['data_quality_checks'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
                final_updates[mapping_output_df] = gr.update(value=mapping_df[mapping_cols_order])
            except Exception as e: print(f"Error creating DataFrame for mappings: {e}"); final_updates[mapping_output_df] = gr.update(value=None)
        else: final_updates[mapping_output_df] = gr.update(value=None)
        yield final_updates


    submit_button.click(
        fn=handle_submit,
        inputs=[use_case_input],
        outputs=[ 
            status_output, summary_output,
            requirements_output_df, requirements_output_json,
            schema_output_df, schema_output_json,
            sources_output_df, sources_output_json,
            mapping_output_df, mapping_output_json
            ]
    )
if __name__ == "__main__":
    print("\nLaunching Gradio Interface...")
    demo.launch(inline=False, share=True, debug=False)
