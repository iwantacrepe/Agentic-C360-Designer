import google.generativeai as genai
import gradio as gr
import json
import os
import re
import time
import pandas as pd
from fuzzywuzzy import fuzz, process # For smarter source matching
import copy # For deep copying state

print("--------------------------------------------------")
print(f"Gradio version: {gr.__version__}")
print(f"Google Generative AI version: {genai.__version__}")
print("--------------------------------------------------")

# --- Configuration & Initialization ---
# Recommended: Set GOOGLE_API_KEY environment variable.
# Fallback: Replace "YOUR_ACTUAL_API_KEY_HERE" directly (less secure).
API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_ACTUAL_API_KEY_HERE") # <<< REPLACE FALLBACK KEY

llm = None
initialization_error = None
if not API_KEY or API_KEY == "YOUR_ACTUAL_API_KEY_HERE":
    initialization_error = "API Key not found. Please set the GOOGLE_API_KEY environment variable or replace the placeholder in the script."
    print(f"❌ CONFIG ERROR: {initialization_error}")
else:
    try:
        print("Configuring Google Generative AI...")
        genai.configure(api_key=API_KEY)
        print("Initializing Gemini Model ('gemini-1.5-flash')...")
        # Using gemini-1.5-flash as it's generally available and fast.
        llm = genai.GenerativeModel('gemini-1.5-flash')
        # Perform a quick test call to verify connectivity and API key validity
        print("Testing Gemini Model connection...")
        llm.generate_content(
            "connection test",
            generation_config=genai.types.GenerationConfig(temperature=0.1, candidate_count=1)
        )
        print("✅ Gemini Model configured and tested successfully.")
    except Exception as e:
        # Catch a wider range of potential errors during init
        error_details = f"{type(e).__name__}: {e}"
        print(f"❌ CONFIG ERROR: Failed to configure/test Gemini API: {error_details}")
        # Provide more specific guidance if possible (e.g., authentication errors)
        if "permission" in str(e).lower() or "authenticate" in str(e).lower():
            initialization_error = f"Gemini API Authentication/Permission Error: {error_details}. Please check your API key and project settings."
        else:
            initialization_error = f"Gemini API Configuration Error: {error_details}."
        llm = None # Ensure llm is None if setup fails

# --- Constants ---
MIN_FUZZY_SCORE = 70 # Adjusted threshold for fuzzy matching

# --- Mock Data Catalog (More Detailed) ---
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
    if not isinstance(text, str): return ""
    text = re.sub(r'```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def parse_json_from_llm(text, agent_name):
    """Attempts to parse JSON from LLM output, handling potential errors."""
    cleaned_text = clean_llm_response(text)
    try:
        if not cleaned_text:
             raise json.JSONDecodeError("Empty string cannot be parsed as JSON", "", 0)
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        error_msg = f"⚠️ Failed to parse JSON response from {agent_name}. Error: {e}. Raw response:\n'{cleaned_text}'"
        print(error_msg)
        return {"error": f"JSON Parse Error from {agent_name}", "details": str(e), "raw_response": cleaned_text}
    except Exception as e:
        error_msg = f"⚠️ Unexpected error parsing response from {agent_name}. Error: {e}. Raw response:\n'{cleaned_text}'"
        print(error_msg)
        return {"error": f"Unexpected Parse Error from {agent_name}", "details": str(e), "raw_response": cleaned_text}

def generate_with_retry(prompt, agent_name, max_retries=2, initial_delay=3):
    """Calls the LLM with retries on failure."""
    if not llm:
        print(f"❌ LLM not initialized. Cannot run {agent_name}.")
        return {"error": f"{agent_name} cannot proceed: Gemini model not initialized.", "details": initialization_error or "Unknown init error"}

    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            print(f"🤖 [LLM Call - {agent_name}] Attempt {retries + 1}/{max_retries}...")
            # Use safety settings to reduce refusals for valid banking terms if needed
            safety_settings = {
                # Adjust based on Gemini documentation for specific categories
                # Example: genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            }
            response = llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.25, # Slightly increased for variation but still structured
                    candidate_count=1
                ),
                safety_settings=safety_settings if safety_settings else None
            )

            # Check for blocked response first (more robust)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 safety_ratings = response.prompt_feedback.safety_ratings
                 error_detail = f"LLM prompt blocked. Reason: {block_reason}. Ratings: {safety_ratings}"
                 print(f"⚠️ [LLM Call - {agent_name}] {error_detail}")
                 # Do not retry safety blocks immediately, treat as failure for this attempt
                 raise ValueError(error_detail)

            # Check for empty text content
            if not hasattr(response, 'text') or not response.text:
                error_detail = "LLM returned empty response content."
                print(f"⚠️ [LLM Call - {agent_name}] {error_detail}")
                # Retry potentially empty responses
                raise ValueError(error_detail)


            print(f"✅ [LLM Call - {agent_name}] Successful generation.")
            parsed = parse_json_from_llm(response.text, agent_name)
            # If JSON parsing failed within parse_json_from_llm, it returns an error dict
            if isinstance(parsed, dict) and "error" in parsed:
                # Raise an exception to trigger retry if JSON parsing fails
                raise json.JSONDecodeError(parsed.get("details", "JSON Parsing failed"), parsed.get("raw_response", ""), 0)

            return parsed # Return the successfully parsed JSON

        except Exception as e:
            retries += 1
            error_type = type(e).__name__
            error_message = str(e)
            print(f"⚠️ [LLM Call - {agent_name}] Failed. Type: {error_type}. Details: {error_message}.")
            if retries >= max_retries:
                 final_error_msg = f"{agent_name} LLM call failed after {max_retries} attempts"
                 print(f"❌ [LLM Call - {agent_name}] Max retries reached. Giving up.")
                 return {"error": final_error_msg, "details": error_message, "error_type": error_type}
            else:
                print(f"   Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2 # Exponential backoff

    # Fallback return if loop finishes unexpectedly
    return {"error": f"{agent_name} unexpected error in retry loop."}

# --- Core Agent Functions (with Placeholder Logic) ---

# Agent 1: Requirements Analyst
def analyze_requirements_agent(use_case_text):
    print("\n🧠 [Agent 1] Requirements Analysis - Running...")
    prompt = f"""
    Act as an expert Requirements Analyst for Retail Banking C360. Analyze the following business use case.

    Business Use Case:
    "{use_case_text}"

    Extract and structure the following information. **If any detail is ambiguous or missing, infer a common or plausible value typical for retail banking C360 use cases.** Avoid using phrases like "unknown", "error", or "requires review".
    1.  **primary_goal:** State the main business objective (e.g., "Improve Targeted Marketing", "Assess Churn Risk", "Enhance Customer Service View"). Infer if necessary.
    2.  **target_audience:** Identify likely consumers (e.g., "Marketing Team", "Risk Analytics Department", "Customer Service Representatives"). Infer if necessary.
    3.  **key_entities:** List core subjects (e.g., ["Customer", "Account", "Transaction"]). Default to common C360 entities if unclear.
    4.  **required_data_attributes:** List specific data points needed. If requirements are very vague, list common C360 attributes like "customer_id", "full_name", "email", "phone", "account_balance", "last_activity_date". Use descriptive names. Ensure this is a list of strings.
    5.  **time_sensitivity:** Note timeframes or refresh frequency. Default to "Near Real-time" or "Daily Refresh" if not specified.
    6.  **key_metrics_or_calculations:** Identify calculations needed. If none are clear, list common ones like ["customer_age", "account_tenure", "total_balance"]. Ensure this is a list of strings.
    7.  **filters_or_segmentation:** List criteria for data subsets. If none specified, list common examples like ["active_customers", "specific_segment"]. Ensure this is a list of strings.

    Output results strictly as a single JSON object with snake_case keys. Ensure all list values contain only strings. **Do not indicate uncertainty in the output JSON.**
    """
    result = generate_with_retry(prompt, "Agent 1: Requirements")

    if isinstance(result, dict) and "error" in result:
        error_detail = result.get('details', result['error'])
        print(f"⚠️ [Agent 1] LLM call failed. Generating plausible placeholder requirements. Error: {error_detail}")
        placeholder = {
            "primary_goal": "Enhance Customer Understanding (Inferred Default)",
            "target_audience": "General Analytics / Marketing (Inferred Default)",
            "key_entities": ["Customer", "Account"],
            "required_data_attributes": ["customer_id", "full_name", "account_balance", "last_activity_date"],
            "time_sensitivity": "Daily Refresh (Assumed)",
            "key_metrics_or_calculations": ["customer_age (if DOB available)"],
            "filters_or_segmentation": ["active_customers (standard filter)"],
            "llm_placeholder_generated": True,
            "placeholder_reason": f"LLM Failure: {str(error_detail)[:150]}"
        }
        return placeholder

    if isinstance(result, dict):
        print("✅ [Agent 1] Requirements Analysis - Complete.")
    else:
         print(f"❌ [Agent 1] Requirements Analysis - Failed unexpectedly after placeholder check. Result type: {type(result)}")
         # Return a structured error if result is not a dict
         return {"error": "Agent 1 returned non-dict result", "details": str(result)}
    return result

# Agent 2: Data Architect (+ Enrichment)
def design_and_enrich_schema_agent(structured_requirements):
    print("\n🏗️ [Agent 2] Data Product Schema Design & Enrichment - Running...")
    if not isinstance(structured_requirements, dict) or "required_data_attributes" not in structured_requirements:
        error_msg = "Invalid input to Agent 2: Structured requirements missing or malformed."
        print(f"❌ [Agent 2] Failed - {error_msg}")
        return { # Return consistent placeholder structure on input error
            "data_product_name": "ERROR_INVALID_INPUT_SCHEMA",
            "schema": [{"attribute_name": "error_flag", "data_type": "STRING", "description": error_msg, "is_primary_key": False, "added_by": "error"}],
            "llm_placeholder_generated": True, "placeholder_reason": "Invalid Input to Agent 2" }

    # --- Part 1: Initial Schema Design via LLM ---
    prompt_design = f"""
    Act as a meticulous Data Architect for Retail Banking C360. Use the requirements below to design a flat target data product schema.

    Requirements:
    ```json
    {json.dumps(structured_requirements, indent=2)}
    ```
    Instructions:
    1.  Suggest a descriptive `data_product_name` (snake_case). Infer a name like `c360_standard_profile` if requirements are unclear.
    2.  Create a `schema` list based *primarily* on 'required_data_attributes'.
    3.  For each attribute: define `attribute_name` (standardized snake_case), `data_type` (standard SQL - STRING, INTEGER, DECIMAL(precision,scale), DATE, TIMESTAMP, BOOLEAN), and `description`.
    4.  **If requirements for attributes are vague, infer common C360 attributes and reasonable data types/descriptions.** Avoid indicating uncertainty. Make best guess for data types (STRING for IDs, DECIMAL for money).

    Output results strictly as a JSON object with keys 'data_product_name' and 'schema'. No explanations outside JSON.
    """
    initial_schema = generate_with_retry(prompt_design, "Agent 2: Initial Design")

    # --- Placeholder for Initial Design Failure ---
    if isinstance(initial_schema, dict) and "error" in initial_schema:
        error_detail = initial_schema.get('details', initial_schema['error'])
        print(f"⚠️ [Agent 2] LLM initial design failed. Generating placeholder schema. Error: {error_detail}")
        placeholder_schema_list = [
            {"attribute_name": "customer_id", "data_type": "STRING", "description": "Customer Identifier (Placeholder)"},
            {"attribute_name": "full_name", "data_type": "STRING", "description": "Customer Full Name (Placeholder)"},
            # Add metadata placeholders directly here
            {"attribute_name": "c360_master_customer_id", "data_type": "STRING", "description": "Consolidated unique ID (Placeholder)", "is_primary_key": True, "added_by": "placeholder"},
            {"attribute_name": "data_product_load_ts", "data_type": "TIMESTAMP", "description": "Load Timestamp (Placeholder)", "is_primary_key": False, "added_by": "placeholder"},
            {"attribute_name": "source_system_tracker", "data_type": "STRING", "description": "Source System (Placeholder)", "is_primary_key": False, "added_by": "placeholder"}
        ]
        placeholder = { "data_product_name": "c360_default_view_LLM_Error", "schema": placeholder_schema_list,
                       "llm_placeholder_generated": True, "placeholder_reason": f"LLM Error during initial design: {str(error_detail)[:150]}"}
        print("✅ [Agent 2] Schema Design & Enrichment - Complete (using placeholder).")
        return placeholder

    # --- Part 2: Rule-Based Enrichment ---
    try:
        if not isinstance(initial_schema, dict): raise TypeError(f"Initial schema not a dict, got {type(initial_schema)}")
        enriched_schema = initial_schema.copy()
        schema_attributes = enriched_schema.get("schema", [])
        if not isinstance(schema_attributes, list): raise TypeError(f"Schema key not a list, got {type(schema_attributes)}")

        print("  🔧 [Agent 2] Enriching and Validating Schema...")
        standard_attrs = { # Standard fields to add/check
            "c360_master_customer_id": {"data_type": "STRING", "description": "Consolidated unique identifier", "added_by": "enrichment"},
            "data_product_load_ts": {"data_type": "TIMESTAMP", "description": "Timestamp of data load", "added_by": "enrichment"},
            "source_system_tracker": {"data_type": "STRING", "description": "Contributing source system(s)", "added_by": "enrichment"} }
        existing_names_lower = {attr.get("attribute_name", "").lower() for attr in schema_attributes if isinstance(attr, dict)}
        new_attrs_to_add = []
        for name, details in standard_attrs.items():
            if name.lower() not in existing_names_lower:
                new_attrs_to_add.append({"attribute_name": name, "is_primary_key": (name == "c360_master_customer_id"), **details})
        schema_attributes = new_attrs_to_add + schema_attributes # Prepend standard fields
        if new_attrs_to_add: print(f"    + Added/Ensured {len(new_attrs_to_add)} standard attribute(s).")

        # Suggest Primary Key
        pk_found = any(attr.get("is_primary_key") for attr in schema_attributes if isinstance(attr, dict))
        if not pk_found:
            potential_pk_candidates = ["c360_master_customer_id", "customer_id", "account_id", "transaction_id", "loan_id", "case_id", "interaction_id"]
            for pk_candidate in potential_pk_candidates:
                for attr in schema_attributes:
                    if isinstance(attr, dict) and attr.get("attribute_name", "").lower() == pk_candidate:
                        attr["is_primary_key"] = True
                        attr["description"] = (attr.get("description", "") or "") + " (Suggested Primary Key)"
                        pk_found = True; print(f"    🔑 Suggested '{attr['attribute_name']}' as Primary Key.")
                        break
                if pk_found: break
        if not pk_found: print("    ⚠️ No obvious Primary Key candidate found or suggested.")

        # Ensure all entries are dicts and have basic keys
        final_schema = []
        for attr in schema_attributes:
             if isinstance(attr, dict):
                  attr.setdefault("attribute_name", "UNKNOWN_ATTRIBUTE")
                  attr.setdefault("data_type", "STRING")
                  attr.setdefault("description", "No description provided.")
                  attr.setdefault("is_primary_key", False)
                  attr.setdefault("added_by", None)
                  final_schema.append(attr)
             else:
                  print(f"⚠️ Found non-dict entry in schema list: {attr}. Skipping.")
        enriched_schema["schema"] = final_schema

        print("✅ [Agent 2] Schema Design & Enrichment - Complete.")
        return enriched_schema

    except Exception as e:
        error_msg = f"Error during schema enrichment: {type(e).__name__}: {e}"
        print(f"❌ [Agent 2] Failed during enrichment - {error_msg}")
        if isinstance(initial_schema, dict):
            initial_schema["enrichment_error"] = error_msg; initial_schema["llm_placeholder_generated"] = True; initial_schema["placeholder_reason"] = f"Enrichment failed: {str(error_msg)[:100]}"
            return initial_schema
        else: return {"error": "Enrichment failed", "details": error_msg, "llm_placeholder_generated": True, "placeholder_reason": "Enrichment Failed"}

# Agent 3: Source Identification Specialist (with LLM Inference Fallback)
def find_potential_sources(target_attributes):
    print("\n🔍 [Agent 3] Source Identification - Running...")
    final_sources_output = {"potential_sources": {}, "error": None} # Initialize structure
    if not target_attributes:
        print("  [Agent 3] No target attributes provided. Skipping source identification.")
        return final_sources_output # Return empty sources, no error

    # 1. Prepare catalog for searching
    all_sources = {} # Map: full_path -> (description, table, column)
    try:
        for table_name, details in MOCK_CATALOG.items():
            if isinstance(details, dict) and "columns" in details and isinstance(details["columns"], dict):
                for col_name, col_desc in details["columns"].items():
                    all_sources[f"{table_name}.{col_name}"] = (col_desc or "", table_name, col_name)
            else: print(f"⚠️ [Agent 3] Skipping malformed MOCK_CATALOG entry: {table_name}")
        if not all_sources: raise ValueError("Mock catalog processing yielded no usable sources.")
    except Exception as e:
         error_msg = f"[Agent 3] Critical Error processing mock catalog: {e}"
         print(f"❌ {error_msg}"); final_sources_output["error"] = error_msg
         return final_sources_output # Cannot proceed without catalog

    # 2. Find matches / infer for each attribute
    potential_matches = {}
    source_paths = list(all_sources.keys())
    column_names_map = {col_name: path for path, (_, _, col_name) in all_sources.items()}
    column_names_only = list(column_names_map.keys())

    for attribute in target_attributes:
        if not attribute or not isinstance(attribute, str): continue
        print(f"  🔎 Searching for source for: '{attribute}'...")
        potential_matches[attribute] = [] # Initialize list for this attribute

        try:
            # Fuzzy Matching
            match_path, score_path = process.extractOne(attribute, source_paths, scorer=fuzz.WRatio) or (None, 0)
            match_col, score_col = process.extractOne(attribute, column_names_only, scorer=fuzz.WRatio) or (None, 0)
            best_match_source, best_score = (match_path, score_path) if score_path >= score_col else (column_names_map.get(match_col), score_col) if match_col else (None, 0)

            if best_match_source and best_score >= MIN_FUZZY_SCORE:
                source_desc, _, _ = all_sources.get(best_match_source, ("Catalog lookup error", "", ""))
                potential_matches[attribute].append((best_match_source, best_score, source_desc))
                print(f"    ✅ Using catalog source: {best_match_source} (Score: {best_score})")
            else:
                # LLM Inference Fallback
                print(f"    🤔 No strong catalog match (Best score: {best_score} < {MIN_FUZZY_SCORE}). Inferring via LLM...")
                prompt_infer = f"""
                Act as a Data Catalog Expert for Retail Banking C360.
                Infer the *single most plausible* source system/table and column for the attribute '{attribute}'. Consider standard systems like Core Banking, CRM, Loans, Marketing, Support.

                Provide:
                1. `inferred_source_location`: Best guess source (e.g., "crm_interactions.last_contact_date").
                2. `justification`: Brief reason for choice.

                Output *only* a single JSON object with these keys.
                """
                llm_inference_result = generate_with_retry(prompt_infer, "Agent 3: Source Inference")

                if isinstance(llm_inference_result, dict) and "error" not in llm_inference_result and "inferred_source_location" in llm_inference_result:
                    inferred_loc = llm_inference_result.get("inferred_source_location", "LLM_INFERENCE_MISSING_KEY")
                    justification = llm_inference_result.get("justification", "N/A")
                    inferred_source_path = f"[INFERRED] {inferred_loc}"
                    potential_matches[attribute].append((inferred_source_path, 10, f"LLM Suggestion: {justification}")) # Low confidence score
                    print(f"    🤖 LLM Inferred Source: {inferred_loc} (Confidence: 10)")
                else:
                    error_detail = llm_inference_result.get('details', llm_inference_result.get('error', 'Unknown'))
                    print(f"    ⚠️ LLM inference failed for '{attribute}'. Error: {error_detail}")
                    potential_matches[attribute].append(("[INFERENCE_FAILED]", 0, f"LLM Error: {str(error_detail)[:100]}"))

        except Exception as e:
            print(f"❌ Error processing attribute '{attribute}' in Agent 3: {e}")
            potential_matches[attribute] = [("[PROCESSING_ERROR]", 0, f"Error: {e}")]
            # Optionally set a global error flag if needed
            # final_sources_output["error"] = f"Error processing {attribute}: {e}"

    final_sources_output["potential_sources"] = potential_matches
    print("✅ [Agent 3] Source Identification - Complete.")
    return final_sources_output

# Agent 4: Mapping Specialist (+ Refinement)
def generate_and_refine_mappings_agent(schema_design, potential_sources):
    print("\n🗺️ [Agent 4] Mapping Generation & Refinement - Running...")
    # --- Input Validation ---
    schema_ok = isinstance(schema_design, dict) and "schema" in schema_design and isinstance(schema_design["schema"], list)
    sources_ok = isinstance(potential_sources, dict) and "potential_sources" in potential_sources and isinstance(potential_sources["potential_sources"], dict)
    placeholder_reason = None
    if not schema_ok: placeholder_reason = "Invalid Schema Input"
    if not sources_ok: placeholder_reason = f"{placeholder_reason or ''} Invalid Sources Input".strip()

    if placeholder_reason:
         print(f"❌ [Agent 4] Failed - {placeholder_reason}")
         placeholder_mappings = []
         if schema_ok: # Try to create placeholders based on schema
              for attr_entry in schema_design.get("schema", []):
                   if isinstance(attr_entry, dict) and attr_entry.get("attribute_name"):
                       map_type = "METADATA" if attr_entry.get("added_by") == "enrichment" else "TRANSFORMATION"
                       logic = "Populated by ETL (Assumed)" if map_type == "METADATA" else "Transformation needed (Input Error)"
                       placeholder_mappings.append({
                           "target_attribute": attr_entry["attribute_name"], "best_potential_source": "N/A - Input Error",
                           "mapping_type": map_type, "transformation_logic_summary": logic, "data_quality_checks": ["Verify due to input error"]})
         return {"mappings": placeholder_mappings, "llm_placeholder_generated": True, "placeholder_reason": placeholder_reason}

    # --- Part 1: Initial Mapping via LLM ---
    prompt_map = f"""
    Act as expert Data Mapping Specialist for Retail Banking C360. Create source-to-target mapping specifications.

    Target Schema:
    ```json
    {json.dumps(schema_design, indent=2)}
    ```
    Potential Sources (Target Attribute -> List of [Source Location, Score, Description]):
    ```json
    {json.dumps(potential_sources['potential_sources'], indent=2)}
    ```
    Instructions:
    For each target schema attribute:
    1. If 'added_by' is 'enrichment', use 'METADATA' type and "Populated by ETL process" logic.
    2. Otherwise, examine potential sources. Select the *single best* source candidate (consider score/relevance/inferred status). List multiple *only if essential* for transformation (e.g., names).
    3. Determine `mapping_type`: 'DIRECT', 'TRANSFORMATION', 'METADATA'. **Avoid 'MANUAL_REVIEW'.**
    4. If no good source or confidence is low ([INFERRED] or score < {MIN_FUZZY_SCORE+5}), **make a plausible assumption**: guess the source and set type to 'TRANSFORMATION' with simple logic (e.g., casting) OR assume a common transformation (e.g., calculate age, set default).
    5. Provide brief `transformation_logic_summary`. Note assumptions made (e.g., "Assumed direct map", "Standard age calc", "Default value 'Unknown'").
    6. Suggest optional basic `data_quality_checks` (list of 1-2 strings).

    Output *only* a JSON object with key "mappings" holding a list of mapping objects. Each object needs keys: "target_attribute", "best_potential_source", "mapping_type", "transformation_logic_summary", "data_quality_checks" (use [] if none). **Provide a mapping for every target attribute.**
    """
    initial_mappings = generate_with_retry(prompt_map, "Agent 4: Initial Mapping")

    # --- Placeholder for Initial Mapping Failure ---
    if isinstance(initial_mappings, dict) and "error" in initial_mappings:
        error_detail = initial_mappings.get('details', initial_mappings['error'])
        print(f"⚠️ [Agent 4] LLM mapping failed. Generating placeholder mappings. Error: {error_detail}")
        placeholder_mappings = []
        schema_list = schema_design.get("schema", [])
        sources_dict = potential_sources.get("potential_sources", {})
        if isinstance(schema_list, list):
            for attr in schema_list:
                if isinstance(attr, dict) and attr.get("attribute_name"):
                    target_name = attr["attribute_name"]
                    best_source_guess = "N/A - LLM Mapping Error"
                    src_cands = sources_dict.get(target_name, [])
                    if src_cands and isinstance(src_cands[0], (list, tuple)) and len(src_cands[0]) > 0:
                        best_source_guess = f"{src_cands[0][0]} (Score: {src_cands[0][1]})"
                        if "NOT_FOUND" in src_cands[0][0] or "FAILED" in src_cands[0][0] or "ERROR" in src_cands[0][0]:
                            best_source_guess = f"{src_cands[0][0]} (LLM Mapping Failed)"
                    map_type = "METADATA" if attr.get("added_by") == "enrichment" else "TRANSFORMATION"
                    logic = "Populated by ETL (Placeholder)" if map_type == "METADATA" else f"Transformation needed (LLM Failed: {str(error_detail)[:50]})"
                    dq = ["Verify logic"] if map_type != "METADATA" else []
                    placeholder_mappings.append({
                        "target_attribute": target_name, "best_potential_source": best_source_guess, "mapping_type": map_type,
                        "transformation_logic_summary": logic, "data_quality_checks": dq })
        placeholder = {"mappings": placeholder_mappings, "llm_placeholder_generated": True, "placeholder_reason": f"LLM Error during mapping: {str(error_detail)[:150]}"}
        print("✅ [Agent 4] Mapping Generation & Refinement - Complete (using placeholder).")
        return placeholder

    # --- Part 2: Refinement (Completeness Check) ---
    try:
        if not isinstance(initial_mappings, dict): raise TypeError("Initial mapping result not a dict")
        refined_mappings = initial_mappings.copy()
        if "mappings" not in refined_mappings or not isinstance(refined_mappings["mappings"], list): raise ValueError("Mappings list missing/invalid in LLM output")

        print("  🔧 [Agent 4] Refining Mappings (Ensuring completeness)...")
        processed_targets = set()
        valid_mappings = []
        num_assumed = 0

        for i, mapping in enumerate(refined_mappings["mappings"]):
            if not isinstance(mapping, dict): continue
            target_attr = mapping.get("target_attribute")
            if not target_attr: continue
            processed_targets.add(target_attr)

            # Ensure essential keys & defaults
            mapping.setdefault("best_potential_source", "N/A - Missing from LLM")
            map_type = mapping.setdefault("mapping_type", "TRANSFORMATION")
            logic = mapping.setdefault("transformation_logic_summary", "Logic missing - Review needed")
            dq = mapping.setdefault("data_quality_checks", [])
            if not isinstance(dq, list): mapping["data_quality_checks"] = []

            # Replace any lingering MANUAL_REVIEW
            if map_type == "MANUAL_REVIEW":
                mapping["mapping_type"] = "TRANSFORMATION"; mapping["transformation_logic_summary"] += " (Assumed standard transformation)"; num_assumed += 1
            # Flag potential assumptions for review
            if "assume" in logic.lower() or "default" in logic.lower() or "guess" in logic.lower(): num_assumed += 1
            if "N/A" in str(mapping["best_potential_source"]) or "[INFERRED]" in str(mapping["best_potential_source"]): num_assumed += 1

            valid_mappings.append(mapping)

        refined_mappings["mappings"] = valid_mappings

        # Add placeholders for missing targets
        missing_targets = []
        schema_list = schema_design.get("schema", [])
        if isinstance(schema_list, list):
            for schema_attr in schema_list:
                if isinstance(schema_attr, dict):
                    target_name = schema_attr.get("attribute_name")
                    if target_name and target_name not in processed_targets:
                        missing_targets.append(target_name)
                        map_type = "METADATA" if schema_attr.get("added_by") == "enrichment" else "TRANSFORMATION"
                        logic = "Populated by ETL (Assumed)" if map_type == "METADATA" else "Transformation assumed (Missing from LLM)"
                        best_source = "ETL Process" if map_type == "METADATA" else "N/A - Missing"
                        refined_mappings["mappings"].append({
                            "target_attribute": target_name, "best_potential_source": best_source,
                            "mapping_type": map_type, "transformation_logic_summary": logic,
                            "data_quality_checks": ["Verify mapping"] })
                        num_assumed +=1 # Count these as assumptions

        if missing_targets: print(f"    ⚠️ Added placeholder mappings for {len(missing_targets)} attributes missing from LLM output: {missing_targets}")
        if num_assumed > 0: print(f"    🚩 Note: Approx {num_assumed} mapping(s) involved assumptions or lacked clear sources.")

        print("✅ [Agent 4] Mapping Generation & Refinement - Complete.")
        return refined_mappings

    except Exception as e:
         error_msg = f"Error during mapping refinement: {type(e).__name__}: {e}"
         print(f"❌ [Agent 4] Failed during refinement - {error_msg}")
         if isinstance(initial_mappings, dict):
             initial_mappings["refinement_error"] = error_msg; initial_mappings["llm_placeholder_generated"] = True; initial_mappings["placeholder_reason"] = f"Refinement failed: {str(error_msg)[:100]}"
             return initial_mappings
         else: return {"error": "Mapping refinement failed", "details": error_msg, "llm_placeholder_generated": True, "placeholder_reason": "Refinement Failed"}


# --- Feedback Processing Agent/Function ---
def process_feedback_agent(current_state_summary, feedback_text):
    print("\n🧐 [Feedback Agent] Analyzing Feedback - Running...")
    if not feedback_text:
        print("  [Feedback Agent] No feedback text provided.")
        return {"action": "NONE", "reason": "No feedback provided.", "target_details": None}

    # State summary helps LLM understand context (optional but good)
    prompt = f"""
    Act as an Orchestrator's Assistant analyzing user feedback on a generated Customer 360 data product design.

    Current State Summary (for context only):
    ```json
    {json.dumps(current_state_summary, indent=2)}
    ```

    User Feedback:
    "{feedback_text}"

    Analyze the feedback and determine the **primary intent** and the **target**. Decide the **next action**.

    Possible Actions:
    - `REVISE_REQUIREMENTS`: Feedback relates to fundamental goals or overall data points.
    - `REVISE_SCHEMA`: Feedback targets schema structure, attributes, or data types.
    - `REVISE_MAPPINGS`: Feedback targets specific source-to-target rules or logic. (Assume source corrections often lead here).
    - `ACCEPT`: Feedback indicates approval or is minor/informational.
    - `CLARIFY`: Feedback is unclear, asks a question, or cannot be automatically processed.
    - `NONE`: Feedback is irrelevant.

    Output Instructions:
    Provide *only* a JSON object with keys:
    - `action`: Chosen action (e.g., "REVISE_SCHEMA").
    - `target_details`: Brief explanation of *what* needs revision (e.g., "Add 'account_tenure' attribute", "Correct mapping for 'customer_age'").
    - `reason`: Short justification for the chosen action.
    """
    result = generate_with_retry(prompt, "Feedback Agent")

    if isinstance(result, dict) and "error" not in result and "action" in result:
        print(f"✅ [Feedback Agent] Analysis Complete. Action: {result.get('action')}, Target: {result.get('target_details')}")
        result["original_feedback"] = feedback_text # Add original feedback for context
        return result
    else:
        error_detail = result.get('details', result.get('error', 'Unknown feedback analysis error'))
        print(f"❌ [Feedback Agent] Failed. Error: {error_detail}")
        return { "action": "CLARIFY", "target_details": "Could not automatically parse feedback.",
                 "reason": f"Feedback analysis failed: {error_detail}", "original_feedback": feedback_text,
                 "error": result.get("error", "Feedback analysis failed") }


# --- Orchestrator Logic ---

def run_initial_agentic_workflow(use_case_text, progress=gr.Progress()):
    """ Runs A1->A2->A3->A4 sequence for the first time. Returns workflow state dict. """
    print("\n🚀 --- Starting Initial Agentic Workflow --- 🚀")
    start_time = time.time()
    workflow_state = { "use_case": use_case_text, "status": "Starting...", "requirements": None, "schema": None, "sources": None, "mappings": None, "summary": None, "warnings": [], "feedback_history": [], "iteration": 1, "final_message": ""}
    unique_warnings = set()
    def add_warning(message):
        if message and message not in unique_warnings: workflow_state["warnings"].append(message); unique_warnings.add(message)
    def update_status(stage_name, stage_index, total_stages, error=None):
        progress_fraction = stage_index / total_stages; status_msg = f"[Iter {workflow_state['iteration']}] Running: {stage_name} ({stage_index}/{total_stages})"
        if error: status_msg = f"[Iter {workflow_state['iteration']}] Error in: {stage_name}"; workflow_state["status"] = status_msg; add_warning(f"Failed at {stage_name}: {error}"); print(f" Workflow failed at stage: {stage_name}"); progress(progress_fraction, desc=f"{status_msg}")
        else: workflow_state["status"] = status_msg; print(f"Stage Complete: {stage_name}"); progress(progress_fraction, desc=workflow_state["status"])

    total_stages = 4; current_stage = 0
    # Pre-checks
    if not use_case_text: workflow_state["final_message"] = "Please enter use case."; workflow_state["status"] = "Failed (Input Missing)"; progress(0, desc="Input Missing"); return workflow_state
    if not llm: workflow_state["final_message"] = f"LLM Error: {initialization_error}"; workflow_state["status"] = "Failed (LLM Init Error)"; progress(0, desc="LLM Error"); return workflow_state

    # Agent 1
    current_stage = 1; update_status("Requirements Analysis", current_stage, total_stages)
    req_result = analyze_requirements_agent(use_case_text); workflow_state["requirements"] = req_result
    if isinstance(req_result, dict) and req_result.get("llm_placeholder_generated"): add_warning(f"A1 used defaults: {req_result.get('placeholder_reason', '')[:50]}...")
    if isinstance(req_result, dict) and "error" in req_result: update_status("Requirements Analysis", current_stage, total_stages, error=req_result.get("details", req_result["error"])); workflow_state["final_message"] = "Stopped: Error in Requirements."; return workflow_state

    # Agent 2
    current_stage = 2; update_status("Schema Design & Enrichment", current_stage, total_stages)
    schema_result = design_and_enrich_schema_agent(req_result); workflow_state["schema"] = schema_result
    if isinstance(schema_result, dict) and schema_result.get("llm_placeholder_generated"): add_warning(f"A2 used defaults: {schema_result.get('placeholder_reason', '')[:50]}...")
    schema_error = schema_result.get("error") or schema_result.get("enrichment_error")
    if schema_error: update_status("Schema Design & Enrichment", current_stage, total_stages, error=schema_error); workflow_state["final_message"] = "Stopped: Error in Schema Design."; return workflow_state

    # Agent 3
    current_stage = 3; update_status("Source Identification", current_stage, total_stages)
    target_attributes = []; agent_3_error_msg = None; sources_result = {"potential_sources": {}, "error": None}
    try:
        if isinstance(schema_result, dict) and "schema" in schema_result and isinstance(schema_result["schema"], list):
            target_attributes = [a.get("attribute_name") for a in schema_result["schema"] if isinstance(a,dict) and a.get("attribute_name") and not a.get("added_by")=='enrichment'] # Exclude enrichment fields
            if not target_attributes: add_warning("A3: No non-enrichment attributes found.")
        else: raise ValueError("Schema invalid for source ID.")
        sources_result = find_potential_sources(target_attributes); workflow_state["sources"] = sources_result
        if isinstance(sources_result, dict) and sources_result.get("error"): raise ValueError(f"A3 failed: {sources_result['error']}")
        if isinstance(sources_result, dict) and not sources_result.get("potential_sources") and target_attributes: add_warning("A3: No sources found.")
    except Exception as e:
        agent_3_error_msg = f"Crit Err during Source ID: {type(e).__name__}: {e}"; print(f" {agent_3_error_msg}"); update_status("Source Identification", current_stage, total_stages, error=agent_3_error_msg)
        workflow_state["final_message"] = "Stopped: " + agent_3_error_msg; workflow_state["sources"] = {"error": agent_3_error_msg}; return workflow_state

    # Agent 4
    current_stage = 4; update_status("Mapping Generation & Refinement", current_stage, total_stages)
    mappings_result = generate_and_refine_mappings_agent(schema_result, sources_result); workflow_state["mappings"] = mappings_result
    if isinstance(mappings_result, dict) and mappings_result.get("llm_placeholder_generated"): add_warning(f"A4 used defaults: {mappings_result.get('placeholder_reason', '')[:50]}...")
    mapping_error = mappings_result.get("error") or mappings_result.get("refinement_error")
    if mapping_error: update_status("Mapping Generation & Refinement", current_stage, total_stages, error=mapping_error); add_warning(f"A4 issues: {mapping_error}"); workflow_state["final_message"] = "Completed with errors in Mapping."
    manual_review_count = 0
    if isinstance(mappings_result, dict) and "mappings" in mappings_result and isinstance(mappings_result["mappings"], list):
        for item in mappings_result["mappings"]:
             if isinstance(item, dict):
                  logic = item.get("transformation_logic_summary", "").lower(); map_type = item.get("mapping_type")
                  if map_type == "MANUAL_REVIEW" or "review" in logic or "assume" in logic or "missing" in logic or "N/A" in str(item.get("best_potential_source")): manual_review_count += 1
        if manual_review_count > 0: add_warning(f"A4: {manual_review_count} mapping(s) may need review.")

    # Wrap-up
    end_time = time.time(); duration = end_time - start_time
    final_status = "Completed Initial Run" + (" with Warnings/Inferred Data" if workflow_state["warnings"] else "")
    workflow_state["status"] = final_status; progress(1.0, desc=final_status)
    workflow_state["final_message"] = f"✅ Initial workflow {final_status} in {duration:.2f}s. Review & provide feedback."
    print(f"🏁 --- Initial Workflow {final_status} ({duration:.2f}s) --- 🏁")

    # Generate Summary
    summary_lines = []; req_result = workflow_state.get("requirements"); schema_result = workflow_state.get("schema")
    if isinstance(req_result, dict) and not req_result.get("error") and not req_result.get("llm_placeholder_generated"): summary_lines.append(f"**Goal:** {req_result.get('primary_goal', 'N/A')}")
    elif isinstance(req_result, dict) and req_result.get("llm_placeholder_generated"): summary_lines.append("**Reqs:** Using defaults.")
    else: summary_lines.append("**Reqs:** Error.")
    if isinstance(schema_result, dict) and not schema_result.get("error") and not schema_result.get("llm_placeholder_generated"):
        summary_lines.append(f"**Product:** `{schema_result.get('data_product_name', 'N/A')}`"); num_attrs = len([a for a in schema_result.get("schema", []) if isinstance(a, dict)])
        summary_lines.append(f"**Attrs:** {num_attrs}")
    elif isinstance(schema_result, dict) and schema_result.get("llm_placeholder_generated"): summary_lines.append(f"**Schema:** Using defaults."); num_attrs = len([a for a in schema_result.get("schema", []) if isinstance(a, dict)]); summary_lines.append(f"**Attrs:** {num_attrs} (Default)")
    else: summary_lines.append("**Schema:** Error.")
    if workflow_state["warnings"]:
         summary_lines.append(f"\n**⚠️ Warnings ({len(output['warnings'])}):**"); max_warn = 5; unique_warns_list = list(unique_warnings)
         for i, warn in enumerate(unique_warns_list):
              if i < max_warn: summary_lines.append(f"- {warn[:100]}{'...' if len(warn)>100 else ''}") # Truncate long warnings
              elif i == max_warn: summary_lines.append(f"- ... and {len(unique_warns_list) - max_warn} more."); break
    else: summary_lines.append("\n**✅ No major warnings.**")
    summary_lines.append("\n**Next:** Review outputs. Use feedback box for revisions.")
    workflow_state["summary"] = "\n".join(summary_lines)

    return workflow_state


def refine_workflow_with_feedback(current_state, feedback_text, progress=gr.Progress()):
    """ Processes feedback, potentially re-runs agents, updates state. Returns updated state dict. """
    print("\n🔄 --- Refining Workflow Based on Feedback --- 🔄")
    start_time = time.time()
    state = copy.deepcopy(current_state) # Work on a copy
    state["iteration"] += 1; state["final_message"] = "" # Reset message
    status_prefix = f"[Iter {state['iteration']}]"
    state["status"] = f"{status_prefix} Processing Feedback..."
    progress(0, desc=state["status"])
    unique_warnings = set(state.get("warnings", [])) # Load existing warnings
    def add_warning(message):
        if message and message not in unique_warnings: state["warnings"].append(message); unique_warnings.add(message)
    def update_status(stage_name, current_stage, total_stages, error=None): # Stage counts relative to refinement phase
        progress_fraction = current_stage / total_stages; status_msg = f"{status_prefix} {stage_name} ({current_stage}/{total_stages})"
        if error: status_msg = f"{status_prefix} Error in: {stage_name.split(' ')[1]}"; state["status"] = status_msg; add_warning(f"Failed at {stage_name}: {error}"); print(f" Refinement failed at stage: {stage_name}"); progress(progress_fraction, desc=f"{status_msg}")
        else: state["status"] = status_msg; print(f"Refinement Stage Complete: {stage_name}"); progress(progress_fraction, desc=state["status"])

    if not feedback_text:
        add_warning("Attempted refinement without feedback text.")
        state["status"] = f"{status_prefix} Refinement Skipped (No Feedback)"; progress(1.0, desc=state["status"]); return state

    state["feedback_history"].append({"iteration": state["iteration"], "feedback": feedback_text})

    # --- Feedback Agent ---
    update_status("Analyzing Feedback", 0, 5) # Stage 0/N (N depends on reruns)
    state_summary = {k: (v if not isinstance(v,(dict,list)) else f"{type(v).__name__} present") for k,v in state.items() if k not in ['feedback_history']} # Simple summary
    feedback_analysis = process_feedback_agent(state_summary, feedback_text)
    state["last_feedback_analysis"] = feedback_analysis
    action = feedback_analysis.get("action", "CLARIFY"); target_details = feedback_analysis.get("target_details", "N/A")
    add_warning(f"Feedback Action: {action} - Target: {target_details[:60]}...")

    # --- Decide Re-runs ---
    rerun_flags = {"A1": False, "A2": False, "A3": False, "A4": False}
    if action == "REVISE_REQUIREMENTS": rerun_flags = {"A1": True, "A2": True, "A3": True, "A4": True}; state.update({"requirements": None, "schema": None, "sources": None, "mappings": None})
    elif action == "REVISE_SCHEMA": rerun_flags = {"A1": False, "A2": True, "A3": True, "A4": True}; state.update({"schema": None, "sources": None, "mappings": None})
    elif action == "REVISE_MAPPINGS": rerun_flags = {"A1": False, "A2": False, "A3": False, "A4": True}; state.update({"mappings": None})
    elif action in ["ACCEPT", "CLARIFY", "NONE"]:
        state["final_message"] = f"Feedback processed (Action: {action}). No agent re-run needed."; state["status"] = f"{status_prefix} Refinement Complete (Action: {action})"; progress(1.0, desc=state["status"])
        # Regenerate summary based on current (likely unchanged) state
        # ... (Summary generation logic - duplicated from initial run, reading from 'state') ...
        summary_lines = []; req_result = state.get("requirements"); schema_result = state.get("schema") #... rest of summary generation ...
        if state["warnings"]: summary_lines.append(f"\n** Warnings ({len(state['warnings'])}):**"); # ... rest of summary generation ...
        else: summary_lines.append("\n No major warnings."); summary_lines.append("\nNext: Provide feedback if needed."); state["summary"] = "\n".join(summary_lines)
        return state

    # --- Re-run Agents ---
    total_rerun_stages = sum(rerun_flags.values()); current_rerun_stage = 0
    progress(0.1, desc=f"{status_prefix} Starting Agent Re-runs...") # Initial progress after feedback analysis

    if rerun_flags["A1"]:
        current_rerun_stage += 1; update_status("Re-running Requirements", current_rerun_stage, total_rerun_stages)
        req_result = analyze_requirements_agent(state["use_case"]); state["requirements"] = req_result
        if isinstance(req_result, dict) and req_result.get("llm_placeholder_generated"): add_warning(f"A1 (Re-run) used defaults: {req_result.get('placeholder_reason','N/A')[:50]}...")
        if isinstance(req_result, dict) and "error" in req_result: add_warning(f"Refinement Err A1: {req_result.get('error')}"); state["status"] = f"{status_prefix} Refinement Error"; progress(1.0, desc=state["status"]); return state # Stop

    if rerun_flags["A2"]:
        current_rerun_stage += 1; update_status("Re-running Schema Design", current_rerun_stage, total_rerun_stages)
        if not state.get("requirements") or isinstance(state.get("requirements"),dict) and "error" in state.get("requirements",{}): add_warning("Cannot re-run A2: Reqs invalid."); state["status"] = f"{status_prefix} Refinement Error"; progress(1.0, desc=state["status"]); return state # Stop
        schema_result = design_and_enrich_schema_agent(state["requirements"]); state["schema"] = schema_result
        if isinstance(schema_result, dict) and schema_result.get("llm_placeholder_generated"): add_warning(f"A2 (Re-run) used defaults: {schema_result.get('placeholder_reason','N/A')[:50]}...")
        schema_error = schema_result.get("error") or schema_result.get("enrichment_error")
        if schema_error: add_warning(f"Refinement Err A2: {schema_error}"); state["status"] = f"{status_prefix} Refinement Error"; progress(1.0, desc=state["status"]); return state # Stop

    if rerun_flags["A3"]:
        current_rerun_stage += 1; update_status("Re-running Source ID", current_rerun_stage, total_rerun_stages)
        target_attributes = []; agent_3_error_msg = None; sources_result = {"potential_sources": {}, "error": None}
        try: # Same logic as initial run, using current state
            schema_state = state.get("schema")
            if isinstance(schema_state, dict) and "schema" in schema_state and isinstance(schema_state["schema"], list):
                target_attributes = [a.get("attribute_name") for a in schema_state["schema"] if isinstance(a,dict) and a.get("attribute_name") and not a.get("added_by")=='enrichment']
                if not target_attributes: add_warning("A3 (Re-run): No non-enrichment attrs.")
            else: raise ValueError("Current schema state invalid.")
            sources_result = find_potential_sources(target_attributes); state["sources"] = sources_result
            if isinstance(sources_result, dict) and sources_result.get("error"): raise ValueError(f"A3 failed: {sources_result['error']}")
            if isinstance(sources_result, dict) and not sources_result.get("potential_sources") and target_attributes: add_warning("A3 (Re-run): No sources found.")
        except Exception as e: # Critical error stops refinement
             agent_3_error_msg = f"Crit Err during Source ID Re-run: {type(e).__name__}: {e}"; print(f" {agent_3_error_msg}"); add_warning(f"Refinement Err A3: {agent_3_error_msg}")
             state["final_message"] = "Stopped: " + agent_3_error_msg; state["sources"] = {"error": agent_3_error_msg}; state["status"] = f"{status_prefix} Refinement Error"; progress(1.0, desc=state["status"]); return state

    if rerun_flags["A4"]:
        current_rerun_stage += 1; update_status("Re-running Mapping Gen", current_rerun_stage, total_rerun_stages)
        schema_state = state.get("schema"); sources_state = state.get("sources")
        if not schema_state or not sources_state or (isinstance(sources_state,dict) and "error" in sources_state): add_warning("Cannot re-run A4: Schema/Sources invalid."); state["status"] = f"{status_prefix} Refinement Error"; progress(1.0, desc=state["status"]); return state # Stop
        mappings_result = generate_and_refine_mappings_agent(schema_state, sources_state); state["mappings"] = mappings_result
        if isinstance(mappings_result, dict) and mappings_result.get("llm_placeholder_generated"): add_warning(f"A4 (Re-run) used defaults: {mappings_result.get('placeholder_reason','N/A')[:50]}...")
        mapping_error = mappings_result.get("error") or mappings_result.get("refinement_error")
        if mapping_error: add_warning(f"A4 (Re-run) issues: {mapping_error}"); state["final_message"] = "Refinement completed with errors in Mapping." # Don't stop
        # Re-check manual flags
        manual_review_count = 0 # ... (count logic) ...
        if isinstance(mappings_result, dict) and "mappings" in mappings_result and isinstance(mappings_result["mappings"], list):
            for item in mappings_result["mappings"]:
                if isinstance(item, dict):
                    logic = item.get("transformation_logic_summary", "").lower(); map_type = item.get("mapping_type")
                    if map_type == "MANUAL_REVIEW" or "review" in logic or "assume" in logic or "missing" in logic or "N/A" in str(item.get("best_potential_source")): manual_review_count += 1
            if manual_review_count > 0: add_warning(f"A4 (Re-run): {manual_review_count} mapping(s) may need review.")

    # --- Wrap-up Refinement ---
    end_time = time.time(); duration = end_time - start_time
    final_status = f"Refinement Iteration {state['iteration']} Complete" + (" with Warnings" if state["warnings"] else "")
    state["status"] = final_status; progress(1.0, desc=final_status)
    if not state.get("final_message"): state["final_message"] = f"✅ Refinement iteration {state['iteration']} completed in {duration:.2f}s. Review updated results."
    print(f"🏁 --- Refinement Iteration {state['iteration']} {final_status} ({duration:.2f}s) --- 🏁")

    # Regenerate summary based on the *new* state
    summary_lines = []; req_result = state.get("requirements"); schema_result = state.get("schema")
    if isinstance(req_result, dict) and "error" not in req_result and not req_result.get("llm_placeholder_generated"): summary_lines.append(f"**Goal:** {req_result.get('primary_goal', 'N/A')}")
    elif isinstance(req_result, dict) and req_result.get("llm_placeholder_generated"): summary_lines.append("**Reqs:** Using defaults.")
    else: summary_lines.append("**Reqs:** Error.")
    if isinstance(schema_result, dict) and "error" not in schema_result and not schema_result.get("llm_placeholder_generated"):
        summary_lines.append(f"**Product:** `{schema_result.get('data_product_name', 'N/A')}`"); num_attrs = len([a for a in schema_result.get("schema", []) if isinstance(a, dict)])
        summary_lines.append(f"**Attrs:** {num_attrs}")
    elif isinstance(schema_result, dict) and schema_result.get("llm_placeholder_generated"): summary_lines.append(f"**Schema:** Using defaults."); num_attrs = len([a for a in schema_result.get("schema", []) if isinstance(a, dict)]); summary_lines.append(f"**Attrs:** {num_attrs} (Default)")
    else: summary_lines.append("**Schema:** Error.")
    if state["warnings"]:
         summary_lines.append(f"\n**⚠️ Warnings ({len(state['warnings'])}):**"); max_warn = 7; displayed_warnings_iter = set(); warnings_to_show_iter = []
         all_warnings_ordered = [w for w in state["warnings"]] # Show all warnings
         for i, warn in enumerate(all_warnings_ordered):
              if i < max_warn: summary_lines.append(f"- {warn[:100]}{'...' if len(warn)>100 else ''}")
              elif i == max_warn: summary_lines.append(f"- ... and {len(all_warnings_ordered) - max_warn} more unique warnings."); break
    else: summary_lines.append("\n**✅ No warnings detected.**")
    summary_lines.append("\n**Next:** Review outputs. Provide further feedback if needed.")
    state["summary"] = "\n".join(summary_lines)

    return state

# --- Gradio UI ---
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; max-width: 95%; margin: auto;}
.gr-button { border-radius: 8px; padding: 10px 20px; font-weight: bold; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.1s ease; }
.gr-button:hover { transform: translateY(-1px); box-shadow: 3px 3px 7px rgba(0,0,0,0.15); }
#submit_button { background-color: #28a745; color: white; border: 1px solid #218838; } /* Green */
#submit_button:hover { background-color: #218838; border-color: #1e7e34;}
#feedback_button { background-color: #ffc107; color: #212529; border: 1px solid #e0a800; } /* Yellow */
#feedback_button:hover { background-color: #e0a800; border-color: #d39e00; }
.warning { background-color: #fffbe6; border: 1px solid #ffe58f; padding: 10px; border-radius: 4px; color: #d46b08; margin-bottom: 10px;}
.error { background-color: #fff1f0; border: 1px solid #ffa39e; padding: 10px; border-radius: 4px; color: #cf1322; margin-bottom: 10px;}
.summary { background-color: #f0f5ff; border: 1px solid #adc6ff; padding: 15px; border-radius: 4px; color: #1d39c4; margin-top: 15px; line-height: 1.6;}
#status_display { font-weight: bold; margin-top: 10px; padding: 8px; border-radius: 4px; text-align: center; transition: background-color 0.3s ease, color 0.3s ease;}
#status_display.idle { background-color: #f0f0f0; color: #595959; }
#status_display.running { background-color: #e6f7ff; color: #096dd9; }
#status_display.completed { background-color: #f6ffed; color: #389e0d; } /* Green for success */
#status_display.warning { background-color: #fffbe6; color: #d46b08; } /* Yellow for warnings */
#status_display.error { background-color: #fff1f0; color: #cf1322; } /* Red for errors */
.gr-tabitem { padding: 10px; background-color: #ffffff; border: 1px solid #e8e8e8; border-radius: 4px; margin-top:-1px;} /* Style tabs */
.gr-dataframe { overflow-x: auto; } /* Allow horizontal scroll for wide tables */
.gr-dataframe table { width: 100%; font-size: 0.9em; table-layout: auto;} /* Adjust table layout */
.gr-dataframe th { background-color: #f0f5ff; text-align: left; padding: 6px;}
.gr-dataframe td { text-align: left; padding: 6px; vertical-align: top;}
#feedback_box { border: 1px solid #ccc; padding: 15px; border-radius: 5px; margin-top: 20px; background-color: #fafafa; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), css=css) as demo:
    workflow_state = gr.State({}) # Holds the workflow state dictionary

    gr.Markdown("# 🏦 Dynamic Agentic C360 Data Product Designer w/ Feedback Loop")
    gr.Markdown("*Enter use case -> Generate Initial Design -> Review -> Provide Feedback -> Refine Iteratively*")

    with gr.Row():
        use_case_input = gr.Textbox( lines=6, label="Enter Business Use Case Description Here:", placeholder="e.g., 'Profile high-value customers for a wealth management offer. Need demographics, total assets under management (sum of balances), investment product holdings, and risk tolerance score.'", scale=3)
        with gr.Column(scale=1):
            submit_button = gr.Button("🚀 Generate Initial Design", elem_id="submit_button")
            status_output = gr.Textbox(label="Workflow Status", interactive=False, elem_id="status_display", value="Idle", elem_classes=["idle"])
            summary_output = gr.Markdown(elem_id="summary", value="*Summary will appear here after processing...*")

    with gr.Accordion("📊 Detailed Agent Outputs (Review Here)", open=False):
        with gr.Tabs():
            with gr.TabItem("🧠 Agent 1: Requirements"):
                 requirements_output_df = gr.DataFrame(label="Requirements Summary Table", interactive=False, wrap=True)
                 requirements_output_json = gr.JSON(label="Raw Requirements JSON")
            with gr.TabItem("🏗️ Agent 2: Schema Design"):
                 schema_output_df = gr.DataFrame(label="Recommended & Enriched Schema", headers=["attribute_name", "data_type", "description", "is_primary_key", "added_by"], interactive=False, wrap=True)
                 schema_output_json = gr.JSON(label="Raw Schema JSON")
            with gr.TabItem("🔍 Agent 3: Source Identification"):
                 sources_output_df = gr.DataFrame( label="Potential Source Candidates", headers=["target_attribute", "potential_source_location", "confidence_score", "source_description"], interactive=False, wrap=True)
                 sources_output_json = gr.JSON(label="Raw Source JSON")
            with gr.TabItem("🗺️ Agent 4: Mappings"):
                 mapping_output_df = gr.DataFrame(label="Source-to-Target Mappings", headers=["target_attribute", "best_potential_source", "mapping_type", "transformation_logic_summary", "data_quality_checks"], interactive=False, wrap=True)
                 mapping_output_json = gr.JSON(label="Raw Mappings JSON")

    with gr.Box(elem_id="feedback_box"):
         gr.Markdown("**🔁 Provide Feedback for Refinement:** (Submit after initial design is generated)")
         feedback_input = gr.Textbox(lines=3, label="Enter feedback here:", placeholder="e.g., 'Add account opening date to the schema.', 'Source for segment should be crm.segment', 'Map total balance by summing across all accounts.'")
         feedback_button = gr.Button("SUBMIT FEEDBACK & REFINE", elem_id="feedback_button")

    gr.Markdown( "***Disclaimer:** Prototype using AI. Outputs require manual review. Assumes access to configured LLM API.*")

    # --- Gradio Update Function (Common for Initial Run & Feedback) ---
    def update_ui_from_state(state_dict):
        """Takes workflow state dict, returns updates for all UI components."""
        print("Updating UI from state...")
        updates = { # Initialize update dict with all components
            status_output: None, summary_output: None,
            requirements_output_df: None, requirements_output_json: None,
            schema_output_df: None, schema_output_json: None,
            sources_output_df: None, sources_output_json: None,
            mapping_output_df: None, mapping_output_json: None,
            feedback_input: None # Include feedback input to clear it
        }
        if not isinstance(state_dict, dict): # Handle invalid state gracefully
            print("Error: Invalid state received by update_ui_from_state.")
            updates[status_output] = gr.update(value="Internal State Error", elem_classes=["error"])
            updates[summary_output] = "Error displaying results due to internal state problem."
            return updates

        # Status and Summary
        status_val = state_dict.get("status", "Unknown Status")
        final_status_class = "error" if "error" in status_val.lower() or "fail" in status_val.lower() else "completed"
        if "warning" in status_val.lower() and final_status_class == "completed": final_status_class = "warning"
        if "running" in status_val.lower() or "processing" in status_val.lower(): final_status_class = "running"
        if status_val == "Idle": final_status_class = "idle"
        updates[status_output] = gr.update(value=status_val, elem_classes=[final_status_class])
        updates[summary_output] = state_dict.get("summary", "*No summary available.*")

        # Agent 1 Outputs
        req_result = state_dict.get("requirements")
        updates[requirements_output_json] = req_result # Show raw JSON regardless of errors
        req_df = None
        if isinstance(req_result, dict) and "error" not in req_result:
            try:
                display_req = {k: (str(v) if isinstance(v, list) else v) for k, v in req_result.items() if k not in ["llm_placeholder_generated", "placeholder_reason"]}
                req_df = pd.DataFrame([display_req])
                expected_req_cols = ["primary_goal", "target_audience", "key_entities", "required_data_attributes", "time_sensitivity", "key_metrics_or_calculations", "filters_or_segmentation"]
                for col in expected_req_cols:
                    if col not in req_df.columns: req_df[col] = None
                req_df = req_df[expected_req_cols]
            except Exception as e: print(f"⚠️ Error creating DataFrame for requirements: {e}")
        updates[requirements_output_df] = gr.update(value=req_df)

        # Agent 2 Outputs
        schema_result = state_dict.get("schema")
        updates[schema_output_json] = schema_result
        schema_data = None; schema_df=None
        if isinstance(schema_result, dict) and "schema" in schema_result and isinstance(schema_result["schema"], list): schema_data = schema_result["schema"]
        if schema_data:
            try:
                schema_df = pd.DataFrame(schema_data); schema_cols_order = ["attribute_name", "data_type", "description", "is_primary_key", "added_by"]
                for col in schema_cols_order:
                    if col not in schema_df.columns: schema_df[col] = False if col == "is_primary_key" else None
                schema_df["is_primary_key"] = schema_df["is_primary_key"].fillna(False).astype(bool)
                schema_df["added_by"] = schema_df["added_by"].fillna("")
                schema_df = schema_df.fillna("") # Fill remaining NaNs for display
            except Exception as e: print(f"⚠️ Error creating DataFrame for schema: {e}"); schema_df = None
        updates[schema_output_df] = gr.update(value=schema_df)

        # Agent 3 Outputs
        sources_result = state_dict.get("sources")
        updates[sources_output_json] = sources_result
        source_rows = []; sources_df = None
        if isinstance(sources_result, dict) and "potential_sources" in sources_result and isinstance(sources_result["potential_sources"], dict):
            potential_sources_dict = sources_result["potential_sources"]
            for target_attr, sources_list in potential_sources_dict.items():
                 if isinstance(sources_list, list):
                      for source_tuple in sources_list:
                           if isinstance(source_tuple, (list, tuple)) and len(source_tuple) == 3:
                                path, score, desc = source_tuple; source_rows.append({"target_attribute": target_attr, "potential_source_location": path, "confidence_score": score, "source_description": desc})
                           else: source_rows.append({"target_attribute": target_attr, "potential_source_location": "PARSE_ERROR", "confidence_score": 0, "source_description": f"Err: {source_tuple}"})
        if source_rows:
            try:
                sources_df = pd.DataFrame(source_rows); source_cols_order = ["target_attribute", "potential_source_location", "confidence_score", "source_description"]
                for col in source_cols_order:
                    if col not in sources_df.columns: sources_df[col] = None
                sources_df = sources_df.fillna("")[source_cols_order] # Fill NaNs and order cols
            except Exception as e: print(f"⚠️ Error creating DataFrame for sources: {e}"); sources_df = None
        updates[sources_output_df] = gr.update(value=sources_df)

        # Agent 4 Outputs
        mapping_result = state_dict.get("mappings")
        updates[mapping_output_json] = mapping_result
        mapping_data_list = None; mapping_df = None
        if isinstance(mapping_result, dict) and "mappings" in mapping_result and isinstance(mapping_result["mappings"], list): mapping_data_list = mapping_result["mappings"]
        if mapping_data_list:
            try:
                mapping_df = pd.DataFrame(mapping_data_list); mapping_cols_order = ["target_attribute", "best_potential_source", "mapping_type", "transformation_logic_summary", "data_quality_checks"]
                for col in mapping_cols_order:
                    if col not in mapping_df.columns: mapping_df[col] = None
                if 'data_quality_checks' in mapping_df.columns: mapping_df['data_quality_checks'] = mapping_df['data_quality_checks'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x).fillna("")
                mapping_df = mapping_df.fillna("")[mapping_cols_order] # Fill NaNs and order cols
            except Exception as e: print(f"⚠️ Error creating DataFrame for mappings: {e}"); mapping_df = None
        updates[mapping_output_df] = gr.update(value=mapping_df)

        # Clear feedback input box after processing it
        updates[feedback_input] = gr.update(value="")

        return updates

    # --- Gradio Event Handlers ---
    def handle_initial_submit(use_case_text, progress=gr.Progress(track_bandits=True)):
        yield { status_output: gr.update(value="Running Initial Workflow...", elem_classes=["running"]), summary_output: "*Processing...*", requirements_output_df: None, requirements_output_json: None, schema_output_df: None, schema_output_json: None, sources_output_df: None, sources_output_json: None, mapping_output_df: None, mapping_output_json: None, feedback_input: "" }
        initial_state = run_initial_agentic_workflow(use_case_text, progress)
        final_ui_updates = update_ui_from_state(initial_state)
        yield {**final_ui_updates, workflow_state: initial_state}

    def handle_feedback_submit(current_state_dict, feedback_text, progress=gr.Progress(track_bandits=True)):
        yield { status_output: gr.update(value="Processing Feedback and Refining...", elem_classes=["running"]), feedback_input: gr.update(value="") # Clear feedback box immediately
               # Keep other outputs as they are until refinement finishes
             }
        if not isinstance(current_state_dict, dict) or not current_state_dict: # Check if state is valid before proceeding
             print("⚠️ Feedback submitted but current state is invalid or empty. Resetting.")
             yield { status_output: gr.update(value="Error: State lost. Please start over.", elem_classes=["error"]), summary_output: "Cannot process feedback without valid state.", workflow_state: {} } # Reset state
             return

        refined_state = refine_workflow_with_feedback(current_state_dict, feedback_text, progress)
        final_ui_updates = update_ui_from_state(refined_state)
        yield {**final_ui_updates, workflow_state: refined_state}

    # --- Connect Buttons to Handlers ---
    submit_button.click(
        fn=handle_initial_submit,
        inputs=[use_case_input],
        outputs=[ status_output, summary_output, requirements_output_df, requirements_output_json, schema_output_df, schema_output_json, sources_output_df, sources_output_json, mapping_output_df, mapping_output_json, feedback_input, workflow_state ] )

    feedback_button.click(
        fn=handle_feedback_submit,
        inputs=[workflow_state, feedback_input],
        outputs=[ status_output, summary_output, requirements_output_df, requirements_output_json, schema_output_df, schema_output_json, sources_output_df, sources_output_json, mapping_output_df, mapping_output_json, feedback_input, workflow_state ] )

# --- Launch App ---
if __name__ == "__main__":
    print("--------------------------------------------------")
    if initialization_error:
         print(f"❌ APPLICATION STARTUP FAILED: {initialization_error}")
         print("   Please fix the API key configuration and restart.")
    else:
         print("Launching Gradio Interface...")
         print("Please wait for the Gradio server to start...")
         # share=True creates a public link (requires login/tunnel)
         # debug=True provides verbose Gradio logs in console
         demo.launch(inline=False, share=True, debug=False)
    print("--------------------------------------------------")
