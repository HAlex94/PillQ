import streamlit as st
import pandas as pd
import requests
import io
import urllib
import urllib.parse
from helper_functions import (
    get_openfda_searchable_fields,
    get_label_field,
    get_combined_label_field,
    get_setid_from_search,
    fallback_ndc_search,
    unify_source_string,
    convert_df
)
import json
import os
from pathlib import Path
import time
import base64

# ------------------------------------------------------------------
# Setup and Configuration
# ------------------------------------------------------------------

# Set page config
st.set_page_config(
    page_title="PillQ - Pharmaceutical Information Tool",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Settings and session state initialization
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'ollama_url' not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"
if 'deepseek_api_key' not in st.session_state:
    st.session_state.deepseek_api_key = ""
if 'huggingface_api_key' not in st.session_state:
    st.session_state.huggingface_api_key = ""
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "HuggingFace"
if 'model_name' not in st.session_state:
    st.session_state.model_name = "google/flan-t5-small"

# Load settings from file if it exists
def load_settings():
    settings_path = Path("settings.json")
    if settings_path.exists():
        try:
            with open(settings_path, "r") as f:
                settings = json.load(f)
                st.session_state.openai_api_key = settings.get("openai_api_key", "")
                st.session_state.ollama_url = settings.get("ollama_url", "http://localhost:11434")
                st.session_state.deepseek_api_key = settings.get("deepseek_api_key", "")
                st.session_state.huggingface_api_key = settings.get("huggingface_api_key", "")
                st.session_state.selected_model = settings.get("selected_model", "HuggingFace")
                st.session_state.model_name = settings.get("model_name", "google/flan-t5-small")
        except Exception as e:
            print(f"Error loading settings: {e}")

# Save settings to file
def save_settings():
    settings_path = Path("settings.json")
    try:
        settings = {
            "openai_api_key": st.session_state.openai_api_key,
            "ollama_url": st.session_state.ollama_url,
            "deepseek_api_key": st.session_state.deepseek_api_key,
            "huggingface_api_key": st.session_state.huggingface_api_key,
            "selected_model": st.session_state.selected_model,
            "model_name": st.session_state.model_name
        }
        with open(settings_path, "w") as f:
            json.dump(settings, f)
    except Exception as e:
        print(f"Error saving settings: {e}")

# Load settings at startup
load_settings()

# ------------------------------------------------------------------
# Title & Description
# ------------------------------------------------------------------
st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <h1>PillQ – Pill Queries, Simplified 💊</h1>
    </div>
    """, 
    unsafe_allow_html=True
)
st.markdown("""
Spend less time searching and more time making decisions.  
**PillQ** deciphers complex drug data in an instant—whether for formulary management, verification, or documentation.  
Pill Queries, Simplified.
""")

# ------------------------------------------------------------------
# 1) Enhanced function to fetch brand/generic name info from the drugsfda endpoint
#    so we can retrieve dosage_form, route, strength, etc.
# ------------------------------------------------------------------

def fetch_ndcs_for_name_drugsfda(name_str, limit=50):
    """
    Query the openFDA 'drugsfda' endpoint by brand_name OR generic_name.
    If no openfda.product_ndc is found, fallback to /drug/ndc endpoint
    by brand or chemical name to retrieve actual NDC codes.
    """
    base_url = "https://api.fda.gov/drug/drugsfda.json"
    
    uppercase_term = name_str.upper()
    
    # E.g. (products.brand_name:"ACETAMINOPHEN" OR products.generic_name:"ACETAMINOPHEN")
    raw_expr = f'(products.brand_name:"{uppercase_term}" OR products.generic_name:"{uppercase_term}")'
    safe_chars = '()+:"'
    encoded_expr = urllib.parse.quote(raw_expr, safe=safe_chars)
    
    params = {
        "search": encoded_expr,
        "limit": limit
    }
    
    results_list = []
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "results" in data and data["results"]:
            for item in data["results"]:
                products = item.get("products", [])
                for prod in products:
                    brand = prod.get("brand_name", "Not Available")
                    gen = prod.get("generic_name", "Not Available")
                    dform = prod.get("dosage_form", "Not Available")
                    route = prod.get("route", "Not Available")
                    
                    # Get manufacturer information
                    manufacturer = item.get("sponsor_name", "Not Available")
                    
                    # Get or create product description
                    prod_type = prod.get("product_type", "")
                    marketing_status = prod.get("marketing_status", "")
                    product_description = f"{prod_type} {marketing_status}".strip()
                    if not product_description:
                        product_description = "Not Available"
                    
                    # active_ingredients => parse name + strength
                    ai_list = prod.get("active_ingredients", [])
                    if ai_list:
                        pieces = []
                        strength_only = []
                        for ai in ai_list:
                            nm = ai.get("name", "")
                            stg = ai.get("strength", "")
                            if nm or stg:
                                pieces.append(f"{nm} {stg}".strip())
                                strength_only.append(stg.strip())
                        active_ingr = " | ".join(pieces) if pieces else "Not Available"
                        # Extract only the strength part
                        strength = " | ".join(strength_only) if strength_only else "Not Available"
                    else:
                        active_ingr = "Not Available"
                        strength = "Not Available"
                    
                    # If generic_name is "Not Available", use active_ingredients as "gen"
                    if gen == "Not Available" and ai_list:
                        chem_names = []
                        for ai in ai_list:
                            nm = ai.get("name", "")
                            if nm:
                                chem_names.append(nm)
                        if chem_names:
                            gen = " / ".join(set(chem_names))
                    
                    # Check openfda sub-object for product_ndc
                    ofda = prod.get("openfda", {})
                    ndcs = ofda.get("product_ndc", [])
                    
                    if not ndcs:
                        # Fallback approach: if brand != "Not Available", we try brand
                        # else if brand is also "Not Available", we fallback to 'gen'
                        if brand != "Not Available":
                            fallback_name = brand
                        else:
                            fallback_name = gen if gen != "Not Available" else uppercase_term
                        
                        from urllib.parse import quote
                        # Use fallback function to get NDCs
                        fallback_list = fallback_ndc_search(fallback_name)
                        
                        if not fallback_list:
                            # No fallback NDC found
                            results_list.append({
                                "NDC": "Not Available",
                                "Brand Name": brand,
                                "Generic Name": gen,
                                "Strength": strength,
                                "Route": route,
                                "Dosage Form": dform,
                                "Manufacturer": manufacturer,
                                "Product Description": product_description
                            })
                        else:
                            # For each fallback code, get package NDCs
                            for ndc_val in fallback_list:
                                # Get package-level NDCs for this product
                                package_ndcs = get_package_ndcs_for_product(ndc_val)
                                
                                if package_ndcs:
                                    # Add each package NDC as a separate row
                                    for package_ndc in package_ndcs:
                                        results_list.append({
                                            "NDC": package_ndc,
                                            "Brand Name": brand,
                                            "Generic Name": gen,
                                            "Strength": strength,
                                            "Route": route,
                                            "Dosage Form": dform,
                                            "Manufacturer": manufacturer,
                                            "Product Description": product_description
                                        })
                                else:
                                    # If no package NDCs found, use the product NDC
                                    results_list.append({
                                        "NDC": ndc_val,
                                        "Brand Name": brand,
                                        "Generic Name": gen,
                                        "Strength": strength,
                                        "Route": route,
                                        "Dosage Form": dform,
                                        "Manufacturer": manufacturer,
                                        "Product Description": product_description
                                    })
                    else:
                        # If openfda product_ndc is found, get package NDCs for each product
                        for ndc_val in ndcs:
                            # Get package-level NDCs for this product
                            package_ndcs = get_package_ndcs_for_product(ndc_val)
                            
                            if package_ndcs:
                                # Add each package NDC as a separate row
                                for package_ndc in package_ndcs:
                                    results_list.append({
                                        "NDC": package_ndc,
                                        "Brand Name": brand,
                                        "Generic Name": gen,
                                        "Strength": strength,
                                        "Route": route,
                                        "Dosage Form": dform,
                                        "Manufacturer": manufacturer,
                                        "Product Description": product_description
                                    })
                            else:
                                # If no package NDCs found, use the product NDC
                                results_list.append({
                                    "NDC": ndc_val,
                                    "Brand Name": brand,
                                    "Generic Name": gen,
                                    "Strength": strength,
                                    "Route": route,
                                    "Dosage Form": dform,
                                    "Manufacturer": manufacturer,
                                    "Product Description": product_description
                                })
    except Exception as e:
        st.error(f"Error retrieving data from openFDA drugsfda endpoint: {e}")
    
    # Sort by the Brand Name column
    if results_list:
        results_list = sorted(results_list, key=lambda x: x["Brand Name"])
    return results_list

# ------------------------------------------------------------------
# 2) Searching by NDC with your existing logic (ensures source is set properly)
# ------------------------------------------------------------------
def search_ndc(ndc_str, labels, include_source=False):
    """
    For each label in 'labels':
      - If 'active_ingredient' or 'inactive_ingredient', call get_combined_label_field -> 'openFDA + DailyMed'
      - Else call get_label_field -> might return 'openFDA' or 'DailyMed' in the second value 'src'.
    """
    result = {"NDC": ndc_str}
    for lbl in labels:
        if lbl in ["active_ingredient", "inactive_ingredient"]:
            data_list = get_combined_label_field(
                ndc_str,
                lbl,
                ["active ingredient"] if lbl == "active_ingredient" else ["inactive ingredient"]
            )
            result[lbl] = ", ".join(data_list)
            if include_source:
                # We'll assume 'openFDA + DailyMed' for these, since get_combined_label_field uses both
                result[lbl + "_source"] = "openFDA + DailyMed"
        else:
            data_vals, src = get_label_field(ndc_str, lbl, [lbl.replace("_", " ")])
            result[lbl] = ", ".join(data_vals)
            if include_source:
                unified = unify_source_string(src)
                result[lbl + "_source"] = unified
    return result

# ------------------------------------------------------------------
# 3) Searching by Name (placeholder for multiple names)
#    If the user typed multiple brand/generic names, we do a simple approach.
#    If the user typed a single name, we do 'fetch_ndcs_for_name_drugsfda'
# ------------------------------------------------------------------
def search_name_placeholder(name_str, labels, include_source=False):
    data_dict = {"Name": name_str}
    for lbl in labels:
        data_dict[lbl] = f"Placeholder for {lbl} (Name: {name_str})"
        if include_source:
            data_dict[lbl + "_source"] = "openFDA"
    return data_dict

def get_single_item_source(data_dict, labels, include_source):
    """
    We parse each 'lbl + "_source"' for 'openfda' or 'dailymed'.
    If we see both, it's 'openFDA + DailyMed'. If we see only one, that's it. Else 'None'.
    """
    if not include_source:
        return "None"
    sources_used = set()
    for lbl in labels:
        src_key = lbl + "_source"
        if src_key in data_dict:
            src = data_dict[src_key].lower()
            if "openfda" in src:
                sources_used.add("openFDA")
            if "dailymed" in src:
                sources_used.add("DailyMed")
    if len(sources_used) == 0:
        return "None"
    elif len(sources_used) == 1:
        return next(iter(sources_used))
    else:
        return "openFDA + DailyMed"

# ------------------------------------------------------------------
# Layout with three tabs: Single/Multiple Search, File Upload, Settings
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Single/Multiple Search", "File Upload", "Settings"])

with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Single/Multiple Search")
        
        # Initialize session state variables if they don't exist
        if 'multi_input' not in st.session_state:
            st.session_state.multi_input = ""
        if 'search_performed' not in st.session_state:
            st.session_state.search_performed = False
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'selected_ndcs' not in st.session_state:
            st.session_state.selected_ndcs = []
        
        # Function to update search state
        def update_search():
            st.session_state.multi_input = multi_input
            st.session_state.search_performed = True
            # Reset the selected NDCs and search results when starting a new search
            st.session_state.selected_ndcs = []
            st.session_state.search_results = []
        
        # Text input for NDC or drug name
        multi_input = st.text_area(
            "Enter values (comma-separated)",
            value=st.session_state.multi_input,
            placeholder="Example NDC: 12345-6789-0\nExample Name: Aspirin",
            height=150
        )
        
        try:
            available_fields = get_openfda_searchable_fields()
            default_fields = ["active_ingredient", "inactive_ingredient", "indications_and_usage", "dosage_form"]
            
            field_options = []
            for field in default_fields:
                if field in available_fields:
                    field_options.append(field)
                    
            for field in sorted(available_fields):
                if field not in default_fields:
                    field_options.append(field)
        except Exception as e:
            st.warning(f"Could not fetch all OpenFDA fields: {e}")
            field_options = ["active_ingredient", "inactive_ingredient", "indications_and_usage", 
                           "dosage_form", "warnings", "description", "contraindications"]
        
        labels_to_get = st.multiselect(
            "Select fields to retrieve",
            options=field_options,
            default=["active_ingredient", "inactive_ingredient"],
            key="fields_single_search"
        )
        
        include_sources = st.checkbox("Include data sources", value=True, key="sources_single_search")
        
        # Store the preview button state and handle clicks
        preview_btn = st.button("Preview Results", on_click=update_search)
        
        export_format = st.radio(
            "Output Format",
            ["JSON", "CSV", "Excel", "TXT"],
            horizontal=True,
            key="format_single_search"
        )

    with col_right:
        # Process search if button was clicked or if we have previous search results
        if (preview_btn or st.session_state.search_performed) and st.session_state.multi_input.strip():
            items = [x.strip() for x in st.session_state.multi_input.split(",") if x.strip()]
            
            all_results = []
            with st.spinner(f"Processing {len(items)} items..."):
                for item in items:
                    if item.replace('-', '').isdigit():
                        result = search_ndc(item, labels_to_get, include_sources)
                        all_results.append(result)  
                        
                        img_url = get_product_image(item)
                        if img_url:
                            st.image(img_url, caption="Product Label", width=400)
                        else:
                            st.info("No product image available for this NDC.")
                    else:
                        # For name search, store results in session state
                        if not preview_btn and st.session_state.search_results:
                            # Use cached results if not clicking the button again
                            name_results = st.session_state.search_results
                        else:
                            # Perform new search and update session state
                            name_results = fetch_ndcs_for_name_drugsfda(item)
                            st.session_state.search_results = name_results
                        
                        if name_results:
                            # Create container for search results
                            st.markdown("### Search Results")
                            
                            # Create DataFrame from search results
                            df_display = pd.DataFrame(name_results)
                            
                            # Calculate dynamic height based on number of rows, capped at ~8 rows
                            num_rows = len(df_display)
                            # Base height per row (~40px) + header (~45px) + some padding
                            dynamic_height = min(max(num_rows * 40 + 45, 100), 375)  # Min height 100px, max ~8 rows (375px)
                            
                            # Display the table with static data
                            st.dataframe(df_display, height=dynamic_height, use_container_width=True)

                            # Add an export button for the complete table
                            st.markdown("### Export Search Results Table")
                            export_table = df_display.copy()
                            
                            # Add option to include search field data in export
                            include_field_data = st.checkbox("Include data from selected search fields", value=True, key="include_fields_export")
                            
                            # If user wants to include field data, fetch and add it to the export table
                            if include_field_data and labels_to_get:
                                # Create a progress bar for field data fetching
                                fetch_progress = st.progress(0)
                                st.write("Fetching field data for each NDC...")
                                
                                # Create columns for each selected field
                                for field in labels_to_get:
                                    export_table[field] = None
                                
                                # Fetch detailed data for each NDC and add the fields
                                for i, row in export_table.iterrows():
                                    ndc = row['NDC']
                                    # Update progress bar
                                    fetch_progress.progress((i + 1) / len(export_table))
                                    
                                    try:
                                        # Get detailed data for this NDC
                                        detailed_data = search_ndc(ndc, labels_to_get, include_source=False)
                                        
                                        # Add each selected field to the export table
                                        for field in labels_to_get:
                                            if field in detailed_data:
                                                export_table.at[i, field] = detailed_data[field]
                                    except Exception as e:
                                        st.warning(f"Could not fetch detailed data for NDC {ndc}: {e}")
                                
                                # Complete the progress
                                fetch_progress.progress(1.0)
                                st.success("Field data fetched successfully!")
                            
                            # If user selects export, provide download in selected format
                            if st.button("Export Table Results"):
                                if export_format == "CSV":
                                    csv = convert_df(export_table, "CSV")
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name="search_results.csv",
                                        mime="text/csv"
                                    )
                                elif export_format == "Excel":
                                    excel = convert_df(export_table, "Excel")
                                    st.download_button(
                                        label="Download Excel",
                                        data=excel,
                                        file_name="search_results.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                elif export_format == "TXT":
                                    txt = convert_df(export_table, "TXT")
                                    st.download_button(
                                        label="Download TXT",
                                        data=txt,
                                        file_name="search_results.txt",
                                        mime="text/plain"
                                    )
                                else:  # JSON
                                    json_str = convert_df(export_table, "JSON")
                                    st.download_button(
                                        label="Download JSON",
                                        data=json_str,
                                        file_name="search_results.json",
                                        mime="application/json"
                                    )
                            
                            # Create a list of NDC options for the multiselect
                            ndc_options = [ndc_item['NDC'] for ndc_item in name_results]
                            
                            # Create a multiselect for the NDCs - use empty list when options change
                            st.markdown("### Select NDC(s) to preview")
                            
                            # Only use existing selections if they're still valid options
                            valid_defaults = [ndc for ndc in st.session_state.selected_ndcs if ndc in ndc_options]
                            
                            selected_ndcs = st.multiselect(
                                "Choose one or more NDCs to view details",
                                options=ndc_options,
                                default=valid_defaults,
                                key="ndc_multiselect"
                            )
                            
                            # Update the session state
                            st.session_state.selected_ndcs = selected_ndcs
                            
                            # Display detailed information for selected NDCs
                            if selected_ndcs:
                                st.markdown("### Detailed Information for Selected NDC(s)")
                                
                                # Create tabs for each selected NDC
                                ndc_tabs = st.tabs(selected_ndcs)
                                
                                # For each tab, display the detailed information for that NDC
                                for i, ndc in enumerate(selected_ndcs):
                                    with ndc_tabs[i]:
                                        # Get all fields for the NDC
                                        detailed_data = search_ndc(ndc, labels_to_get, include_sources)
                                        
                                        # Display the data in the selected format
                                        if export_format == "JSON":
                                            st.json(detailed_data)
                                        elif export_format == "CSV":
                                            df_detailed = pd.DataFrame([detailed_data])
                                            st.dataframe(df_detailed)
                                            csv = convert_df(df_detailed, "CSV")
                                            st.download_button(
                                                label="Download CSV",
                                                data=csv,
                                                file_name=f"{ndc}_details.csv",
                                                mime="text/csv"
                                            )
                                        elif export_format == "Excel":
                                            df_detailed = pd.DataFrame([detailed_data])
                                            st.dataframe(df_detailed)
                                            excel = convert_df(df_detailed, "Excel")
                                            st.download_button(
                                                label="Download Excel",
                                                data=excel,
                                                file_name=f"{ndc}_details.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                            )
                                        elif export_format == "TXT":
                                            df_detailed = pd.DataFrame([detailed_data])
                                            st.dataframe(df_detailed)
                                            txt = convert_df(df_detailed, "TXT")
                                            st.download_button(
                                                label="Download TXT",
                                                data=txt,
                                                file_name=f"{ndc}_details.txt",
                                                mime="text/plain"
                                            )
            
            # Export all combined results if not name search
            if all_results and len(all_results) == len(items) and all(isinstance(item, dict) for item in all_results):
                df_combined = pd.DataFrame(all_results)
                combined_data = convert_df(df_combined, export_format)
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=combined_data,
                    file_name=f"ndc_search_results.{export_format.lower()}",
                    mime="text/csv" if export_format == "CSV" else "application/octet-stream"
                )

                # For NDC searches, add export functionality
                if all_results and len(all_results) == 1:
                    st.markdown("### Export Search Result")
                    if st.button("Export NDC Details"):
                        result_df = pd.DataFrame([all_results[0]])
                        if export_format == "JSON":
                            json_str = convert_df(result_df, "JSON")
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name="ndc_details.json",
                                mime="application/json"
                            )
                        elif export_format == "CSV":
                            csv = convert_df(result_df, "CSV")
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="ndc_details.csv",
                                mime="text/csv"
                            )
                        elif export_format == "Excel":
                            excel = convert_df(result_df, "Excel")
                            st.download_button(
                                label="Download Excel",
                                data=excel,
                                file_name="ndc_details.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        elif export_format == "TXT":
                            txt = convert_df(result_df, "TXT")
                            st.download_button(
                                label="Download TXT",
                                data=txt,
                                file_name="ndc_details.txt",
                                mime="text/plain"
                            )

def get_package_ndcs_for_product(product_ndc):
    """
    Given a product NDC, retrieve all package NDCs associated with it from the NDC directory.
    Returns a list of package NDCs.
    """
    base_url = "https://api.fda.gov/drug/ndc.json"
    query = f'product_ndc:"{product_ndc}"'
    params = {"search": query, "limit": 1}
    
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "results" in data and data["results"]:
            result = data["results"][0]
            packaging = result.get("packaging", [])
            package_ndcs = []
            
            for pkg in packaging:
                package_ndc = pkg.get("package_ndc")
                if package_ndc:
                    package_ndcs.append(package_ndc)
            
            return package_ndcs
    except Exception as e:
        print(f"Error retrieving package NDCs: {e}")
    
    return []

# AI Assistant Section - Only show when requested
st.markdown("---")
show_assistant = st.checkbox("Show AI Assistant", value=False, key="show_assistant")
if show_assistant:
    # Integrate AI chatbot here
    st.markdown("### PillQ AI Assistant")
    st.markdown("Ask me anything about drug information and I'll help you find answers.")
    
    # Initialize chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**AI:** {message['content']}")
    
    user_question = st.text_area("Your question:", key="ai_question", height=100)
    
    if st.button("Ask AI Assistant"):
        if user_question:
            # Display the user message
            st.write(f"**You:** {user_question}")
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Call the AI assistant with the question
            with st.spinner("Thinking..."):
                response = call_ai_assistant(user_question)
                
            # Display the AI response
            st.write(f"**AI:** {response}")
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Clear the question input
            st.session_state.ai_question = ""
        else:
            st.warning("Please enter a question first.")

# Function to call AI assistant
def call_ai_assistant(question, context=""):
    """
    Call the selected AI model with the question and return the response.
    Supports OpenAI, Ollama, and DeepSeek.
    """
    provider = st.session_state.selected_model
    
    if provider == "OpenAI":
        if not st.session_state.openai_api_key:
            return "Error: OpenAI API key is not configured. Please set it up in the Settings tab."
        
        try:
            import openai
            openai.api_key = st.session_state.openai_api_key
            
            prompt = f"You are an AI assistant for a pharmaceutical information tool called PillQ. Answer the following question about drugs or pharmaceuticals:\n\n{question}"
            if context:
                prompt += f"\n\nContext information: {context}"
            
            response = openai.chat.completions.create(
                model=st.session_state.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful pharmaceutical assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"
    
    elif provider == "Ollama":
        if not st.session_state.ollama_url:
            return "Error: Ollama URL is not configured. Please set it up in the Settings tab."
        
        try:
            url = f"{st.session_state.ollama_url}/api/chat"
            
            prompt = f"You are an AI assistant for a pharmaceutical information tool called PillQ. Answer the following question about drugs or pharmaceuticals:\n\n{question}"
            if context:
                prompt += f"\n\nContext information: {context}"
            
            payload = {
                "model": st.session_state.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful pharmaceutical assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Error calling Ollama API: {str(e)}"
    
    elif provider == "DeepSeek":
        if not st.session_state.deepseek_api_key:
            return "Error: DeepSeek API key is not configured. Please set it up in the Settings tab."
        
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            
            prompt = f"You are an AI assistant for a pharmaceutical information tool called PillQ. Answer the following question about drugs or pharmaceuticals:\n\n{question}"
            if context:
                prompt += f"\n\nContext information: {context}"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {st.session_state.deepseek_api_key}"
            }
            
            payload = {
                "model": st.session_state.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful pharmaceutical assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling DeepSeek API: {str(e)}"
    
    elif provider == "HuggingFace":
        try:
            import torch
            from transformers import pipeline
            
            model_name = st.session_state.model_name
            model = pipeline("text-generation", model=model_name)
            
            prompt = f"You are an AI assistant for a pharmaceutical information tool called PillQ. Answer the following question about drugs or pharmaceuticals:\n\n{question}"
            if context:
                prompt += f"\n\nContext information: {context}"
            
            response = model(prompt, max_length=1000)
            return response[0]["generated_text"]
        except Exception as e:
            return f"Error calling HuggingFace API: {str(e)}"
    
    else:
        return "Error: Unknown AI provider selected."

with tab2:
    st.subheader("File Upload")
    
    uploaded_file = st.file_uploader("Upload a file containing NDC codes:", type=['csv', 'xlsx', 'txt', 'json'])
    
    if uploaded_file is not None:
        # Determine file type from extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext == 'xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_ext == 'txt':
                # Try to determine delimiter
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            elif file_ext == 'json':
                df = pd.read_json(uploaded_file)
            
            st.success(f"File uploaded successfully! Found {len(df)} rows.")
            
            # Show the first 5 rows
            st.write("Preview of your data:")
            st.dataframe(df.head())
            
            # Let user select columns that contain NDC codes
            st.markdown("### Select columns containing NDC codes")
            ndc_columns = st.multiselect(
                "Choose columns containing NDC codes:",
                options=df.columns.tolist()
            )
            
            if ndc_columns:
                # Let user select which columns to keep
                st.markdown("### Select columns to keep")
                keep_columns = st.multiselect(
                    "Choose columns to keep in output:",
                    options=df.columns.tolist(),
                    default=ndc_columns
                )
                
                # Select OpenFDA fields to add
                st.markdown("### Select OpenFDA fields to add")
                openfda_fields = st.multiselect(
                    "Choose OpenFDA fields to add:",
                    options=field_options,
                    default=["active_ingredient", "inactive_ingredient"]
                )
                
                include_sources_upload = st.checkbox("Include data sources", value=True)
                
                if st.button("Process File"):
                    with st.spinner("Processing your file..."):
                        # Create a new dataframe with selected columns
                        output_df = df[keep_columns].copy()
                        
                        # Process NDC columns
                        for col in ndc_columns:
                            # Create progress bar
                            total_rows = len(df)
                            progress_bar = st.progress(0)
                            
                            for i, ndc in enumerate(df[col]):
                                # Update progress
                                progress_bar.progress((i + 1) / total_rows)
                                
                                # Skip empty/NA values
                                if pd.isna(ndc) or ndc == "":
                                    continue
                                
                                # Clean the NDC string
                                ndc_str = str(ndc).strip()
                                
                                # Get data for this NDC
                                try:
                                    ndc_data = search_ndc(ndc_str, openfda_fields, include_sources_upload)
                                    
                                    # Add each field as a new column
                                    for field in openfda_fields:
                                        field_col_name = f"{col}_{field}"
                                        output_df.at[i, field_col_name] = ndc_data.get(field, "Not Available")
                                        
                                        if include_sources_upload and field + "_source" in ndc_data:
                                            source_col_name = f"{col}_{field}_source"
                                            output_df.at[i, source_col_name] = ndc_data[field + "_source"]
                                except Exception as e:
                                    st.warning(f"Error processing NDC {ndc_str}: {e}")
                        
                        # Display the output
                        st.success("Processing complete!")
                        st.write("Output:")
                        st.dataframe(output_df)
                        
                        # Allow downloading the result
                        output_format = st.radio(
                            "Output Format",
                            ["CSV", "Excel", "JSON", "TXT"],
                            horizontal=True
                        )
                        
                        output_data = convert_df(output_df, output_format)
                        st.download_button(
                            label=f"Download {output_format}",
                            data=output_data,
                            file_name=f"processed_ndc_data.{output_format.lower()}",
                            mime="text/csv" if output_format == "CSV" else "application/octet-stream"
                        )
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

with tab3:
    st.subheader("AI Assistant Settings")
    st.markdown("Configure your AI assistant to use different providers. API keys and settings are stored locally on your machine.")
    
    # AI Provider selection
    st.subheader("AI Provider")
    provider_col1, provider_col2 = st.columns([1, 2])
    
    with provider_col1:
        provider_options = ["OpenAI", "Ollama", "DeepSeek", "HuggingFace"]
        selected_provider = st.selectbox(
            "Select AI Provider",
            options=provider_options,
            index=provider_options.index(st.session_state.selected_model)
        )
        
        if selected_provider != st.session_state.selected_model:
            st.session_state.selected_model = selected_provider
    
    with provider_col2:
        # Different model options based on selected provider
        if selected_provider == "OpenAI":
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            model_name = st.selectbox(
                "Select Model",
                options=model_options,
                index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0
            )
        elif selected_provider == "Ollama":
            model_options = ["llama2", "mistral", "mixtral", "gemma", "phi"]
            model_name = st.selectbox(
                "Select Model",
                options=model_options,
                index=0
            )
        elif selected_provider == "DeepSeek":
            model_options = ["deepseek-chat", "deepseek-coder"]
            model_name = st.selectbox(
                "Select Model",
                options=model_options,
                index=0
            )
        elif selected_provider == "HuggingFace":
            model_name = st.text_input(
                "Enter HuggingFace Model Name",
                value=st.session_state.model_name,
                help="Enter the name of the HuggingFace model you want to use. You can find the list of available models at https://huggingface.co/models",
                key="huggingface_model_name_dropdown"
            )
        
        if model_name != st.session_state.model_name:
            st.session_state.model_name = model_name
    
    # API Key configuration
    st.subheader("API Configuration")
    
    # OpenAI configuration
    if selected_provider == "OpenAI":
        st.markdown("##### OpenAI API Key")
        openai_key = st.text_input(
            "Enter your OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Your OpenAI API key. Get one at https://platform.openai.com/api-keys"
        )
        if openai_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = openai_key
    
    # Ollama configuration
    elif selected_provider == "Ollama":
        st.markdown("##### Ollama URL")
        st.markdown("Ollama runs locally on your machine. Make sure you have it installed and running.")
        ollama_url = st.text_input(
            "Ollama URL",
            value=st.session_state.ollama_url,
            help="The URL of your Ollama instance. Default is http://localhost:11434"
        )
        if ollama_url != st.session_state.ollama_url:
            st.session_state.ollama_url = ollama_url
    
    # DeepSeek configuration
    elif selected_provider == "DeepSeek":
        st.markdown("##### DeepSeek API Key")
        deepseek_key = st.text_input(
            "Enter your DeepSeek API Key",
            value=st.session_state.deepseek_api_key,
            type="password",
            help="Your DeepSeek API key. Get one at https://platform.deepseek.com/"
        )
        if deepseek_key != st.session_state.deepseek_api_key:
            st.session_state.deepseek_api_key = deepseek_key
    
    # HuggingFace configuration
    elif selected_provider == "HuggingFace":
        st.markdown("##### HuggingFace Model Name")
        st.markdown("Enter the name of the HuggingFace model you want to use. You can find the list of available models at https://huggingface.co/models")
        huggingface_model_name = st.text_input(
            "Enter HuggingFace Model Name",
            value=st.session_state.model_name,
            help="Enter the name of the HuggingFace model you want to use. You can find the list of available models at https://huggingface.co/models",
            key="huggingface_model_name_settings"
        )
        if huggingface_model_name != st.session_state.model_name:
            st.session_state.model_name = huggingface_model_name
    
    # Save settings button
    if st.button("Save Settings"):
        save_settings()
        st.success("Settings saved successfully!")
    
    # Test connection button
    if st.button("Test Connection"):
        with st.spinner("Testing connection..."):
            test_result = call_ai_assistant("This is a test message to check the connection.")
            if "Error" in test_result:
                st.error(test_result)
            else:
                st.success("Connection successful! Your AI assistant is ready to use.")

# Add the artistic pills illustration to the bottom left
try:
    # Read the SVG file and convert to base64
    svg_path = os.path.join(os.path.dirname(__file__), "artistic_pills_illustration.svg")
    if os.path.exists(svg_path):
        with open(svg_path, "rb") as f:
            svg_data = base64.b64encode(f.read()).decode("utf-8")
            
        st.markdown(
            f"""
            <style>
            .pill-illustration {{
                position: fixed;
                bottom: 20px;
                left: 20px;
                width: 150px;
                height: auto;
                z-index: 1000;
            }}
            </style>
            <div class="pill-illustration">
                <img src="data:image/svg+xml;base64,{svg_data}" alt="Artistic Pills Illustration">
            </div>
            """,
            unsafe_allow_html=True
        )
except Exception as e:
    st.write(f"Note: Illustration couldn't be loaded. This doesn't affect app functionality.")