import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime
from openai import OpenAI
import requests
import os

from helper_functions import (
    get_openfda_searchable_fields,
    get_product_image,
    get_product_ndc,
    get_label_field,
    get_combined_label_field,
    get_setid_from_search,
    fallback_ndc_search,
    unify_source_string,
    fetch_ndcs_for_name_drugsfda,
    search_ndc,
    search_name_placeholder,
    get_single_item_source,
    convert_df,
    clean_html_table,
    is_safe_table
)

# ------------------------------------------------------------------
# Initialize session state
# ------------------------------------------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'chat_id' not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if 'ai_provider' not in st.session_state:
    st.session_state.ai_provider = "ollama"  # Default to free model
    
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
    
if 'deepseek_api_key' not in st.session_state:
    st.session_state.deepseek_api_key = ""

if 'selected_ndcs' not in st.session_state:
    st.session_state.selected_ndcs = []

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

if 'current_ai_model' not in st.session_state:
    st.session_state.current_ai_model = 'openai'  # Default to OpenAI

if 'export_form_submitted' not in st.session_state:
    st.session_state.export_form_submitted = False

# ------------------------------------------------------------------
# Title & Description
# ------------------------------------------------------------------
st.set_page_config(
    page_title="PillQ - Drug Lookup Tool",
    page_icon="💊",
    layout="wide"
)

# Page Header and Description
st.title("💊 PillQ - Drug Information Lookup")
st.markdown("""
PillQ deciphers complex drug data in an instant—whether for formulary management, verification, or documentation.  
Pill Queries, Simplified.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Single/Multiple Search", "File Upload", "Settings"])

# ------------------------------------------------------------------
# Layout with three tabs
# ------------------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Single/Multiple Search")
        
        # Initialize session state variables if they don't exist
        if 'multi_input' not in st.session_state:
            st.session_state.multi_input = ""
        if 'search_performed' not in st.session_state:
            st.session_state.search_performed = False
        
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
        
        include_sources = st.checkbox("Include data sources", value=False, key="sources_single_search")
        
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
                        result = search_ndc(item, labels_to_get, include_source=include_sources)
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
                            
                            # Create DataFrame from search results with specific fields and order
                            df_display = pd.DataFrame(name_results)[["NDC", "brand_name", "generic_name", "strength", 
                                                                   "route", "dosage_form", "manufacturer", "package_description"]]
                            
                            # Ensure all columns are string or compatible types for PyArrow
                            for col in df_display.columns:
                                df_display[col] = df_display[col].astype(str)
                            
                            # Calculate dynamic height based on number of rows, capped at ~8 rows
                            num_rows = len(df_display)
                            # Base height per row (~40px) + header (~45px) + some padding
                            dynamic_height = min(max(num_rows * 40 + 45, 100), 375)  # Min height 100px, max ~8 rows (375px)
                            
                            # Display the table with static data
                            st.dataframe(df_display, height=dynamic_height, use_container_width=True)
                            
                            # Create a list of NDC options for the multiselect
                            ndc_options = [ndc_item['NDC'] for ndc_item in name_results]
                            
                            # Step 1: Create a multiselect for the NDCs - use empty list when options change
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
                                st.markdown("---")
                                st.markdown("### Detailed Information for Selected NDC(s)")
                                
                                # For each selected NDC
                                for ndc in selected_ndcs:
                                    try:
                                        # Get detailed data for this NDC
                                        detailed_data = search_ndc(ndc, labels_to_get, include_source=include_sources)
                                        
                                        # Create an expander for each NDC
                                        with st.expander(f"**NDC: {ndc}**", expanded=True):
                                            # Create tabs for different ways to view the data
                                            view_tabs = st.tabs(["Table View", "Key-Value View", "JSON View"])
                                            
                                            # Tab 1: Table View
                                            with view_tabs[0]:
                                                # Convert dictionary to DataFrame
                                                detailed_df = pd.DataFrame([detailed_data])
                                                
                                                # Transpose the DataFrame for better display
                                                detailed_df_transposed = detailed_df.T.reset_index()
                                                detailed_df_transposed.columns = ["Field", "Value"]
                                                
                                                # Handle list values to ensure proper display
                                                detailed_df_transposed["Value"] = detailed_df_transposed["Value"].apply(
                                                    lambda x: json.dumps(x) if isinstance(x, list) else x
                                                )
                                                
                                                # Create a copy of the data for display purposes
                                                display_df = detailed_df_transposed.copy()
                                                
                                                # Check for HTML table fields that should be rendered
                                                html_table_fields = ["dosage_and_administration_table", "clinical_studies_table", 
                                                                    "indications_usage_table", "warnings_table"]
                                                
                                                # Add a Source column if include_sources is True
                                                if include_sources:
                                                    # Create a new column for Source
                                                    sources = []
                                                    for field in detailed_df_transposed["Field"]:
                                                        if field.endswith("_source"):
                                                            sources.append("N/A")
                                                        else:
                                                            # Get source for this field
                                                            source = get_single_item_source(detailed_data, field)
                                                            sources.append(source)
                                                    
                                                    # Add Source column to DataFrame
                                                    detailed_df_transposed["Source"] = sources
                                                    
                                                    # Filter out _source fields
                                                    detailed_df_transposed = detailed_df_transposed[~detailed_df_transposed["Field"].str.endswith("_source")]
                                                
                                                # Display the transposed DataFrame
                                                st.dataframe(detailed_df_transposed, use_container_width=True)
                                                
                                                # Render HTML tables for specific fields if they exist
                                                for field in html_table_fields:
                                                    row = detailed_df_transposed[detailed_df_transposed["Field"] == field]
                                                    
                                                    if not row.empty and isinstance(row["Value"].iloc[0], str) and "<table" in row["Value"].iloc[0]:
                                                        table_html = row["Value"].iloc[0]
                                                        
                                                        st.markdown(f"### Rendered {field.replace('_', ' ').title()}")
                                                        if is_safe_table(table_html):  # Validate before rendering
                                                            cleaned_html = clean_html_table(table_html)
                                                            st.markdown(cleaned_html, unsafe_allow_html=True)
                                                        else:
                                                            st.warning(f"HTML content contains potentially unsafe elements and was not rendered.")
                                                        st.markdown("---")
                                                
                                                # Also check regular fields for HTML tables 
                                                regular_fields_with_tables = ["adverse_reactions", "clinical_studies", "instructions_for_use", "dosage_and_administration"]
                                                for field in regular_fields_with_tables:
                                                    row = detailed_df_transposed[detailed_df_transposed["Field"] == field]
                                                    
                                                    if not row.empty and isinstance(row["Value"].iloc[0], str) and "<table" in row["Value"].iloc[0]:
                                                        table_html = row["Value"].iloc[0]
                                                        
                                                        st.markdown(f"### Rendered {field.replace('_', ' ').title()} Table")
                                                        if is_safe_table(table_html):  # Validate before rendering
                                                            cleaned_html = clean_html_table(table_html)
                                                            st.markdown(cleaned_html, unsafe_allow_html=True)
                                                        else:
                                                            st.warning(f"HTML content contains potentially unsafe elements and was not rendered.")
                                                        st.markdown("---")
                                            
                                            # Tab 2: Key-Value View
                                            with view_tabs[1]:
                                                # Create columns for Field/Value pairs
                                                for field, value in detailed_data.items():
                                                    # Skip source fields
                                                    if field.endswith("_source"):
                                                        continue
                                                    
                                                    # Create columns for field and value
                                                    cols = st.columns([1, 3])
                                                    
                                                    # Display field name in first column
                                                    cols[0].markdown(f"**{field}:**")
                                                    
                                                    # Display value in second column
                                                    if field in html_table_fields and isinstance(value, str) and "<table" in value:
                                                        # For HTML table fields, render the HTML if safe
                                                        if is_safe_table(value):
                                                            cleaned_html = clean_html_table(value)
                                                            cols[1].markdown(cleaned_html, unsafe_allow_html=True)
                                                        else:
                                                            cols[1].warning("HTML content not displayed due to security concerns.")
                                                    else:
                                                        # For regular fields, use standard display
                                                        cols[1].write(value)
                                                    
                                                    # Display source if include_sources is True
                                                    if include_sources:
                                                        source = get_single_item_source(detailed_data, field)
                                                        cols[1].caption(f"Source: {source}")
                                            
                                            # Tab 3: JSON View
                                            with view_tabs[2]:
                                                # Display the raw JSON
                                                st.json(detailed_data)
                                                
                                            # Add a divider after each NDC section
                                            st.divider()
                                    except Exception as e:
                                        st.error(f"Error loading data for NDC {ndc}: {e}")
                            
                            # Step 2: Show export options ONLY when export button is clicked
                            st.markdown("---")
                            
                            # Create a container for export options to maintain state
                            export_container = st.container()
                            
                            # Button to open export options
                            if not st.session_state.get('show_export_options', False):
                                if st.button("Open Export Options"):
                                    st.session_state.show_export_options = True
                                    st.session_state.export_form_submitted = False
                                    st.rerun()
                            
                            # Show export options if the button was clicked
                            if st.session_state.get('show_export_options', False):
                                with export_container:
                                    st.markdown("### Export Search Results Table")
                                    export_table = df_display.copy()
                                    
                                    # Create a form to maintain state during submission
                                    with st.form(key="export_form"):
                                        # Add option to include search field data in export
                                        include_field_data = st.checkbox("Include data from selected search fields", value=True, key="include_fields_export")
                                        
                                        # Add option to select specific columns to export
                                        st.markdown("#### Choose columns to include in export")
                                        
                                        # Default columns (always selected)
                                        default_columns = ["NDC", "brand_name", "generic_name", "strength", "route", "dosage_form", "manufacturer", "package_description"]
                                        
                                        # Let user choose which base columns to include
                                        base_columns_to_export = st.multiselect(
                                            "Base table columns",
                                            options=default_columns,
                                            default=default_columns,
                                            key="base_columns_export"
                                        )
                                        
                                        # Field columns selection (only shown if include_field_data is checked)
                                        field_columns_to_export = []
                                        if include_field_data and labels_to_get:
                                            field_columns_to_export = st.multiselect(
                                                "Additional field data columns",
                                                options=labels_to_get,
                                                default=labels_to_get,
                                                key="field_columns_export"
                                            )
                                        
                                        # Submit button for the form
                                        export_submitted = st.form_submit_button("Prepare Export")
                                        
                                        if export_submitted:
                                            st.session_state.export_form_submitted = True
                                            st.session_state.include_field_data = include_field_data
                                            st.session_state.base_columns = base_columns_to_export
                                            st.session_state.field_columns = field_columns_to_export
                                    
                                    # Handle form submission outside the form
                                    if st.session_state.export_form_submitted:
                                        # Process the export based on selections stored in session state
                                        include_field_data = st.session_state.include_field_data
                                        base_columns_to_export = st.session_state.base_columns
                                        field_columns_to_export = st.session_state.field_columns
                                        
                                        # If user wants to include field data, fetch and add it to the export table
                                        if include_field_data and labels_to_get and field_columns_to_export:
                                            # Create a progress bar for field data fetching
                                            fetch_progress = st.progress(0)
                                            st.write("Fetching field data for each NDC...")
                                            
                                            # Create columns for each selected field
                                            for field in field_columns_to_export:
                                                if field not in export_table.columns:
                                                    export_table[field] = None
                                            
                                            # Fetch detailed data for each NDC and add the fields
                                            for i, row in export_table.iterrows():
                                                ndc = row['NDC']
                                                # Update progress bar
                                                fetch_progress.progress((i + 1) / len(export_table))
                                                
                                                try:
                                                    # Get detailed data for this NDC
                                                    detailed_data = search_ndc(ndc, labels_to_get, include_source=include_sources)
                                                    
                                                    # Add each selected field to the export table
                                                    for field in field_columns_to_export:
                                                        if field in detailed_data:
                                                            export_table.at[i, field] = detailed_data[field]
                                                except Exception as e:
                                                    st.warning(f"Could not fetch detailed data for NDC {ndc}: {e}")
                                            
                                            # Complete the progress
                                            fetch_progress.progress(1.0)
                                            st.success("Field data fetched successfully!")
                                            
                                            # Final list of columns to export
                                            final_columns_to_export = base_columns_to_export + [col for col in field_columns_to_export if col in labels_to_get]
                                        else:
                                            final_columns_to_export = base_columns_to_export
                                        
                                        # Filter the export table to include only the selected columns
                                        filtered_export_table = export_table[final_columns_to_export]
                                        
                                        # Provide download options
                                        if export_format == "CSV":
                                            csv = convert_df(filtered_export_table, "CSV")
                                            st.download_button(
                                                label="Download CSV",
                                                data=csv,
                                                file_name="search_results.csv",
                                                mime="text/csv",
                                                key="download_csv_button"
                                            )
                                        elif export_format == "Excel":
                                            excel = convert_df(filtered_export_table, "Excel")
                                            st.download_button(
                                                label="Download Excel",
                                                data=excel,
                                                file_name="search_results.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                key="download_excel_button"
                                            )
                                        elif export_format == "TXT":
                                            txt = convert_df(filtered_export_table, "TXT")
                                            st.download_button(
                                                label="Download TXT",
                                                data=txt,
                                                file_name="search_results.txt",
                                                mime="text/plain",
                                                key="download_txt_button"
                                            )
                                        else:  # JSON
                                            json_str = convert_df(filtered_export_table, "JSON")
                                            st.download_button(
                                                label="Download JSON",
                                                data=json_str,
                                                file_name="search_results.json",
                                                mime="application/json",
                                                key="download_json_button"
                                            )
                                        
                                        # Add a button to close the export options
                                        if st.button("Close Export Options"):
                                            st.session_state.show_export_options = False
                                            st.session_state.export_form_submitted = False
                                            st.rerun()
            
            if all_results:
                if len(all_results) == 1:
                    if export_format == "JSON":
                        st.json(all_results[0])
                    elif export_format == "CSV":
                        st.dataframe(pd.DataFrame([all_results[0]]))
                    elif export_format == "Excel":
                        st.dataframe(pd.DataFrame([all_results[0]]))
                    elif export_format == "TXT":
                        st.text(pd.DataFrame([all_results[0]]).to_csv(sep="\t", index=False))
                else:
                    df = pd.DataFrame(all_results)
                    
                    if 'ndc' in df.columns:
                        df = df.rename(columns={'ndc': 'NDC'})
                        
                    st.dataframe(df)
                
                st.download_button(
                    label=f"Download as {export_format}",
                    data=convert_df(pd.DataFrame(all_results), export_format),
                    file_name=f"pillq_export.{export_format.lower()}",
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

    # AI Assistant Section - Only show when requested
    st.markdown("---")
    show_assistant = st.checkbox("Show AI Assistant", value=False, key="show_assistant")
    
    if show_assistant:
        st.subheader("AI Assistant")
        
        col1, col2 = st.columns([2, 1])
        
        model_options = {
            "gemini": "Gemini 1.0 Pro (Google, Free API)",
            "cloudflare": "Claude Instant (Cloudflare, Free)",
            "zephyr": "Zephyr 7B (HuggingFace, Free Cloud)",
            "ollama": "Llama 3 (Ollama, Free Local)",
            "cohere": "Command (Cohere, Free API)",
            "anthropic": "Claude 3 Haiku (Anthropic, Free API)",
            "openai": "GPT-3.5 Turbo (OpenAI, API Required)",
            "deepseek": "DeepSeek Chat (DeepSeek, API Required)"
        }
        
        st.session_state.current_ai_model = st.selectbox(
            "Select AI Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.current_ai_model),
            key="ai_model_selector"
        )
        
        with st.container(height=250, border=True):
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"<div style='background-color:#2E4057; padding:5px; border-radius:3px; margin-bottom:5px; color:white;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color:#1E293B; padding:5px; border-radius:3px; margin-bottom:5px; color:white;'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)
        
        input_col, button_col = st.columns([3, 1])
        
        user_input = st.text_area("Ask about drugs, ingredients, or how to use this tool", height=70, key="chat_input")
        
        st.write("")  # Add some vertical spacing
        submit_button = st.button("Ask", type="primary", key="submit_chat", use_container_width=True)
        
        def query_drug_info(ndc_or_name, fields_to_get=None):
            if fields_to_get is None:
                fields_to_get = ["active_ingredient", "inactive_ingredient", "indications_and_usage", "dosage_form"]
            
            if str(ndc_or_name).replace('-', '').isdigit():
                return search_ndc(ndc_or_name, fields_to_get, include_source=True)
            else:
                return fetch_ndcs_for_name_drugsfda(ndc_or_name)
        
        def get_available_fields():
            return list(get_openfda_searchable_fields())
        
        if submit_button or user_input and user_input != st.session_state.get('previous_input', ''):
            st.session_state.previous_input = user_input
            
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.current_ai_model == "openai" and st.session_state.openai_api_key:
                        client = OpenAI(api_key=st.session_state.openai_api_key)
                        
                        system_message = """
                        You are PillQ Assistant, an AI helper specialized in pharmaceutical information. 
                        You can help users find information about drugs, analyze ingredients, and understand medication data.
                        
                        You have access to the following tools:
                        1. query_drug_info(ndc_or_name, fields_to_get): Look up drug information by NDC code or name
                        2. get_available_fields(): Get a list of all available data fields from OpenFDA
                        
                        When users ask about specific drugs, use the query_drug_info tool to fetch accurate information.
                        Explain pharmaceutical concepts in clear, accessible language.
                        """
                        
                        messages = [
                            {"role": "system", "content": system_message}
                        ]
                        
                        for message in st.session_state.chat_history[-10:]:
                            messages.append({"role": message["role"], "content": message["content"]})
                        
                        tools = [
                            {
                                "type": "function",
                                "function": {
                                    "name": "query_drug_info",
                                    "description": "Get information about a drug by NDC or name",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "ndc_or_name": {
                                                "type": "string",
                                                "description": "NDC code or drug name to search for"
                                            },
                                            "fields_to_get": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "List of fields to get from OpenFDA"
                                            }
                                        },
                                        "required": ["ndc_or_name"]
                                    }
                                }
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_available_fields",
                                    "description": "Get a list of all available data fields from OpenFDA",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {}
                                    }
                                }
                            }
                        ]
                        
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo-0125",
                            messages=messages,
                            tools=tools,
                            tool_choice="auto"
                        )
                        
                        response_message = response.choices[0].message
                        tool_calls = response_message.tool_calls
                        
                        if tool_calls:
                            for tool_call in tool_calls:
                                function_name = tool_call.function.name
                                function_args = json.loads(tool_call.function.arguments)
                                
                                if function_name == "query_drug_info":
                                    ndc_or_name = function_args.get("ndc_or_name")
                                    fields_to_get = function_args.get("fields_to_get")
                                    
                                    function_response = query_drug_info(ndc_or_name, fields_to_get)
                                    
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": function_name,
                                        "content": json.dumps(function_response)
                                    })
                                
                                elif function_name == "get_available_fields":
                                    function_response = get_available_fields()
                                    
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": function_name,
                                        "content": json.dumps(function_response)
                                    })
                            
                            second_response = client.chat.completions.create(
                                model="gpt-3.5-turbo-0125",
                                messages=messages
                            )
                            
                            final_response = second_response.choices[0].message.content
                        else:
                            final_response = response_message.content
                    
                    elif st.session_state.current_ai_model == "deepseek" and st.session_state.deepseek_api_key:
                        headers = {
                            "Authorization": f"Bearer {st.session_state.deepseek_api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        messages = []
                        for message in st.session_state.chat_history[-10:]:
                            messages.append({"role": message["role"], "content": message["content"]})
                        
                        context = ""
                        if any(keyword in user_input.lower() for keyword in ["drug", "ndc", "information", "ingredient"]):
                            words = user_input.split()
                            for word in words:
                                if word.replace('-', '').isdigit() or len(word) > 4:
                                    try:
                                        drug_info = query_drug_info(word)
                                        context = f"Drug information for {word}: {json.dumps(drug_info)}\n\n"
                                        break
                                    except Exception:
                                        pass
                        
                        system_message = f"""You are PillQ Assistant, an AI helper specialized in pharmaceutical information.
                        {context}
                        Provide helpful, accurate information about drugs. Explain pharmaceutical concepts in clear language."""
                        
                        data = {
                            "model": "deepseek-chat",
                            "messages": [{"role": "system", "content": system_message}] + messages
                        }
                        
                        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data)
                        if response.status_code == 200:
                            final_response = response.json()["choices"][0]["message"]["content"]
                        else:
                            final_response = f"Error with DeepSeek API: {response.text}"
                    
                    elif st.session_state.current_ai_model == "gemini":
                        try:
                            import google.generativeai as genai
                            
                            gemini_api_key = st.session_state.get('gemini_api_key', '')
                            
                            if not gemini_api_key:
                                final_response = """Please add your Google Gemini API key in the Settings tab to use this model.
                                
Get a free API key at https://aistudio.google.com/app/apikey"""
                            else:
                                genai.configure(api_key=gemini_api_key)
                                
                                context = ""
                                if any(keyword in user_input.lower() for keyword in ["drug", "ndc", "information", "ingredient"]):
                                    words = user_input.split()
                                    for word in words:
                                        if word.replace('-', '').isdigit() or len(word) > 4:
                                            try:
                                                drug_info = query_drug_info(word)
                                                context = f"Drug information for {word}: {json.dumps(drug_info)}\n\n"
                                                break
                                            except Exception:
                                                pass
                                
                                system_prompt = f"""You are PillQ Assistant, an AI helper specialized in pharmaceutical information.
                                {context}
                                Provide helpful, accurate information about drugs. Explain pharmaceutical concepts in clear language."""
                                
                                chat_history = []
                                for message in st.session_state.chat_history[-5:]:
                                    role = "user" if message["role"] == "user" else "model"
                                    chat_history.append({"role": role, "parts": [message["content"]]})
                                
                                model = genai.GenerativeModel('gemini-1.0-pro')
                                
                                chat = model.start_chat(history=chat_history)
                                
                                response = chat.send_message(user_input)
                                final_response = response.text
                        except Exception as e:
                            final_response = f"""I'm sorry, I couldn't connect to the Google Gemini API. Error: {str(e)}
                            
Please make sure you've added a valid API key in the Settings tab."""
                    
                    elif st.session_state.current_ai_model == "cloudflare":
                        try:
                            context = ""
                            if any(keyword in user_input.lower() for keyword in ["drug", "ndc", "information", "ingredient"]):
                                words = user_input.split()
                                for word in words:
                                    if word.replace('-', '').isdigit() or len(word) > 4:
                                        try:
                                            drug_info = query_drug_info(word)
                                            context = f"Drug information for {word}: {json.dumps(drug_info)}\n\n"
                                            break
                                        except Exception:
                                            pass
                            
                            conversation = []
                            for message in st.session_state.chat_history[-5:]:
                                role_prefix = "Human: " if message["role"] == "user" else "Assistant: "
                                conversation.append(f"{role_prefix}{message['content']}")
                            
                            conversation_history = "\n".join(conversation)
                            
                            system_prompt = f"""You are PillQ Assistant, an AI helper specialized in pharmaceutical information.
                            {context}
                            Provide helpful, accurate information about drugs. Explain pharmaceutical concepts in clear language."""
                            
                            full_prompt = f"{system_prompt}\n\n{conversation_history}\nHuman: {user_input}\nAssistant:"
                            
                            response = requests.post(
                                "https://api.cloudflare.com/client/v4/accounts/1decaca3cbd0c7ebffc7cd487f91dce0/ai/run/@cf/claude/claude-instant-1.2",
                                headers={"Content-Type": "application/json"},
                                json={"prompt": full_prompt}
                            )
                            
                            if response.status_code == 200:
                                final_response = response.json()["result"]["response"].strip()
                            else:
                                final_response = """I'm sorry, I couldn't connect to the Cloudflare Workers AI service.
                                
This free service might be experiencing high traffic. Please try again in a moment or select a different model."""
                        except Exception as e:
                            final_response = f"""I'm sorry, I couldn't connect to the Cloudflare Workers AI service. Error: {str(e)}
                            
This free service might be experiencing high traffic. Please try again in a moment or select a different model."""
                    
                    elif st.session_state.current_ai_model == "cohere":
                        try:
                            import cohere
                            
                            cohere_api_key = st.session_state.get('cohere_api_key', '')
                            
                            if not cohere_api_key:
                                final_response = """Please add your Cohere API key in the Settings tab to use this model.
                                
Get a free API key at https://dashboard.cohere.com/api-keys"""
                            else:
                                co = cohere.Client(cohere_api_key)
                                
                                context = ""
                                if any(keyword in user_input.lower() for keyword in ["drug", "ndc", "information", "ingredient"]):
                                    words = user_input.split()
                                    for word in words:
                                        if word.replace('-', '').isdigit() or len(word) > 4:
                                            try:
                                                drug_info = query_drug_info(word)
                                                context = f"Drug information for {word}: {json.dumps(drug_info)}\n\n"
                                                break
                                            except Exception:
                                                pass
                                
                                system_prompt = f"""You are PillQ Assistant, an AI helper specialized in pharmaceutical information.
                                {context}
                                Provide helpful, accurate information about drugs. Explain pharmaceutical concepts in clear language."""
                                
                                chat_history = []
                                for message in st.session_state.chat_history[-5:]:
                                    role = "USER" if message["role"] == "user" else "CHATBOT"
                                    chat_history.append({"role": role, "message": message["content"]})
                                
                                response = co.chat(
                                    message=user_input,
                                    chat_history=chat_history,
                                    model="command",
                                    preamble=system_prompt
                                )
                                
                                final_response = response.text
                        except Exception as e:
                            final_response = f"""I'm sorry, I couldn't connect to the Cohere API. Error: {str(e)}
                            
Please make sure you've added a valid API key in the Settings tab."""
                    
                    elif st.session_state.current_ai_model == "anthropic":
                        try:
                            import anthropic
                            
                            claude_api_key = st.session_state.get('claude_api_key', '')
                            
                            if not claude_api_key:
                                final_response = """Please add your Anthropic Claude API key in the Settings tab to use this model.
                                
Get a free API key at https://console.anthropic.com/settings/keys"""
                            else:
                                client = anthropic.Anthropic(api_key=claude_api_key)
                                
                                context = ""
                                if any(keyword in user_input.lower() for keyword in ["drug", "ndc", "information", "ingredient"]):
                                    words = user_input.split()
                                    for word in words:
                                        if word.replace('-', '').isdigit() or len(word) > 4:
                                            try:
                                                drug_info = query_drug_info(word)
                                                context = f"Drug information for {word}: {json.dumps(drug_info)}\n\n"
                                                break
                                            except Exception:
                                                pass
                                
                                system_message = f"""You are PillQ Assistant, an AI helper specialized in pharmaceutical information.
                                {context}
                                Provide helpful, accurate information about drugs. Explain pharmaceutical concepts in clear language."""
                                
                                messages = [{"role": "system", "content": system_message}]
                                
                                for message in st.session_state.chat_history[-5:]:
                                    role = "user" if message["role"] == "user" else "assistant"
                                    messages.append({"role": role, "content": message["content"]})
                                
                                messages.append({"role": "user", "content": user_input})
                                
                                response = client.messages.create(
                                    model="claude-3-haiku-20240307",
                                    max_tokens=1000,
                                    messages=messages
                                )
                                
                                final_response = response.content[0].text
                        except Exception as e:
                            final_response = f"""I'm sorry, I couldn't connect to the Anthropic Claude API. Error: {str(e)}
                            
Please make sure you've added a valid API key in the Settings tab."""
                    
                    elif st.session_state.current_ai_model == "zephyr":
                        try:
                            context = ""
                            if any(keyword in user_input.lower() for keyword in ["drug", "ndc", "information", "ingredient"]):
                                words = user_input.split()
                                for word in words:
                                    if word.replace('-', '').isdigit() or len(word) > 4:
                                        try:
                                            drug_info = query_drug_info(word)
                                            context = f"Drug information for {word}: {json.dumps(drug_info)}\n\n"
                                            break
                                        except Exception:
                                            pass
                            
                            system_prompt = f"""You are PillQ Assistant, an AI helper specialized in pharmaceutical information.
                            {context}
                            Provide helpful, accurate information about drugs. Explain pharmaceutical concepts in clear language."""
                            
                            conversation = []
                            for message in st.session_state.chat_history[-5:]:
                                role_prefix = "User: " if message["role"] == "user" else "Assistant: "
                                conversation.append(f"{role_prefix}{message['content']}")
                            
                            conversation_history = "\n".join(conversation)
                            full_prompt = f"{system_prompt}\n\n{conversation_history}\nUser: {user_input}\nAssistant:"
                            
                            response = requests.post(
                                "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
                                headers={"Content-Type": "application/json"},
                                json={"inputs": full_prompt, "parameters": {"max_new_tokens": 500}}
                            )
                            
                            if response.status_code == 200:
                                generated_text = response.json()[0]["generated_text"]
                                if "Assistant:" in generated_text:
                                    final_response = generated_text.split("Assistant:")[-1].strip()
                                else:
                                    final_response = generated_text.strip()
                            else:
                                final_response = f"""I'm sorry, I couldn't connect to the HuggingFace service. 
                                
The free cloud model might be experiencing high traffic. Please try again in a moment or select a different model."""
                        except Exception as e:
                            final_response = f"""I'm sorry, I couldn't connect to the HuggingFace service. Error: {str(e)}
                            
The free cloud model might be experiencing high traffic. Please try again in a moment or select a different model."""
                        
                    else:
                        try:
                            context = ""
                            if any(keyword in user_input.lower() for keyword in ["drug", "ndc", "information", "ingredient"]):
                                words = user_input.split()
                                for word in words:
                                    if word.replace('-', '').isdigit() or len(word) > 4:
                                        try:
                                            drug_info = query_drug_info(word)
                                            context = f"Drug information for {word}: {json.dumps(drug_info)}\n\n"
                                            break
                                        except Exception:
                                            pass
                            
                            system_message = f"""You are PillQ Assistant, an AI helper specialized in pharmaceutical information.
                            {context}
                            Provide helpful, accurate information about drugs. Explain pharmaceutical concepts in clear language."""
                            
                            messages = []
                            messages.append({"role": "system", "content": system_message})
                            
                            for message in st.session_state.chat_history[-5:]:
                                messages.append({"role": message["role"], "content": message["content"]})
                            
                            response = requests.post("http://localhost:11434/api/chat", 
                                                   json={
                                                       "model": "llama3:latest", 
                                                       "messages": messages,
                                                       "stream": False
                                                   })
                            
                            if response.status_code == 200:
                                response_data = response.json()
                                final_response = response_data["message"]["content"]
                            else:
                                final_response = """I'm sorry, I couldn't connect to the Ollama service. 
                                
To use the free AI assistant, please:
1. Install Ollama from https://ollama.com/
2. Run this command in your terminal: `ollama run llama3`
3. Or configure an API key in the Settings tab."""
                        except Exception as e:
                            final_response = f"""I'm sorry, I couldn't connect to the Ollama service. Error: {str(e)}
                            
To use the free AI assistant, please:
1. Install Ollama from https://ollama.com/
2. Run this command in your terminal: `ollama run llama3`
3. Or configure an API key in the Settings tab."""
                        
                    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": f"I'm sorry, I encountered an error: {error_message}"})

# ==================== TAB 2: File Upload Mode ====================
with tab2:
    st.subheader("File Upload Mode")
    file_upload = st.file_uploader("Upload CSV/JSON/Excel/TXT (containing the NDC values)", type=["csv", "json", "xlsx", "txt"])
    
    if file_upload:
        try:
            if file_upload.name.endswith('.csv'):
                df = pd.read_csv(file_upload)
            elif file_upload.name.endswith('.json'):
                df = pd.read_json(file_upload)
            elif file_upload.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_upload)
            elif file_upload.name.endswith('.txt'):
                df = pd.read_csv(file_upload, header=None, names=["NDC"])
            else:
                st.error("Unsupported file format")
                st.stop()
            
            st.subheader("File Preview")
            st.dataframe(df.head(5))
            
            # Move export format selection before Configure Processing
            export_format = st.radio(
                "Output Format",
                ["CSV", "JSON", "Excel", "TXT"],  # Changed default to CSV by putting it first
                horizontal=True,
                key="format_file_upload"
            )
            
            st.subheader("Configure Processing")
            
            available_columns = df.columns.tolist()
            ndc_column = st.selectbox(
                "Select the column containing NDC codes",
                options=available_columns,
                index=available_columns.index("NDC") if "NDC" in available_columns else 0
            )
            
            keep_columns = st.multiselect(
                "Select columns to keep from your original file",
                options=available_columns,
                default=[ndc_column]
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
                "Select OpenFDA fields to retrieve",
                options=field_options,
                default=["active_ingredient", "inactive_ingredient"],
                key="fields_file_upload"
            )
            
            include_sources = st.checkbox("Include data sources", value=False, key="sources_file_upload")
            
            process_btn = st.button("Process File")
            
            if process_btn:
                if ndc_column not in keep_columns:
                    st.warning(f"The NDC column '{ndc_column}' has been automatically added to the output")
                    keep_columns = [ndc_column] + [col for col in keep_columns if col != ndc_column]
                
                with st.spinner(f"Processing {len(df)} NDCs..."):
                    result_df = df[keep_columns].copy()
                    
                    processed_data = []
                    for ndc in df[ndc_column]:
                        result = search_ndc(str(ndc), labels_to_get, include_source=include_sources)
                        processed_data.append(result)
                    
                    openfda_df = pd.DataFrame(processed_data)
                    
                    for col in openfda_df.columns:
                        if col != "NDC":  
                            result_df[col] = openfda_df[col].values
                
                st.subheader("Results")
                st.dataframe(result_df)
                
                st.download_button(
                    label=f"Download Results as {export_format}",
                    data=convert_df(result_df, export_format),
                    file_name=f"pillq_enriched_data.{export_format.lower()}",
                    mime="text/csv" if export_format == "CSV" else "application/octet-stream"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)

# ==================== TAB 3: Settings ====================
with tab3:
    st.subheader("AI Settings")
    st.markdown("""
    Configure your AI assistant settings including API keys for the different models.
    The model selection can be made directly in the AI Assistant section.
    """)
    
    st.subheader("Free Models with API Keys")
    
    st.markdown("### Google Gemini API")
    st.session_state.gemini_api_key = st.text_input(
        "Gemini API Key", 
        value=st.session_state.gemini_api_key if 'gemini_api_key' in st.session_state else "",
        type="password",
        key="gemini_key_input"
    )
    st.markdown("""
    Gemini 1.0 Pro is Google's powerful language model with good pharmaceutical knowledge.
    
    Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)
    """)
    
    st.markdown("### Cohere API")
    st.session_state.cohere_api_key = st.text_input(
        "Cohere API Key", 
        value=st.session_state.cohere_api_key if 'cohere_api_key' in st.session_state else "",
        type="password",
        key="cohere_key_input"
    )
    st.markdown("""
    Cohere's Command model provides excellent response quality with a generous free tier.
    
    Get a free API key at [Cohere Dashboard](https://dashboard.cohere.com/api-keys)
    """)
    
    st.markdown("### Anthropic Claude API")
    st.session_state.claude_api_key = st.text_input(
        "Claude API Key", 
        value=st.session_state.claude_api_key if 'claude_api_key' in st.session_state else "",
        type="password",
        key="claude_key_input"
    )
    st.markdown("""
    Claude 3 Haiku is Anthropic's fastest and most cost-effective model.
    
    Get a free API key at [Anthropic Console](https://console.anthropic.com/settings/keys)
    """)
    
    st.markdown("### OpenAI API")
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key", 
        value=st.session_state.openai_api_key if 'openai_api_key' in st.session_state else "",
        type="password",
        key="openai_key_input"
    )
    st.markdown("""
    GPT-3.5 Turbo offers good pharmaceutical knowledge (credit card required).
    
    Get an API key at [OpenAI Platform](https://platform.openai.com/api-keys)
    """)
    
    st.markdown("### DeepSeek API")
    st.session_state.deepseek_api_key = st.text_input(
        "DeepSeek API Key", 
        value=st.session_state.deepseek_api_key if 'deepseek_api_key' in st.session_state else "",
        type="password", 
        key="deepseek_key_input"
    )
    st.markdown("""
    DeepSeek offers powerful language models with good pharmaceutical knowledge.
    
    Get an API key at [DeepSeek Platform](https://platform.deepseek.com/)
    """)
    
    st.subheader("Completely Free Models (No API Key)")
    st.markdown("""
    **Claude Instant (Cloudflare, Free)**
    - Uses Claude Instant 1.2 hosted on Cloudflare Workers AI
    - No API key or installation required
    - Limited to 10,000 tokens per day
    
    **Zephyr 7B (HuggingFace, Free Cloud)**
    - Uses the Zephyr 7B Beta model hosted on HuggingFace's Inference API
    - No API key or installation required
    - May have rate limits during high traffic periods
    
    **Llama 3 (Ollama, Free Local)**
    - Requires downloading Ollama from [ollama.com](https://ollama.com/)
    - Run this command in your terminal: `ollama run llama3`
    - No API key required
    - Privacy-friendly (no data sent to external servers)
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")