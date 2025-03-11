import streamlit as st
st.set_page_config(page_title="PillQ 💊 – Pill Queries, Simplified", layout="wide")

import pandas as pd
import io
import requests
import urllib.parse
import os
import base64
import json

from helper_functions import (
    get_openfda_searchable_fields,
    get_product_image,
    get_product_ndc,
    get_label_field,
    get_combined_label_field,
    get_setid_from_search,
    fallback_ndc_search,
    unify_source_string,
    convert_df
)

# ------------------------------------------------------------------
# Title & Description
# ------------------------------------------------------------------
st.title("PillQ 💊 – Pill Queries, Simplified")
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
                    
                    # active_ingredients => parse name + strength
                    ai_list = prod.get("active_ingredients", [])
                    if ai_list:
                        pieces = []
                        for ai in ai_list:
                            nm = ai.get("name", "")
                            stg = ai.get("strength", "")
                            if nm or stg:
                                pieces.append(f"{nm} {stg}".strip())
                        strength = " | ".join(pieces) if pieces else "Not Available"
                    else:
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
                        # Use your fallback function from the example above
                        fallback_list = fallback_ndc_search(fallback_name)
                        
                        if not fallback_list:
                            # No fallback NDC found
                            results_list.append({
                                "Brand Name": brand,
                                "Generic Name": gen,
                                "NDC": "Not Available",
                                "Dosage Form": dform,
                                "Route": route,
                                "Strength": strength
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
                                            "Brand Name": brand,
                                            "Generic Name": gen,
                                            "NDC": package_ndc,
                                            "Dosage Form": dform,
                                            "Route": route,
                                            "Strength": strength
                                        })
                                else:
                                    # If no package NDCs found, use the product NDC
                                    results_list.append({
                                        "Brand Name": brand,
                                        "Generic Name": gen,
                                        "NDC": ndc_val,
                                        "Dosage Form": dform,
                                        "Route": route,
                                        "Strength": strength
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
                                        "Brand Name": brand,
                                        "Generic Name": gen,
                                        "NDC": package_ndc,
                                        "Dosage Form": dform,
                                        "Route": route,
                                        "Strength": strength
                                    })
                            else:
                                # If no package NDCs found, use the product NDC
                                results_list.append({
                                    "Brand Name": brand,
                                    "Generic Name": gen,
                                    "NDC": ndc_val,
                                    "Dosage Form": dform,
                                    "Route": route,
                                    "Strength": strength
                                })
    except Exception as e:
        st.error(f"Error retrieving data from openFDA drugsfda endpoint: {e}")
    
    # Sort by the Brand Name column
    if results_list:
        results_list = sorted(results_list, key=lambda x: x["Brand Name"])
    return results_list

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
# Layout with three tabs
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Single/Multiple Search", "File Upload", "Settings"])

# Initialize session state for AI assistant and settings
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'provider' not in st.session_state:
    st.session_state.provider = "HuggingFace"  # Default to free option
if 'model_name' not in st.session_state:
    st.session_state.model_name = "google/flan-t5-small"  # Default to a free model

def get_ai_response(prompt, provider, api_key=None, model_name=None):
    """Get response from the selected AI model."""
    try:
        # Add the prompt to the history
        history = st.session_state.chat_history
        history.append({"role": "user", "content": prompt})
        
        if provider == "OpenAI":
            # OpenAI API
            import openai
            openai.api_key = api_key
            
            messages = []
            for message in history:
                messages.append({"role": message["role"], "content": message["content"]})
            
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages
            )
            answer = response.choices[0].message.content
        
        elif provider == "Ollama":
            # Ollama API
            ollama_url = "http://localhost:11434/api/chat"
            
            headers = {"Content-Type": "application/json"}
            messages = []
            for message in history:
                messages.append({"role": message["role"], "content": message["content"]})
            
            data = {
                "model": model_name,
                "messages": messages
            }
            
            response = requests.post(ollama_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            answer = response.json()['message']['content']
        
        elif provider == "DeepSeek":
            # DeepSeek API
            deepseek_url = "https://api.deepseek.com/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            messages = []
            for message in history:
                messages.append({"role": message["role"], "content": message["content"]})
            
            data = {
                "model": model_name,
                "messages": messages
            }
            
            response = requests.post(deepseek_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            answer = response.json()['choices'][0]['message']['content']
        
        elif provider == "HuggingFace":
            # Use HuggingFace Transformers (no API key required for many models)
            try:
                from transformers import pipeline
                
                # Initialize the pipeline with the specified model
                generator = pipeline("text-generation", model=model_name)
                
                # Combine history into a single string for context
                context = ""
                for message in history:
                    prefix = "User: " if message["role"] == "user" else "Assistant: "
                    context += prefix + message["content"] + "\n"
                
                # Generate response
                response = generator(context + "Assistant: ", max_length=200, do_sample=True)
                
                # Extract answer from the generated text
                full_response = response[0]['generated_text']
                answer = full_response.split("Assistant: ")[-1].strip()
                
            except Exception as e:
                answer = f"Error using HuggingFace model: {str(e)}"
        
        else:
            answer = "Selected AI provider not implemented."
        
        # Add the answer to history
        history.append({"role": "assistant", "content": answer})
        
        return answer
    
    except Exception as e:
        return f"Error: {str(e)}"

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
                            df = pd.DataFrame(name_results)
                            
                            # Use Streamlit's data_editor to display the table
                            edited_df = st.data_editor(
                                df,
                                column_config={
                                    "Brand Name": st.column_config.TextColumn("Brand Name"),
                                    "Generic Name": st.column_config.TextColumn("Generic Name"),
                                    "NDC": st.column_config.TextColumn("NDC"),
                                    "Dosage Form": st.column_config.TextColumn("Dosage Form"),
                                    "Route": st.column_config.TextColumn("Route"),
                                    "Strength": st.column_config.TextColumn("Strength"),
                                },
                                hide_index=True,
                                use_container_width=True,
                                key="data_editor_name_results"
                            )
                            
                            # Multiselect for exporting specific rows
                            if not edited_df.empty:
                                st.markdown("### Export Options")
                                
                                export_format = st.radio("Export Format", ["CSV", "Excel", "JSON"], horizontal=True)
                                
                                # Allow selection of specific NDCs for export
                                selected_ndcs = st.multiselect(
                                    "Select NDCs for Detailed Export",
                                    options=edited_df["NDC"].tolist(),
                                    help="Select one or more NDCs to view detailed information"
                                )
                                
                                # Export button for the entire table
                                export_table = edited_df
                                
                                if st.button("Export Table Results"):
                                    if export_format == "CSV":
                                        csv = convert_df(export_table, "CSV")
                                        st.download_button(
                                            label="Download CSV",
                                            data=csv,
                                            file_name=f"{item}_results.csv",
                                            mime="text/csv"
                                        )
                                    elif export_format == "Excel":
                                        excel = convert_df(export_table, "Excel")
                                        st.download_button(
                                            label="Download Excel",
                                            data=excel,
                                            file_name=f"{item}_results.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                    else:  # JSON
                                        json_data = convert_df(export_table, "JSON")
                                        st.download_button(
                                            label="Download JSON",
                                            data=json_data,
                                            file_name=f"{item}_results.json",
                                            mime="application/json"
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

    # AI Assistant Section - Only show when requested
    st.markdown("---")
    show_assistant = st.checkbox("Show AI Assistant", value=False, key="show_assistant")
    if show_assistant:
        st.subheader("AI Assistant")
        st.markdown(f"Using **{st.session_state.provider}** with model **{st.session_state.model_name}**")
        st.markdown("Ask me questions about drugs, NDCs, or how to use this app.")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
        
        # Input for user question
        user_question = st.text_area("Your question:", key="user_question")
        
        # Process button
        if st.button("Ask"):
            if user_question:
                # Placeholder for AI response
                with st.spinner("Thinking..."):
                    ai_response = get_ai_response(user_question, st.session_state.provider, st.session_state.api_key, st.session_state.model_name)
                    st.session_state.ai_response = ai_response
                    st.markdown(f"**Answer:** {ai_response}")
                    st.experimental_rerun()  # Refresh to show the new message in chat history
            else:
                st.warning("Please enter a question first.")

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
                                    # Get detailed data for this NDC
                                    detailed_data = search_ndc(ndc_str, openfda_fields, include_sources_upload)
                                    
                                    # Add each selected field to the output dataframe
                                    for field in openfda_fields:
                                        field_col_name = f"{col}_{field}"
                                        output_df.at[i, field_col_name] = detailed_data.get(field, "Not Available")
                                        
                                        if include_sources_upload and field + "_source" in detailed_data:
                                            source_col_name = f"{col}_{field}_source"
                                            output_df.at[i, source_col_name] = detailed_data[field + "_source"]
                                except Exception as e:
                                    st.warning(f"Could not fetch detailed data for NDC {ndc_str}: {e}")
                        
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
    
    # Provider selection
    st.markdown("##### Select AI Provider")
    provider_options = ["HuggingFace", "OpenAI", "Ollama", "DeepSeek"]
    selected_provider = st.selectbox(
        "Choose AI Provider",
        options=provider_options,
        index=provider_options.index(st.session_state.provider) if st.session_state.provider in provider_options else 0
    )
    
    # Update provider in session state if changed
    if selected_provider != st.session_state.provider:
        st.session_state.provider = selected_provider
    
    # Display provider-specific settings
    st.markdown("---")
    
    # OpenAI configuration
    if selected_provider == "OpenAI":
        st.markdown("##### OpenAI API Key")
        api_key = st.text_input("Enter OpenAI API Key", value=st.session_state.api_key, type="password")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        # Model selection
        st.markdown("##### Model Selection")
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        model_name = st.selectbox(
            "Choose OpenAI Model",
            options=model_options,
            index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0
        )
        
        if model_name != st.session_state.model_name:
            st.session_state.model_name = model_name
    
    # Ollama configuration
    elif selected_provider == "Ollama":
        st.markdown("##### Ollama Model")
        st.markdown("Ollama runs locally on your machine. Make sure the Ollama server is running.")
        model_options = ["llama2", "mistral", "codellama", "vicuna", "orca-mini"]
        model_name = st.selectbox(
            "Choose Ollama Model",
            options=model_options,
            index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0
        )
        
        if model_name != st.session_state.model_name:
            st.session_state.model_name = model_name
    
    # DeepSeek configuration
    elif selected_provider == "DeepSeek":
        st.markdown("##### DeepSeek API Key")
        api_key = st.text_input("Enter DeepSeek API Key", value=st.session_state.api_key, type="password")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        # Model selection
        st.markdown("##### Model Selection")
        model_options = ["deepseek-chat", "deepseek-coder"]
        model_name = st.selectbox(
            "Choose DeepSeek Model",
            options=model_options,
            index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0
        )
        
        if model_name != st.session_state.model_name:
            st.session_state.model_name = model_name
    
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
        
        st.markdown("##### Free Model Suggestions")
        st.markdown("These models don't require an API key and can run with minimal resources:")
        free_models = [
            "google/flan-t5-small",
            "facebook/bart-large-cnn", 
            "google/t5-small",
            "distilbert/distilbert-base-uncased"
        ]
        for model in free_models:
            if st.button(model, key=f"model_button_{model}"):
                st.session_state.model_name = model
                st.experimental_rerun()
    
    # Test connection button
    st.markdown("---")
    if st.button("Test Connection"):
        try:
            with st.spinner("Testing connection..."):
                # Test prompt
                test_prompt = "Hello, this is a test message. Please respond with 'Connection successful!'"
                
                # Get response
                test_response = get_ai_response(
                    test_prompt, 
                    st.session_state.provider, 
                    st.session_state.api_key, 
                    st.session_state.model_name
                )
                
                st.success("Connection successful! Your AI assistant is ready to use.")
        except Exception as e:
            st.error(f"Connection failed: {e}")
            
    # Reset chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

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
    pass  # Silently ignore if illustration can't be loaded