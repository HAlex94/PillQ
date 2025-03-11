import streamlit as st
st.set_page_config(page_title="PillQ – Pill Queries, Simplified", layout="wide")

import pandas as pd
import io
import requests
import urllib.parse

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
st.title("PillQ – Pill Queries, Simplified")
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
                            # For each fallback code, store a row
                            for ndc_val in fallback_list:
                                results_list.append({
                                    "Brand Name": brand,
                                    "Generic Name": gen,
                                    "NDC": ndc_val,
                                    "Dosage Form": dform,
                                    "Route": route,
                                    "Strength": strength
                                })
                    else:
                        # If openfda product_ndc is found
                        for ndc_val in ndcs:
                            results_list.append({
                                "Brand Name": brand,
                                "Generic Name": gen,
                                "NDC": ndc_val,
                                "Dosage Form": dform,
                                "Route": route,
                                "Strength": strength
                            })
        else:
            st.warning(f"No matching products found in drugsfda for that name: {name_str}.")
    
    except Exception as e:
        st.error(f"Error fetching from drugsfda for name '{name_str}': {e}")
    
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
# Layout with two tabs
# ------------------------------------------------------------------
tab1, tab2 = st.tabs(["Single/Multiple Search", "File Upload"])

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

    # AI Assistant Section - Only show when requested
    st.markdown("---")
    show_assistant = st.checkbox("Show AI Assistant", value=False, key="show_assistant")
    if show_assistant:
        # Integrate AI chatbot here
        st.markdown("### PillQ AI Assistant")
        st.markdown("Ask me anything about drug information and I'll help you find answers.")
        
        user_question = st.text_area("Your question:", key="ai_question", height=100)
        
        if st.button("Ask AI Assistant"):
            if user_question:
                # Placeholder for AI response
                with st.spinner("Thinking..."):
                    # Replace with actual AI integration
                    time.sleep(2)
                    st.markdown(f"""
                    **Answer:** 
                    
                    I see you're asking about: "{user_question}"
                    
                    This is a placeholder for the AI assistant's response. 
                    In the real implementation, this would call the AI model.
                    """)
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