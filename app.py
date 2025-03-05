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
    unify_source_string
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
            val = data_dict[src_key].lower()
            if "openfda" in val:
                sources_used.add("openFDA")
            if "dailymed" in val:
                sources_used.add("DailyMed")
    if not sources_used:
        return "None"
    elif len(sources_used) == 1:
        return next(iter(sources_used))
    else:
        return "openFDA + DailyMed"

def convert_df(df, fmt):
    if fmt == "CSV":
        return df.to_csv(index=False).encode("utf-8")
    elif fmt == "Excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        return output.getvalue()
    elif fmt == "TXT":
        return df.to_csv(sep="\t", index=False).encode("utf-8")
    elif fmt == "JSON":
        return df.to_json(orient="records").encode("utf-8")

# ------------------------------------------------------------------
# Layout with two tabs
# ------------------------------------------------------------------
tab1, tab2 = st.tabs(["Single/Multiple Search", "File Upload"])

# ==================== TAB 1: Single/Multiple Search ====================
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Single/Multiple Search")
        search_mode = st.radio("Search By:", ["NDC", "Name"], horizontal=True)
        multi_input = st.text_input("Enter comma-separated items (NDC or Name)")
        output_format = st.radio("Output Format", ["JSON", "CSV", "Excel", "TXT"], horizontal=True)
        include_source = st.checkbox("Include Source Info in Output", value=False)
        show_header_source = st.checkbox("Show Source in Single Item Header", value=True)

        # Retrieve openFDA fields & ensure brand/generic
        openfda_fields = sorted(list(get_openfda_searchable_fields()))
        if not openfda_fields:
            openfda_fields = ["active_ingredient", "inactive_ingredient", "indications_and_usage"]
        for needed in ["brand_name", "generic_name"]:
            if needed not in openfda_fields:
                openfda_fields.append(needed)
        openfda_fields = sorted(openfda_fields)

        selected_labels = st.multiselect(
            "Select Data Fields to Retrieve",
            options=openfda_fields,
            default=["brand_name", "generic_name"]
        )
        preview_btn = st.button("Preview Data")

    with col_right:
        if preview_btn and multi_input.strip():
            items = [x.strip() for x in multi_input.split(",") if x.strip()]

            # -------------------------------------------------
            # Single Item
            # -------------------------------------------------
            if len(items) == 1:
                single_item = items[0]
                if search_mode == "Name":
                    # Name-based search for a single brand/generic
                    ndc_matches = fetch_ndcs_for_name_drugsfda(single_item, limit=50)
                    if ndc_matches:
                        st.write("### Found Matching NDCs:")
                        df_matches = pd.DataFrame(ndc_matches)
                        st.dataframe(df_matches)
                    else:
                        st.warning(f"No matching products found in drugsfda for '{single_item}' by Name search.")
                else:
                    # Single NDC scenario
                    single_data = search_ndc(single_item, selected_labels, include_source=include_source)
                    final_source = get_single_item_source(single_data, selected_labels, include_source) if show_header_source else "None"

                    # Fallback if the detection logic yields "None"
                    if final_source == "None":
                        final_source = "openFDA"

                    if show_header_source:
                        st.markdown(
                            f"<h3>Processing Single NDC <span style='font-size:14px;'>(Source: {final_source})</span></h3>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown("<h3>Processing Single NDC</h3>", unsafe_allow_html=True)

                    # Optionally display product image
                    img_url = get_product_image(single_item)
                    if img_url:
                        st.image(img_url, caption="Product Label", width=400)
                    else:
                        st.info("No product image available for this NDC.")

                    # Display single_data in chosen format
                    if output_format == "JSON":
                        st.json(single_data)
                    elif output_format in ["CSV", "Excel"]:
                        st.dataframe(pd.DataFrame([single_data]))
                    elif output_format == "TXT":
                        st.text(pd.DataFrame([single_data]).to_csv(sep="\t", index=False))

            # -------------------------------------------------
            # Multiple Items
            # -------------------------------------------------
            else:
                df_result = pd.DataFrame()
                for itm in items:
                    if search_mode == "Name":
                        # Minimal approach if user typed multiple brand/generic names
                        df_result = df_result.append({"Name": itm, "Note": "Placeholder multiple name logic."}, ignore_index=True)
                    else:
                        row_data = search_ndc(itm, selected_labels, include_source=include_source)
                        df_result = df_result.append(row_data, ignore_index=True)
                if not df_result.empty:
                    if output_format == "JSON":
                        st.json(df_result.to_dict(orient="records"))
                    elif output_format in ["CSV", "Excel"]:
                        st.dataframe(df_result)
                    elif output_format == "TXT":
                        st.text(df_result.to_csv(sep="\t", index=False))
                    conv_data = convert_df(df_result, output_format)
                    st.download_button(
                        label="Download Output",
                        data=conv_data,
                        file_name=f"pillq_output.{output_format.lower()}",
                        mime="text/csv" if output_format=="CSV" else
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if output_format=="Excel" else
                             "text/plain" if output_format=="TXT" else
                             "application/json"
                    )
        else:
            st.info("Enter comma-separated items and click 'Preview Data'.")

# ==================== TAB 2: File Upload Mode ====================
with tab2:
    st.subheader("File Upload Mode")
    file_upload = st.file_uploader("Upload CSV (with 'NDC' column)", type=["csv"])
    if file_upload:
        df_input = pd.read_csv(file_upload)
        st.write("### Uploaded Data Preview", df_input.head())
        if "NDC" not in df_input.columns:
            st.error("The uploaded CSV must contain an 'NDC' column.")
        else:
            # Let user pick data fields
            openfda_fields_2 = sorted(list(get_openfda_searchable_fields()))
            if not openfda_fields_2:
                openfda_fields_2 = ["active_ingredient", "inactive_ingredient", "indications_and_usage"]
            for needed in ["brand_name", "generic_name"]:
                if needed not in openfda_fields_2:
                    openfda_fields_2.append(needed)
            openfda_fields_2 = sorted(openfda_fields_2)
            
            selected_labels_2 = st.multiselect("Data Fields for File Processing", options=openfda_fields_2, default=["brand_name", "generic_name"])
            include_source_2 = st.checkbox("Include Source Info in Output (File)", value=True)
            out_fmt = st.radio("Output Format (File)", ["JSON", "CSV", "Excel", "TXT"], horizontal=True)
            
            if st.button("Process File"):
                processed_rows = []
                for idx, row in df_input.iterrows():
                    ndc_val = str(row["NDC"]).strip()
                    row_dict = row.to_dict()
                    computed = search_ndc(ndc_val, selected_labels_2, include_source=include_source_2)
                    row_dict.update(computed)
                    processed_rows.append(row_dict)
                df_output = pd.DataFrame(processed_rows)
                
                input_cols = list(df_input.columns)
                computed_cols = selected_labels_2[:]
                if include_source_2:
                    computed_cols += [c + "_source" for c in selected_labels_2]
                out_cols = list(dict.fromkeys(input_cols + computed_cols))
                chosen_cols = st.multiselect("Select Output Columns", out_cols, default=out_cols)
                df_output = df_output[chosen_cols]
                
                st.write("### Processed Output Preview")
                if out_fmt == "JSON":
                    st.json(df_output.to_dict(orient="records"))
                elif out_fmt in ["CSV", "Excel"]:
                    st.dataframe(df_output)
                elif out_fmt == "TXT":
                    st.text(df_output.to_csv(sep="\t", index=False))
                
                conv_data = convert_df(df_output, out_fmt)
                st.download_button(
                    label="Download Output",
                    data=conv_data,
                    file_name=f"pillq_file_output.{out_fmt.lower()}",
                    mime="text/csv" if out_fmt == "CSV" else
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if out_fmt == "Excel" else
                         "text/plain" if out_fmt == "TXT" else
                         "application/json"
                )
    else:
        st.info("Upload a CSV file to process in batch mode.")
