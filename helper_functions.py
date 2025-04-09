import requests
import re
import io
import pandas as pd
import urllib.parse
import difflib
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import os
import json
import urllib.error
import bleach
import logging

def get_openfda_searchable_fields():
    """
    Dynamically retrieve a set of searchable fields from openFDA's documentation page.
    Returns a set of field names (e.g. {"active_ingredient", "inactive_ingredient", "indications_and_usage", ...}).
    """
    url = "https://open.fda.gov/apis/drug/label/searchable-fields/"
    fields = set()
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Look for <button> tags with class "field-name"
        buttons = soup.find_all("button", class_="field-name")
        for btn in buttons:
            title = btn.get("title")
            if title:
                fields.add(title.strip())
        # Also check <code> tags as a fallback
        codes = soup.find_all("code")
        for code in codes:
            txt = code.get_text(strip=True)
            if txt:
                fields.add(txt)
    except Exception as e:
        print(f"Error fetching openFDA searchable fields: {e}")
    return fields

def fuzzy_match(user_field, available_fields):
    """
    Use difflib to find the best match for user_field from available_fields.
    Returns the best matching field (preserving original case) or None.
    """
    user_field_lower = user_field.lower()
    for field in available_fields:
        if field.lower() == user_field_lower:
            return field
    matches = difflib.get_close_matches(user_field_lower, [f.lower() for f in available_fields], n=1, cutoff=0.3)
    if matches:
        for field in available_fields:
            if field.lower() == matches[0]:
                return field
    return None

def get_product_ndc(ndc):
    """
    Convert a package-level NDC (e.g., "29485-5043-6") into a product-level NDC ("29485-5043").
    """
    parts = ndc.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return ndc

def get_field_from_openfda(ndc, field_key):
    """
    Query the openFDA Drug Labeling API using the product-level NDC.
    Checks both the top-level and the 'openfda' sub-object for the given field_key.
    Returns the field data (string or list) or None.
    """
    base_url = "https://api.fda.gov/drug/label.json"
    query = f'openfda.product_ndc.exact:"{ndc}"'
    params = {"search": query, "limit": 1}
    print(f"Querying openFDA for NDC={ndc}, field={field_key}")
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "results" in data and data["results"]:
            result = data["results"][0]
            if field_key in result and result[field_key]:
                return result[field_key]
            ofda = result.get("openfda", {})
            if field_key in ofda and ofda[field_key]:
                return ofda[field_key]
        print("openFDA returned no data for this field.")
        return None
    except Exception as e:
        print(f"Error in openFDA query: {e}")
        return None

def get_setid_from_search(ndc):
    """
    Given a product-level NDC, query DailyMed's search page and extract the SPL Set ID.
    Uses the URL:
      https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={ndc}&type=search
    Returns the first found setid or None.
    """
    search_url = f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={ndc}&type=search"
    print(f"Searching DailyMed for NDC {ndc} at {search_url}")
    try:
        resp = requests.get(search_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        link = soup.find("a", href=lambda h: h and "drugInfo.cfm?setid=" in h)
        if link and link.has_attr("href"):
            href = link["href"]
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            setid = qs.get("setid", [None])[0]
            if setid:
                print(f"Found Set ID: {setid}")
                return setid
        print("No Set ID found in DailyMed search results.")
        return None
    except Exception as e:
        print(f"Error searching DailyMed: {e}")
        return None

def scrape_label_field(setid, possible_headings):
    """
    Given a DailyMed SPL Set ID and a list of possible headings (case-insensitive),
    scrape the corresponding drug info page for that field and extract its contents.
    Returns a list of text items or ["Not Available"] if not found.
    """
    url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
    print(f"Scraping DailyMed page: {url}")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        headings = soup.find_all("td", class_="formHeadingTitle")
        target_td = None
        for td in headings:
            heading_text = td.get_text(strip=True).lower()
            for candidate in possible_headings:
                if candidate.lower() in heading_text:
                    target_td = td
                    break
            if target_td:
                break
        if target_td:
            table = target_td.find_parent("table")
            if not table:
                return ["Not Available"]
            rows = table.find_all("tr")[1:]  # Skip header row
            items = []
            for row in rows:
                cols = row.find_all("td")
                if not cols:
                    continue
                raw_text = cols[0].get_text(strip=True)
                cleaned = re.sub(r"\s*\(.*?\)", "", raw_text).strip()
                items.append(cleaned)
            return items if items else ["Not Available"]
        else:
            # Fallback: search in div.preview-text (for fields that may not be in a table)
            preview_divs = soup.find_all("div", class_="preview-text")
            for div in preview_divs:
                text = div.get_text(strip=True)
                for candidate in possible_headings:
                    if candidate.lower() in text.lower():
                        return [text]
            return ["Not Available"]
    except Exception as e:
        print(f"Error scraping field from DailyMed: {e}")
        return ["Not Available"]

def get_label_field(original_ndc, openfda_field, daily_med_headings):
    """
    Retrieve a drug label field for a given NDC:
      1. Try openFDA using the product-level NDC (converted from original_ndc).
      2. If openFDA returns no data, fall back to DailyMed web scraping.
    Returns a tuple: (data, source) where source is "openFDA" or "DailyMed Web Scraper".
    """
    product_ndc = get_product_ndc(original_ndc)
    print(f"Using product-level NDC for openFDA query: {product_ndc}")
    openfda_data = get_field_from_openfda(product_ndc, openfda_field)
    if openfda_data:
        if isinstance(openfda_data, str):
            openfda_data = [openfda_data]
        elif isinstance(openfda_data, dict):
            openfda_data = [str(openfda_data)]
        return openfda_data, "openFDA"
    
    print("Falling back to DailyMed scraping...")
    setid_value = get_setid_from_search(product_ndc)
    if setid_value:
        scraped_data = scrape_label_field(setid_value, daily_med_headings)
        return scraped_data, "DailyMed Web Scraper"
    return ["Not Available"], "None"

def get_combined_label_field(original_ndc, openfda_field, daily_med_headings):
    """
    Retrieve a drug label field from both openFDA and DailyMed,
    and combine the results into a single list of unique values.
    """
    product_ndc = get_product_ndc(original_ndc)
    print(f"Using product-level NDC for combined query: {product_ndc}")
    openfda_data = get_field_from_openfda(product_ndc, openfda_field)
    if openfda_data:
        if isinstance(openfda_data, str):
            openfda_data = [openfda_data]
        elif isinstance(openfda_data, dict):
            openfda_data = [str(openfda_data)]
    else:
        openfda_data = []
    
    setid_value = get_setid_from_search(product_ndc)
    dailymed_data = []
    if setid_value:
        dailymed_data = scrape_label_field(setid_value, daily_med_headings)
    
    combined = []
    for item in openfda_data + dailymed_data:
        if item not in combined:
            combined.append(item)
    if not combined:
        combined = ["Not Available"]
    return combined

# --- Processing Functions ---
def process_ndc(ndc_val, desired_labels):
    """
    Process a single NDC and retrieve requested fields.
    Returns a dictionary with keys: NDC and one key per desired label.
    """
    result = {"NDC": ndc_val}
    for label in desired_labels:
        if label in ["active_ingredient", "inactive_ingredient"]:
            data = get_combined_label_field(
                ndc_val,
                label,
                ["active ingredient", "active ingredient/active moiety", "active moiety"] if label=="active_ingredient" 
                else ["inactive ingredient", "inactive ingredients"]
            )
            result[label] = ", ".join(data)
        else:
            data, src = get_label_field(
                ndc_val,
                label,
                ["indications and usage", "uses", "indications"] if label=="indications_and_usage" else [label.replace("_", " ")]
            )
            result[label] = ", ".join(data)
    return result

def convert_df(df, fmt):
    """
    Convert DataFrame to the selected output format.
    """
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
    
def get_product_image(ndc):
    """
    Attempt to retrieve a product label image URL for the given NDC.
    
    1. First, query openFDA using the product-level NDC. If the JSON result includes an "image" key, return its URL.
       (Note: openFDA may not provide images; this is hypothetical.)
    2. If no image is found from openFDA, fall back to scraping the DailyMed drug info page:
         - Retrieve the SPL Set ID for the product-level NDC.
         - Scrape the page for an <img> tag whose alt text includes "label" (case-insensitive).
         - Return the full image URL if found.
    3. If no image is found, return None.
    """
    # Use the product-level NDC for queries.
    from bs4 import BeautifulSoup
    import requests
    from helper_functions import get_product_ndc, get_setid_from_search  # if needed, adjust for your module structure

    product_ndc = get_product_ndc(ndc)
    
    # --- Attempt openFDA first ---
    base_url = "https://api.fda.gov/drug/label.json"
    query = f'openfda.product_ndc.exact:"{product_ndc}"'
    params = {"search": query, "limit": 1}
    print(f"Querying openFDA for NDC={ndc}, image")
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "results" in data and data["results"]:
            result = data["results"][0]
            if "image" in result and result["image"]:
                return result["image"]
        # If no image found, fall through.
    except Exception as e:
        print(f"Error retrieving product image from openFDA: {e}")
    
    # --- Fallback: Scrape DailyMed ---
    setid = get_setid_from_search(product_ndc)
    if setid:
        dm_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
        try:
            resp = requests.get(dm_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Heuristically, look for an <img> with alt text containing "label"
            img_tag = soup.find("img", alt=lambda x: x and "label" in x.lower())
            if img_tag and img_tag.get("src"):
                src = img_tag["src"]
                # If the src is relative, prepend DailyMed's domain.
                if not src.startswith("http"):
                    src = "https://dailymed.nlm.nih.gov" + src
                return src
        except Exception as e:
            print(f"Error retrieving product image from DailyMed: {e}")
    return None
    
# Function to fetch and resize the image
def get_resized_image(img_url, height=300):
    from PIL import Image
    response = requests.get(img_url, stream=True)
    if response.status_code == 200:
        img = Image.open(response.raw)
        aspect_ratio = img.width / img.height  # Maintain aspect ratio
        new_width = int(height * aspect_ratio)  # Adjust width proportionally
        img = img.resize((new_width, height))  # Resize image
        return img
    return None

def fallback_ndc_search(drug_name, limit=10):
    base_url = "https://api.fda.gov/drug/ndc.json"
    uppercase_term = drug_name.upper()
    # brand_name:"TERM" or generic_name:"TERM"
    raw_expr = f'(brand_name:"{uppercase_term}" OR generic_name:"{uppercase_term}")'
    safe_chars = '()+:"'
    encoded_expr = urllib.parse.quote(raw_expr, safe=safe_chars)

    params = {"search": encoded_expr, "limit": limit}
    ndc_list = []
    try:
        r = requests.get(base_url, params=params, timeout=10)
        r.raise_for_status()
        jdata = r.json()
        
        if "results" in jdata and jdata["results"]:
            for rec in jdata["results"]:
                # 'product_ndc' is often the 10-digit code, e.g. "0077-3105"
                # 'packaging' might have "package_ndc" e.g. "0077-3105-02"
                pndc = rec.get("product_ndc", "")
                if pndc:
                    ndc_list.append(pndc)
                # parse packaging array for "package_ndc" if needed
                # ...
    except Exception as e:
        print(f"fallback_ndc_search error: {e}")
    return list(set(ndc_list))  # deduplicate

def unify_source_string(src):
    """
    Takes the raw 'src' from get_label_field (e.g. "DailyMed Web Scraper",
    "openFDA fetch", or "None") and returns a string that definitely contains
    "openfda" or "dailymed" if relevant, so your get_single_item_source code
    can detect them.
    """
    if not src:
        # If src is empty or None, default to "openfda"
        return "openfda"
    
    s = src.lower()
    used = set()
    
    # If src has "openfda" or "open fda" or something close:
    if "openfda" in s or "open fda" in s:
        used.add("openfda")
    
    # If src has "dailymed" or "scraper"
    if "dailymed" in s or "scraper" in s:
        used.add("dailymed")
    
    # If we found nothing, fallback to "openfda"
    if not used:
        used.add("openfda")
    
    if len(used) == 2:
        return "openfda + dailymed"
    else:
        return next(iter(used))  # either "openfda" or "dailymed"

def fetch_ndcs_for_name_drugsfda(name_str, limit=50):
    """
    Query the OpenFDA drug API for products matching a brand or generic name.
    Returns a list of dictionaries with NDC, brand name, generic name, dose form, etc.
    
    Args:
        name_str (str): Brand or generic drug name to search for
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of dictionaries with NDC info
    """
    results = []
    
    # Get API key from environment variable if available
    api_key = os.environ.get('OPENFDA_API_KEY', '')
    api_key_param = f"&api_key={api_key}" if api_key else ""
    
    # Encode the name for URL
    encoded_name = urllib.parse.quote(name_str)
    
    # First try brand name search
    brand_url = f"https://api.fda.gov/drug/ndc.json?search=brand_name:{encoded_name}&limit={limit}{api_key_param}"
    
    try:
        brand_response = requests.get(brand_url)
        brand_response.raise_for_status()
        brand_data = brand_response.json()
        
        if "results" in brand_data:
            for result in brand_data["results"]:
                if "product_ndc" in result and "packaging" in result:
                    for package in result["packaging"]:
                        if "package_ndc" in package:
                            # Extract and clean strength information
                            strength = ""
                            if "active_ingredients" in result and result["active_ingredients"]:
                                # Check if this is a multi-ingredient product
                                if len(result["active_ingredients"]) > 1:
                                    # For multi-ingredient products like Adderall, calculate the total strength
                                    total_strength = 0
                                    strength_unit = ""
                                    
                                    # Sum up the strength values from all ingredients
                                    for ing in result["active_ingredients"]:
                                        strength_str = ing.get("strength", "")
                                        # Get the numeric part of the strength
                                        if " mg" in strength_str:
                                            try:
                                                # Extract number and unit
                                                num_str = strength_str.split(" mg")[0].replace("/1", "")
                                                total_strength += float(num_str)
                                                strength_unit = " mg"
                                            except:
                                                pass
                                        elif " mcg" in strength_str:
                                            try:
                                                num_str = strength_str.split(" mcg")[0].replace("/1", "")
                                                total_strength += float(num_str)
                                                strength_unit = " mcg"
                                            except:
                                                pass
                                    
                                    # Format the total strength
                                    if total_strength > 0:
                                        if total_strength.is_integer():
                                            strength = f"{int(total_strength)}{strength_unit}"
                                        else:
                                            strength = f"{total_strength}{strength_unit}"
                                    else:
                                        # Fallback to listing all strengths
                                        strength = ", ".join([ing.get("strength", "").replace("/1", "") for ing in result["active_ingredients"]])
                                else:
                                    # Single ingredient case - use the first ingredient's strength
                                    try:
                                        strength_str = result["active_ingredients"][0].get("strength", "")
                                        # Remove "/1" if present
                                        strength_str = strength_str.replace("/1", "")
                                        # Try to convert to integer if it's a whole number
                                        if " mg" in strength_str:
                                            strength_num = strength_str.split(" mg")[0]
                                            try:
                                                if float(strength_num).is_integer():
                                                    strength_str = f"{int(float(strength_num))} mg"
                                                else:
                                                    strength_str = f"{float(strength_num)} mg"
                                            except:
                                                pass
                                        strength = strength_str
                                    except:
                                        strength = ", ".join([ing.get("strength", "").replace("/1", "") for ing in result.get("active_ingredients", [])])
                            
                            # Clean up package description - remove any NDC references more thoroughly
                            package_desc = package.get("description", "")
                            
                            # Remove pattern like " (NDC)" or " (NDC: 12345-678-90)" or "(57844-110-01)" at the end
                            import re
                            # First pattern: remove " (NDC)" or similar simple patterns
                            package_desc = re.sub(r'\s*\(\s*NDC\s*\)', '', package_desc, flags=re.IGNORECASE)
                            
                            # Second pattern: remove NDC with the actual number in parentheses
                            package_desc = re.sub(r'\s*\(\s*NDC\s*:?\s*[\d\-]+\s*\)', '', package_desc, flags=re.IGNORECASE)
                            
                            # Third pattern: just a number in parentheses at the end that looks like an NDC
                            package_desc = re.sub(r'\s*\(\s*[\d\-]+\s*\)\s*$', '', package_desc)
                            
                            # Extract the relevant information
                            ndc_info = {
                                "NDC": package.get("package_ndc", ""),
                                "brand_name": result.get("brand_name", ""),
                                "generic_name": result.get("generic_name", ""),
                                "strength": strength,
                                "route": result.get("route", ""),
                                "dosage_form": result.get("dosage_form", ""),
                                "manufacturer": result.get("labeler_name", ""),
                                "package_description": package_desc.strip()
                            }
                            results.append(ndc_info)
    except Exception as e:
        print(f"Error fetching brand name data: {e}")
    
    # If no results from brand name, try generic name search
    if not results:
        generic_url = f"https://api.fda.gov/drug/ndc.json?search=generic_name:{encoded_name}&limit={limit}{api_key_param}"
        
        try:
            generic_response = requests.get(generic_url)
            generic_response.raise_for_status()
            generic_data = generic_response.json()
            
            if "results" in generic_data:
                for result in generic_data["results"]:
                    if "product_ndc" in result and "packaging" in result:
                        for package in result["packaging"]:
                            if "package_ndc" in package:
                                # Extract and clean strength information
                                strength = ""
                                if "active_ingredients" in result and result["active_ingredients"]:
                                    # Check if this is a multi-ingredient product
                                    if len(result["active_ingredients"]) > 1:
                                        # For multi-ingredient products like Adderall, calculate the total strength
                                        total_strength = 0
                                        strength_unit = ""
                                        
                                        # Sum up the strength values from all ingredients
                                        for ing in result["active_ingredients"]:
                                            strength_str = ing.get("strength", "")
                                            # Get the numeric part of the strength
                                            if " mg" in strength_str:
                                                try:
                                                    # Extract number and unit
                                                    num_str = strength_str.split(" mg")[0].replace("/1", "")
                                                    total_strength += float(num_str)
                                                    strength_unit = " mg"
                                                except:
                                                    pass
                                            elif " mcg" in strength_str:
                                                try:
                                                    num_str = strength_str.split(" mcg")[0].replace("/1", "")
                                                    total_strength += float(num_str)
                                                    strength_unit = " mcg"
                                                except:
                                                    pass
                                        
                                        # Format the total strength
                                        if total_strength > 0:
                                            if total_strength.is_integer():
                                                strength = f"{int(total_strength)}{strength_unit}"
                                            else:
                                                strength = f"{total_strength}{strength_unit}"
                                        else:
                                            # Fallback to listing all strengths
                                            strength = ", ".join([ing.get("strength", "").replace("/1", "") for ing in result["active_ingredients"]])
                                    else:
                                        # Single ingredient case - use the first ingredient's strength
                                        try:
                                            strength_str = result["active_ingredients"][0].get("strength", "")
                                            # Remove "/1" if present
                                            strength_str = strength_str.replace("/1", "")
                                            # Try to convert to integer if it's a whole number
                                            if " mg" in strength_str:
                                                strength_num = strength_str.split(" mg")[0]
                                                try:
                                                    if float(strength_num).is_integer():
                                                        strength_str = f"{int(float(strength_num))} mg"
                                                    else:
                                                        strength_str = f"{float(strength_num)} mg"
                                                except:
                                                    pass
                                            strength = strength_str
                                        except:
                                            strength = ", ".join([ing.get("strength", "").replace("/1", "") for ing in result.get("active_ingredients", [])])
                                
                                # Clean up package description - remove any NDC references more thoroughly
                                package_desc = package.get("description", "")
                                
                                # Remove pattern like " (NDC)" or " (NDC: 12345-678-90)" or "(57844-110-01)" at the end
                                import re
                                # First pattern: remove " (NDC)" or similar simple patterns
                                package_desc = re.sub(r'\s*\(\s*NDC\s*\)', '', package_desc, flags=re.IGNORECASE)
                                
                                # Second pattern: remove NDC with the actual number in parentheses
                                package_desc = re.sub(r'\s*\(\s*NDC\s*:?\s*[\d\-]+\s*\)', '', package_desc, flags=re.IGNORECASE)
                                
                                # Third pattern: just a number in parentheses at the end that looks like an NDC
                                package_desc = re.sub(r'\s*\(\s*[\d\-]+\s*\)\s*$', '', package_desc)
                                
                                # Extract the relevant information
                                ndc_info = {
                                    "NDC": package.get("package_ndc", ""),
                                    "brand_name": result.get("brand_name", ""),
                                    "generic_name": result.get("generic_name", ""),
                                    "strength": strength,
                                    "route": result.get("route", ""),
                                    "dosage_form": result.get("dosage_form", ""),
                                    "manufacturer": result.get("labeler_name", ""),
                                    "package_description": package_desc.strip()
                                }
                                results.append(ndc_info)
        except Exception as e:
            print(f"Error fetching generic name data: {e}")
    
    return results

def search_ndc(ndc, labels_to_get, include_source=True):
    """
    Search for an NDC in both OpenFDA and DailyMed, focusing on the specific fields requested.
    
    Args:
        ndc (str): The NDC code to search for
        labels_to_get (list): List of fields to include in the results
        include_source (bool): Include the source of each field in the results
        
    Returns:
        dict: Dictionary containing the results for the requested fields
    """
    api_key_param = ""  # OpenFDA API key parameter, empty if no API key
    result = {}
    
    # Define separate lists for regular fields and fields that might contain HTML tables
    regular_fields = []
    table_fields = ["dosage_and_administration", "clinical_studies", "indications_and_usage", "warnings"]
    special_fields = ["active_ingredient", "inactive_ingredient", "indications_and_usage", "warnings", "description"]
    
    # Categorize requested fields
    for field in labels_to_get:
        if field in table_fields:
            # Add _table suffix for clarity
            table_field = f"{field}_table"
            if table_field not in labels_to_get:
                labels_to_get.append(table_field)
        elif field not in special_fields:
            regular_fields.append(field)
            
    # Process special fields first
    for field in special_fields:
        if field in labels_to_get:
            try:
                print(f"Processing special field: {field}")
                if field == "active_ingredient":
                    data, source = get_label_field(ndc, field, ["Active Ingredient", "Active Ingredients", "Ingredients"])
                elif field == "inactive_ingredient":
                    data, source = get_label_field(ndc, field, ["Inactive Ingredient", "Inactive Ingredients"])
                elif field == "indications_and_usage":
                    data, source = get_label_field(ndc, field, ["Indications", "Uses", "Indications and Usage"])
                elif field == "warnings":
                    data, source = get_label_field(ndc, field, ["Warning", "Warnings"])
                elif field == "description":
                    data, source = get_label_field(ndc, field, ["Description"])
                
                result[field] = data
                if include_source:
                    result[f"{field}_source"] = source
            except Exception as e:
                print(f"Error retrieving special field {field}: {e}")
                result[field] = "Not available"
                if include_source:
                    result[f"{field}_source"] = "Error"
    
    # Process table fields efficiently - look for HTML tables in relevant fields
    try:
        # Try OpenFDA first for table content
        print("Checking OpenFDA for HTML tables")
        openfda_label_url = f"https://api.fda.gov/drug/label.json?search=openfda.package_ndc:{ndc}{api_key_param}&limit=1"
        response = requests.get(openfda_label_url)
        
        if response.status_code == 200:
            data = response.json()
            if "results" in data and len(data["results"]) > 0:
                product = data["results"][0]
                
                # Check each table field
                for field in table_fields:
                    table_field = f"{field}_table"
                    if table_field in labels_to_get and field in product:
                        content = product[field]
                        
                        # Check if it's a list and has content
                        if isinstance(content, list) and content:
                            # Look for HTML tables in the content
                            for item in content:
                                if isinstance(item, str) and "<table" in item:
                                    print(f"Found HTML table in OpenFDA field {field}")
                                    result[table_field] = item
                                    if include_source:
                                        result[f"{table_field}_source"] = "OpenFDA"
                                    break
        
        # If we didn't find tables in OpenFDA, try DailyMed
        if not any(f"{field}_table" in result for field in table_fields):
            print("Checking DailyMed for HTML tables")
            setid = get_setid_from_search(ndc)
            
            if setid:
                print(f"Found setid {setid} for NDC {ndc}")
                # Fetch the HTML content
                dailymed_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
                response = requests.get(dailymed_url)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Map fields to DailyMed section IDs
                    field_to_section = {
                        "dosage_and_administration_table": "dosage-administration",
                        "clinical_studies_table": "clinical-studies",
                        "indications_usage_table": "indications-usage",
                        "warnings_table": "warnings"
                    }
                    
                    # Check each section for tables
                    for table_field, section_id in field_to_section.items():
                        if table_field in labels_to_get:
                            section = soup.find(id=section_id)
                            if section:
                                tables = section.find_all('table')
                                if tables:
                                    print(f"Found {len(tables)} tables in DailyMed section {section_id}")
                                    result[table_field] = str(tables[0])
                                    if include_source:
                                        result[f"{table_field}_source"] = "DailyMed"
    except Exception as e:
        print(f"Error processing table fields: {e}")
    
    # Process regular fields
    if regular_fields:
        try:
            # Try OpenFDA first
            openfda_url = f"https://api.fda.gov/drug/ndc.json?search=package_ndc:{ndc}{api_key_param}"
            response = requests.get(openfda_url)
            
            if response.status_code == 200:
                data = response.json()
                
                if "results" in data and len(data["results"]) > 0:
                    product = data["results"][0]
                    
                    # Add all requested regular fields
                    for label in regular_fields:
                        if label in product:
                            result[label] = product[label]
                            if include_source:
                                result[f"{label}_source"] = "OpenFDA"
                        elif label.startswith("openfda.") and "openfda" in product:
                            # Handle nested openfda fields
                            nested_field = label.split(".", 1)[1]
                            if nested_field in product["openfda"]:
                                result[label] = product["openfda"][nested_field]
                                if include_source:
                                    result[f"{label}_source"] = "OpenFDA"
        except Exception as e:
            print(f"Error in OpenFDA lookup: {e}")
        
        # Try DailyMed for any missing regular fields
        try:
            dailymed_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/ndc/{ndc}.json"
            response = requests.get(dailymed_url)
            
            if response.status_code == 200:
                dailymed_data = response.json()
                
                # Process the DailyMed response
                for label in regular_fields:
                    if label not in result and label in dailymed_data:
                        result[label] = dailymed_data[label]
                        if include_source:
                            result[f"{label}_source"] = "DailyMed"
        except Exception as e:
            print(f"Error in DailyMed lookup: {e}")
    
    # For any requested fields that were not found
    for label in labels_to_get:
        if label not in result:
            result[label] = "Not available"
            if include_source:
                result[f"{label}_source"] = "Not found"
    
    return result

def get_single_item_source(item_dict, field_name):
    """
    Get the source of a single field from a drug data dictionary.
    
    Args:
        item_dict (dict): The dictionary containing drug data
        field_name (str): The name of the field to get the source for
        
    Returns:
        str: The source of the field ("OpenFDA", "DailyMed", or "Multiple Sources")
    """
    source_key = f"{field_name}_source"
    
    if source_key not in item_dict:
        return "Unknown"
    
    source = item_dict[source_key]
    
    # Check for OpenFDA source
    if "openfda" in source.lower():
        return "OpenFDA"
    # Check for DailyMed source
    elif "dailymed" in source.lower():
        return "DailyMed"
    # Handle multiple sources
    elif "openfda + dailymed" in source.lower():
        return "Multiple Sources"
    else:
        return source

def search_name_placeholder(name_str, limit=50):
    """
    A simple placeholder function for drug name searches that returns no data.
    This function is a compatibility wrapper used when the actual search functionality
    is not available or is being migrated to a new implementation.
    
    Args:
        name_str (str): The drug name to search for (unused)
        limit (int): Maximum number of results to return (unused)
        
    Returns:
        list: Empty list
    """
    # This is just a placeholder function to satisfy imports
    # Actual implementation should now use fetch_ndcs_for_name_drugsfda
    return []

def clean_html_table(html_content):
    """
    Clean and prepare an HTML table for safe rendering in Streamlit.
    
    Args:
        html_content (str): HTML table content
        
    Returns:
        str: Cleaned and styled HTML ready for rendering
    """
    if not html_content or not isinstance(html_content, str):
        logging.warning("Empty or non-string HTML content passed to clean_html_table")
        return ""
    
    try:
        # Use bleach to sanitize the HTML
        allowed_tags = [
            'table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th', 
            'caption', 'col', 'colgroup', 'div', 'span', 'p', 'br',
            'ul', 'ol', 'li', 'b', 'i', 'strong', 'em', 'sup', 'sub'
        ]
        
        allowed_attributes = {
            'table': ['width', 'border', 'cellspacing', 'cellpadding', 'align', 'id', 'class'],
            'th': ['scope', 'colspan', 'rowspan', 'width', 'align', 'valign'],
            'td': ['colspan', 'rowspan', 'width', 'align', 'valign'],
            'col': ['width', 'align', 'valign'],
            'div': ['class', 'style'],
            'span': ['class', 'style'],
            'caption': ['align', 'class'],
            '*': ['style', 'class'],
        }
        
        # Sanitize the HTML
        clean_html = bleach.clean(
            html_content,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )
        
        # Parse the cleaned HTML
        soup = BeautifulSoup(clean_html, "html.parser")
        
        # Add responsive container for table scrolling
        responsive_div = f'''
        <div style="overflow-x: auto; max-width: 100%;">
            {str(soup)}
        </div>
        '''
        
        return responsive_div
    
    except Exception as e:
        logging.error(f"Error in clean_html_table: {e}")
        return f"<p>Error processing HTML table: {str(e)}</p>"

def is_safe_table(html_content):
    """
    Verify that the HTML content only contains safe elements for rendering.
    Focuses on detecting dangerous content while being flexible with standard table elements.
    
    Args:
        html_content (str): HTML content to check
        
    Returns:
        bool: True if the content is safe to render
    """
    print(f"Checking HTML safety for content: {html_content[:50]}...")
    
    if not html_content or not isinstance(html_content, str):
        print("HTML content is empty or not a string")
        return False
    
    # Basic check for table tag
    if "<table" not in html_content:
        print("No table tags found in HTML content")
        return False
    
    try:
        # Check for dangerous JavaScript content
        dangerous_patterns = [
            "javascript:", "onerror=", "onclick=", "onload=", "onmouseover=", 
            "eval(", "<script", "alert(", "document.cookie", "window.location"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in html_content.lower():
                print(f"Dangerous pattern found: {pattern}")
                return False
        
        # More sophisticated check using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        
        # List of potentially dangerous tags
        dangerous_tags = {
            "script", "iframe", "object", "embed", "form", "input", "button", 
            "applet", "meta", "link", "base", "frame", "frameset"
        }
        
        # Check for dangerous tags
        for tag in soup.find_all():
            if tag.name.lower() in dangerous_tags:
                print(f"Dangerous tag found: {tag.name}")
                return False
        
        # Check for suspicious attributes 
        suspicious_attrs = ["onclick", "onload", "onerror", "onmouseover", "javascript", "eval"]
        for tag in soup.find_all():
            for attr, value in tag.attrs.items():
                attr_value = str(value).lower()
                if any(susp in attr.lower() or susp in attr_value for susp in suspicious_attrs):
                    print(f"Suspicious attribute detected: {attr}={value}")
                    return False
        
        print("HTML content passed safety check")
        return True
    except Exception as e:
        print(f"Error checking HTML safety: {e}")
        return False

def prepare_df_for_streamlit(df):
    """
    Prepare a DataFrame for display in Streamlit by ensuring all columns
    have PyArrow-compatible data types
    
    Args:
        df (pandas.DataFrame): DataFrame to prepare
        
    Returns:
        pandas.DataFrame: DataFrame with PyArrow-compatible data types
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert all columns to strings
    for col in df.columns:
        df[col] = df[col].astype(str)
        
    # Replace 'None' and 'nan' strings with empty strings
    df = df.replace({'None': '', 'nan': '', 'NaN': ''})
    
    return df