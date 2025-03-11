import requests
import re
import io
import pandas as pd
import urllib.parse
import difflib
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

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
    return list(set(ndc_list))  # deduplicat

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