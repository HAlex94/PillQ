# PillQ – Pill Queries, Simplified

PillQ is a Streamlit application designed to help you quickly retrieve and display drug label data from various FDA endpoints and DailyMed. It’s intended for formulary management, drug verification, and other clinical documentation tasks. By leveraging multiple sources (including openFDA, the FDA NDC directory, and DailyMed web scraping), PillQ aims to present consolidated medication information with minimal effort.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Data Flow and Architecture](#data-flow-and-architecture)
- [DailyMed Web Scraper](#dailymed-web-scraper)
- [Source Detection Logic](#source-detection-logic)
- [Limitations and Disclaimers](#limitations-and-disclaimers)
- [License](#license)

## Overview

PillQ provides both a single/multiple search interface (by NDC or drug name) and a file upload mode for batch processing. Internally, it uses:
- openFDA endpoints for drug labels and the Drugs@FDA data.
- The FDA NDC Directory endpoint for fallback NDC searches.
- A DailyMed web scraper for label data and product images when openFDA results are missing or incomplete.

The application is split into two main tabs:
1. **Single/Multiple Search:**
   - Enter NDCs or brand/generic names (comma-separated).
   - Choose output format (JSON, CSV, Excel, TXT).
   - Optionally show the data’s source (e.g., openFDA, DailyMed).
2. **File Upload:**
   - Upload a CSV containing an “NDC” column.
   - Select which data fields to retrieve.
   - Process the file to get a consolidated preview of the results, then download.

## Key Features

1. **NDC or Name:**
   - Single or comma-separated entries.
   - If searching by name, fallback logic fetches brand/generic info from openFDA.
2. **Fallback:**
   - If certain data fields (like `product_ndc`) are missing, the app tries the NDC Directory or DailyMed.
3. **DailyMed Web Scraper:**
   - If openFDA fails to provide certain label fields or images, the app can scrape the relevant page on DailyMed.
4. **Source Info:**
   - Each data field can note whether it was retrieved from openFDA, DailyMed, or both.
   - Uses a custom `unify_source_string()` function to unify the final source strings.
5. **Downloadable Results:**
   - Output can be previewed in your chosen format and saved locally.

## Dependencies

This project uses the following core libraries:
- `streamlit`: for the interactive web UI.
- `requests`: to perform HTTP requests (openFDA, FDA NDC directory, DailyMed).
- `pandas`: for data manipulation and DataFrame display.
- `BeautifulSoup (bs4)`: for parsing HTML from DailyMed pages in the web scraper.
- `urllib.parse`: to construct and encode query strings (e.g., for openFDA queries).
- `io`: for handling in-memory file objects (Excel/CSV downloads).
- `re`: for regex operations (used in scraping or cleaning text).
- `difflib`: for fuzzy matching (if needed in your code).
- `xslxwriter` (implied by Excel exports)
- `helper_functions` (internal) which includes:
  - `get_openfda_searchable_fields()`
  - `get_product_image()`
  - `get_product_ndc()`
  - `get_label_field()`
  - `get_combined_label_field()`
  - `get_setid_from_search()`
  - `fallback_ndc_search()`
  - `unify_source_string()`

Ensure these packages are installed in your environment. If you maintain a `requirements.txt`, verify it lists them.

## Installation

1. Clone or download this repository.
   ```sh
   git clone https://github.com/YourOrg/PillQ.git
   cd PillQ
   ```
2. Create a Virtual Environment (recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install Dependencies:
   ```sh
   pip install streamlit requests pandas beautifulsoup4 xlsxwriter
   ```

## Usage

1. Run the App:
   ```sh
   streamlit run app.py
   ```
   or:
   ```sh
   streamlit run your_app_file.py
   ```
   If run locally, Streamlit will display a local URL in the terminal (e.g. `http://localhost:8501`).
2. Navigate to the local URL in your browser. You’ll see two tabs:
   - **Single/Multiple Search:**
     1. Choose “NDC” or “Name.”
     2. Type comma-separated entries.
     3. Click **Preview Data**.
     4. (Optional) Display source info, product images, or switch output formats.
   - **File Upload:**
     1. Upload a CSV with an “NDC” column.
     2. Pick the data fields you want (brand_name, generic_name, etc.).
     3. Click **Process File**.
     4. Download your results in CSV, Excel, TXT, or JSON.
3. Check the console or Streamlit logs if you encounter errors or rate-limit warnings.

## Data Flow and Architecture

1. **User Input:**
   - Single/Multiple Search: NDC or brand/generic name(s).
   - File Upload: CSV with “NDC.”
2. **Core Retrieval:**
   - `search_ndc(ndc, labels, include_source)` queries openFDA or calls your fallback logic.
   - `fetch_ndcs_for_name_drugsfda(name_str)` queries the `/drugsfda` endpoint to find brand/generic matches, fallback to the NDC directory if needed.
   - DailyMed web scraper (in `get_label_field` or a separate function) is used if openFDA fails to return certain fields or if you explicitly prefer DailyMed data.
3. **Unifying Source:**
   - `_source` fields are set to contain “openfda” or “dailymed.”
   - The function `get_single_item_source(...)` checks these fields to produce “openFDA,” “DailyMed,” or “openFDA + DailyMed.”
4. **Output:**
   - Results displayed in a table or JSON, with optional download in multiple formats.

## DailyMed Web Scraper

In some scenarios, openFDA data is incomplete. The app’s DailyMed scraper (referenced in `helper_functions.py`) will:
1. Build the correct DailyMed URL (using `set_id` or an NDC search approach).
2. Use `requests` to fetch the HTML.
3. Parse the relevant table rows with BeautifulSoup.
4. Extract fields like inactive ingredients, active moiety, or product images.
5. Return them to the main app.

## Source Detection Logic

1. Each data field has a `_source` key set by `search_ndc(...)`.
2. If a field is retrieved from openFDA, `_source` might contain “openfda.” If it’s from DailyMed scraping, `_source` might contain “dailymed.”
3. The function `unify_source_string(...)` ensures that any raw string from your helper is converted to a consistent “openfda,” “dailymed,” or “openfda + dailymed.”
4. Finally, `get_single_item_source(...)` inspects all fields to determine the final “(Source: X)” displayed in the UI.

## Limitations and Disclaimers

- **Data Completeness:** Not all drug entries in openFDA or DailyMed are fully populated.
- **Rate Limits:** openFDA imposes query limits.
- **Not Medical Advice:** Always verify details with official FDA labeling.
