# PillQ – Pill Queries, Simplified

PillQ is a Streamlit application designed to help you quickly retrieve and display drug label data from various FDA endpoints and DailyMed. It's intended for formulary management, drug verification, and other clinical documentation tasks. By leveraging multiple sources (including openFDA, the FDA NDC directory, and DailyMed web scraping), PillQ aims to present consolidated medication information with minimal effort.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Data Flow and Architecture](#data-flow-and-architecture)
- [DailyMed Web Scraper](#dailymed-web-scraper)
- [Source Detection Logic](#source-detection-logic)
- [AI Assistant](#ai-assistant)
- [Limitations and Disclaimers](#limitations-and-disclaimers)
- [License](#license)

## Overview

PillQ provides both a single/multiple search interface (by NDC or drug name) and a file upload mode for batch processing. Internally, it uses:
- openFDA endpoints for drug labels and the Drugs@FDA data.
- The FDA NDC Directory endpoint for fallback NDC searches.
- A DailyMed web scraper for label data and product images when openFDA results are missing or incomplete.

The application is split into three main tabs:
1. **Single/Multiple Search:**
   - Enter NDCs or brand/generic names (comma-separated).
   - Choose output format (CSV, JSON, Excel, TXT).
   - Access the integrated AI Assistant for query help.
   - Optionally show the data's source (e.g., openFDA, DailyMed).
2. **File Upload:**
   - Upload a CSV containing an "NDC" column.
   - Choose output format before configuration.
   - Select which data fields to retrieve.
   - Process the file to get a consolidated preview of the results, then download.
3. **Settings:**
   - Configure AI provider settings (OpenAI, DeepSeek, Ollama, Zephyr).
   - Set API keys for your preferred AI providers.

## Features

1. **Single/Multiple Search**: Search for drug information by NDC code or name, with automatic detection of input type.
2. **File Upload**: Upload files containing NDC codes and select which columns to keep and which OpenFDA fields to add.
3. **AI Assistant**: Integrated chatbot that helps users answer questions about drugs using app functions.
4. **Multiple AI Provider Support**: Supports multiple AI providers (Ollama, OpenAI, DeepSeek, Zephyr) with a dedicated settings tab for configuration.
5. **Comprehensive Field Selection**: Access all available OpenFDA fields for searches.
6. **Improved Image Retrieval**: Correctly displays product images for specific NDCs.
7. **Enhanced Export Options**: Easily export data with configurable formats (CSV, JSON, Excel, TXT).
8. **Accurate Ingredient Information**: Retrieves both active and inactive ingredients reliably from multiple sources.

## Dependencies

This project uses the following libraries:
- `streamlit`: for the interactive web UI.
- `requests`: to perform HTTP requests (openFDA, FDA NDC directory, DailyMed).
- `pandas`: for data manipulation and DataFrame display.
- `beautifulsoup4`: for parsing HTML from DailyMed pages in the web scraper.
- `xlsxwriter`: for Excel file generation.
- `Pillow`: for image processing.
- `python-dotenv`: for environment variable management.
- `jsonpickle`: for advanced JSON serialization.
- `difflib`: for fuzzy matching in search operations.

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
   pip install -r requirements.txt
   ```
4. (Optional) Set up environment variables:
   Create a `.env` file in the project root with your API keys:
   ```
   OPENFDA_API_KEY=your_api_key_here
   OPENAI_API_KEY=your_openai_key_here
   ```

## Usage

1. Run the App:
   ```sh
   streamlit run app.py
   ```
   If run locally, Streamlit will display a local URL in the terminal (e.g. `http://localhost:8501`).
2. Navigate to the local URL in your browser. You'll see three tabs:
   - **Single/Multiple Search:**
     1. Enter comma-separated NDCs or drug names.
     2. Select OpenFDA fields to retrieve.
     3. Click **Preview Results**.
     4. View detailed information for selected NDCs.
     5. Export data in your preferred format.
     6. Optionally use the AI Assistant for drug-related questions.
   - **File Upload:**
     1. Upload a file containing NDC codes.
     2. Choose your preferred output format (CSV by default).
     3. Configure which columns to keep and which fields to retrieve.
     4. Click **Process File**.
     5. Download your enriched data.
   - **Settings:**
     1. Configure AI model providers.
     2. Set API keys for selected providers.
     3. Save preferences.

## Data Flow and Architecture

1. **User Input:**
   - Single/Multiple Search: NDC or brand/generic name(s).
   - File Upload: CSV/JSON/Excel/TXT with "NDC".
2. **Core Retrieval:**
   - `search_ndc(ndc, labels, include_source)` queries openFDA and DailyMed for comprehensive data.
   - Specialized handling for active and inactive ingredients with fallback mechanisms.
   - DailyMed web scraper used when openFDA data is incomplete.
3. **Unifying Source:**
   - Results tracked with source information.
   - Sources displayed based on user preference.
4. **Output:**
   - Results displayed in customizable formats with download options.

## DailyMed Web Scraper

The app's DailyMed scraper functions:
1. Build DailyMed URLs based on NDC or set_id.
2. Fetch and parse HTML content.
3. Extract structured data including ingredients, dosages, and images.
4. Provide high-quality data when openFDA results are incomplete.

## Source Detection Logic

1. Each data field has a source attribution.
2. Sources are consolidated for user display when requested.
3. Consistent source formatting improves readability.

## AI Assistant

The integrated AI Assistant supports:
1. Multiple AI providers:
   - OpenAI (requires API key)
   - DeepSeek (requires API key)
   - Ollama (local, free - requires installation)
   - Zephyr (via HuggingFace API)
2. Context-aware drug information:
   - Automatically retrieves relevant drug data for references in user questions
   - Provides helpful information on pharmaceutical topics

## Limitations and Disclaimers

- **Data Completeness:** Not all drug entries in openFDA or DailyMed are fully populated.
- **Rate Limits:** openFDA imposes query limits; consider obtaining an API key for production use.
- **AI Assistant Accuracy:** The AI provides helpful information but should not be considered medical advice.
- **Not Medical Advice:** Always verify details with official FDA labeling and consult healthcare professionals.
