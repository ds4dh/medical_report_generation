"""
PMC-Patients Dataset Processing Script

This script processes the PMC-Patients dataset to extract a subset of patient 
case reports and saves them to a CSV file.

Dataset Source: https://github.com/pmc-patients/pmc-patients
"""

import pandas as pd
import json
import random
import os

# --- Configuration ---
INPUT_FILENAME = "PMC-Patients.json"
OUTPUT_FILENAME = "eng_pubmed.csv"
SUBSET_SIZE = 1706
SEED = 42


def process_pmc_patients_data(input_file: str, output_file: str, subset_size: int, seed: int):
    """
    Processes the PMC-Patients dataset to extract a subset of case reports 
    (patient info and PMID) and saves them to a CSV file.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output CSV file
        subset_size: Number of records to extract
        seed: Random seed for reproducibility
    
    Raises:
        FileNotFoundError: If the input file doesn't exist
        json.JSONDecodeError: If the JSON file is corrupted
    """
    
    print(f"Starting data processing (Subset size: {subset_size})...")
    random.seed(seed)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found at '{input_file}'. "
            "Please download PMC-Patients.json from "
            "https://github.com/pmc-patients/pmc-patients and place it in the same directory."
        )
    
    # Load JSON data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON. Ensure the file is not corrupted.")
        raise
    
    print(f"Successfully loaded {len(data)} total records.")
    
    # Extract case reports (patient info and PMID)
    case_reports = []
    skipped = 0
    
    for item in data:
        # Check if 'patient' and 'PMID' keys exist before appending
        if 'patient' in item and 'PMID' in item:
            case_reports.append((item['patient'], item['PMID']))
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} records with missing 'patient' or 'PMID' fields.")
    
    print(f"Extracted {len(case_reports)} valid case reports.")
    
    # Shuffle and select subset
    random.shuffle(case_reports)
    selected_reports = case_reports[:subset_size]
    
    print(f"Selected {len(selected_reports)} records after shuffling.")
    
    # Create DataFrame
    df = pd.DataFrame(selected_reports, columns=['eng_pubmed', 'PMID'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Successfully saved {len(df)} records to '{output_file}'")


if __name__ == "__main__":
    try:
        process_pmc_patients_data(INPUT_FILENAME, OUTPUT_FILENAME, SUBSET_SIZE, SEED)
    except FileNotFoundError as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print("\n📥 ACTION REQUIRED:")
        print("   1. Visit: https://github.com/pmc-patients/pmc-patients")
        print("   2. Download 'PMC-Patients.json'")
        print("   3. Place it in the same directory as this script")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")