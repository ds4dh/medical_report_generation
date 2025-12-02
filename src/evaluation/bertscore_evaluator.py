#!/usr/bin/env python3
"""
Simple BERTScore Evaluation Script

Reads a CSV file with generated and reference texts, computes BERTScore metrics,
and saves the results.

Usage:
    python bertscore_simple.py --input data.csv --output results.csv
"""

import argparse
import pandas as pd
from datetime import datetime
from evaluate import load


def compute_bertscore(input_file, output_file, model_type, 
                      generated_col="generated_text", reference_col="reference_text"):
    """
    Compute BERTScore for texts in a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        model_type (str): BERTScore model type
        generated_col (str): Name of column with generated texts
        reference_col (str): Name of column with reference texts
    """
    print(f"Reading CSV file: {input_file}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Validate columns
    if generated_col not in df.columns or reference_col not in df.columns:
        raise ValueError(f"CSV must contain '{generated_col}' and '{reference_col}' columns")
    
    # Remove rows with missing values
    df = df.dropna(subset=[generated_col, reference_col])
    
    print(f"Found {len(df)} valid text pairs")
    
    # Extract texts
    generated_texts = df[generated_col].astype(str).tolist()
    reference_texts = df[reference_col].astype(str).tolist()
    
    # Load BERTScore
    print(f"Computing BERTScore using {model_type}...")
    bertscore = load("bertscore")
    
    # Compute scores
    results = bertscore.compute(
        predictions=generated_texts,
        references=reference_texts,
        model_type=model_type
    )
    
    # Add scores to dataframe
    df['bertscore_precision'] = results['precision']
    df['bertscore_recall'] = results['recall']
    df['bertscore_f1'] = results['f1']
    
    # Save results
    df.to_csv(output_file, index=False)
    
    # Print summary
    avg_precision = sum(results['precision']) / len(results['precision'])
    avg_recall = sum(results['recall']) / len(results['recall'])
    avg_f1 = sum(results['f1']) / len(results['f1'])
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nSummary Statistics:")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Average Recall:    {avg_recall:.4f}")
    print(f"  Average F1:        {avg_f1:.4f}")
    print(f"\nCompletion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute BERTScore for generated vs reference texts'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='bert-base-uncased',
        help='BERTScore model type'
    )
    
    parser.add_argument(
        '--generated_col',
        type=str,
        default='generated_text',
        help='Name of generated text column (default: generated_text)'
    )
    
    parser.add_argument(
        '--reference_col',
        type=str,
        default='reference_text',
        help='Name of reference text column (default: reference_text)'
    )
    
    args = parser.parse_args()
    
    compute_bertscore(
        args.input,
        args.output,
        args.model_type,
        args.generated_col,
        args.reference_col
    )


if __name__ == "__main__":
    main()
