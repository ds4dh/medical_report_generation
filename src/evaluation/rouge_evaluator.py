"""
Simple ROUGE Evaluation Script

Reads a CSV file with generated and reference texts, computes ROUGE metrics,
and saves the results.
"""

import argparse
import pandas as pd
from datetime import datetime
from rouge_score import rouge_scorer


def compute_rouge(input_file, output_file, rouge_types=['rouge1'],
                  generated_col="generated_text", reference_col="reference_text"):
    """
    Compute ROUGE scores for texts in a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        rouge_types (list): ROUGE metrics to compute (rouge1, rouge2, rougeL)
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
    
    # Initialize ROUGE scorer
    print(f"Computing ROUGE scores: {', '.join(rouge_types)}")
    scorer = rouge_scorer.RougeScorer(rouge_types)
    
    # Compute scores for each pair
    all_scores = {f'{rt}_{metric}': [] 
                  for rt in rouge_types 
                  for metric in ['precision', 'recall', 'fmeasure']}
    
    for gen_text, ref_text in zip(generated_texts, reference_texts):
        scores = scorer.score(ref_text, gen_text)
        
        for rouge_type in rouge_types:
            rouge_score = scores[rouge_type]
            all_scores[f'{rouge_type}_precision'].append(rouge_score.precision)
            all_scores[f'{rouge_type}_recall'].append(rouge_score.recall)
            all_scores[f'{rouge_type}_fmeasure'].append(rouge_score.fmeasure)
    
    # Add scores to dataframe
    for score_name, score_values in all_scores.items():
        df[score_name] = score_values
    
    # Save results
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\nResults saved to: {output_file}")
    print(f"\nSummary Statistics:")
    
    for rouge_type in rouge_types:
        avg_precision = sum(all_scores[f'{rouge_type}_precision']) / len(all_scores[f'{rouge_type}_precision'])
        avg_recall = sum(all_scores[f'{rouge_type}_recall']) / len(all_scores[f'{rouge_type}_recall'])
        avg_fmeasure = sum(all_scores[f'{rouge_type}_fmeasure']) / len(all_scores[f'{rouge_type}_fmeasure'])
        
        print(f"\n  {rouge_type.upper()}:")
        print(f"    Precision: {avg_precision:.4f}")
        print(f"    Recall:    {avg_recall:.4f}")
        print(f"    F-measure: {avg_fmeasure:.4f}")
    
    print(f"\nCompletion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute ROUGE scores for generated vs reference texts'
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
        '--rouge_types',
        type=str,
        nargs='+',
        default=['rouge1'],
        help='ROUGE metrics to compute (default: rouge1 rouge2 rougeL)'
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
    
    compute_rouge(
        args.input,
        args.output,
        args.rouge_types,
        args.generated_col,
        args.reference_col
    )


if __name__ == "__main__":
    main()
