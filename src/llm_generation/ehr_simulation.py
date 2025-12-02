"""
Synthetic EHR Simulator using vLLM

This script generates simulated Electronic Health Records (EHR) from medical case reports using a language model. 
"""

import os
import csv
import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
from vllm import LLM, SamplingParams


# ============================================================================
# PROMPT CREATION
# ============================================================================

def create_ehr_simulation_prompt(case_report_text: str) -> Tuple[str, str]:
    """
    Create a structured prompt for EHR simulation from medical case reports.
    
    Args:
        case_report_text (str): The medical case report text
        
    Returns:
        tuple: (role_description, full_question)
    """
    role_description = (
        'You are an expert in the medical field, specialising in analysing '
        'and writing case reports.'
    )
    
    question = f'''
**** Objective ****
As an experienced physician specializing in medical documentation and electronic health record (EHR) systems, your task is to simulate a realistic EHR entry based on the provided medical case report. Generate a synthetic EHR that captures all clinical information as it would have been originally recorded in the patient's electronic health record, either in structured fields or as free-text clinical notes.

*** Methodology: ***
1. Carefully read and analyze the entire case report
2. Identify all clinically relevant information and entities
3. Strictly maintain the chronological order of events as presented
4. Simulate the EHR with sufficient detail to enable faithful reconstruction of the original report
5. Ensure all simulated entries are medically accurate and precise
6. Include temporal relationships and clinical context

*** Format Requirements: ***
- Present the simulated EHR in a clear, structured format of your choice
- Maintain chronological accuracy
- Use precise medical terminology
- Include all relevant quantitative values and measurements
- Preserve causal and temporal relationships between entries
- If specific information is absent, do not speculate

**** Target Case Report ****

**Case Report:**
{case_report_text}

**Simulated EHR:**
'''
    
    return role_description, question


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def validate_input_file(file_path: str) -> None:
    """
    Validate that the input file exists and has required columns.
    
    Args:
        file_path (str): Path to input CSV file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}\n"
            f"Please ensure the file exists"
        )
    
    # Check for required columns
    df = pd.read_csv(file_path)
    required_columns = ['CR_id', 'language', 'text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {missing_columns}\n"
            f"Required columns: {required_columns}\n"
            f"Found columns: {list(df.columns)}"
        )


def process_batch_generation(
    input_file_path: str,
    llm: LLM,
    sampling_params: SamplingParams,
    output_file_path: str,
    batch_size: int,
    language: str
) -> bool:
    """
    Process EHR simulation in batches.
    
    Args:
        input_file_path (str): Path to input CSV file with case reports
        llm (LLM): Initialized vLLM model
        sampling_params (SamplingParams): Sampling parameters for generation
        output_file_path (str): Path to save output CSV file
        batch_size (int): Number of samples to process in each batch
        language (str): Language to filter reports by
        
    Returns:
        bool: True if successful
        
    Raises:
        Exception: If processing fails
    """
    try:
        # Load input data
        print(f"Loading input data from: {input_file_path}")
        df = pd.read_csv(input_file_path)
        
        # Filter for specified language only
        df = df[df['language'] == language]
        
        case_reports = df['text'].tolist()
        cr_ids = df['CR_id'].tolist()
        
        total_samples = len(case_reports)
        print(f"Total samples to process: {total_samples}")
        
        if total_samples == 0:
            print("Warning: No samples found in input file")
            return False
        
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'CR_id',
                'case_report',
                'simulated_EHR',
            ]
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            
            # Process in batches
            total_batches = (total_samples + batch_size - 1) // batch_size
            
            for batch_idx in range(0, total_samples, batch_size):
                batch_end = min(batch_idx + batch_size, total_samples)
                current_batch = (batch_idx // batch_size) + 1
                
                print(f"Processing batch {current_batch}/{total_batches}: "
                      f"samples {batch_idx} to {batch_end-1}")
                
                # Get batch data
                case_report_batch = case_reports[batch_idx:batch_end]
                cr_id_batch = cr_ids[batch_idx:batch_end]
                
                # Create prompts for batch
                all_messages = []
                for case_report in case_report_batch:
                    role, query = create_ehr_simulation_prompt(case_report)
                    chat_message = [
                        {"role": "system", "content": role},
                        {"role": "user", "content": query},
                    ]
                    all_messages.append(chat_message)
                
                # Generate simulated EHRs
                if all_messages:
                    try:
                        batch_outputs = llm.chat(
                            all_messages,
                            sampling_params,
                        )
                        
                        # Save results
                        for j, output in enumerate(batch_outputs):
                            generated_text = output.outputs[0].text.strip()
                            
                            instance = {
                                'CR_id': cr_id_batch[j],
                                'case_report': case_report_batch[j],
                                'simulated_EHR': generated_text,
                            }
                            
                            csv_writer.writerow(instance)
                    
                    except Exception as e:
                        print(f"Error processing batch {current_batch}: {str(e)}")
                        raise
        
        print(f"\nResults saved to: {output_file_path}")
        print(f"Successfully processed {total_samples} samples")
        return True
    
    except Exception as e:
        print(f"Error in batch generation: {str(e)}", file=sys.stderr)
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Synthetic EHR Simulator using vLLM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model_name',
        type=str,
        default='microsoft/phi-4',
        help='HuggingFace model identifier'
    )
    model_group.add_argument(
        '--max_model_len',
        type=int,
        default=16000,
        help='Maximum sequence length'
    )
    model_group.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism'
    )
    model_group.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.95,
        help='Fraction of GPU memory to use (0.0-1.0)'
    )
    
    # Sampling Parameters
    sampling_group = parser.add_argument_group('Sampling Parameters')
    sampling_group.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Sampling temperature'
    )
    sampling_group.add_argument(
        '--top_p',
        type=float,
        default=0.85,
        help='Top-p sampling threshold'
    )
    sampling_group.add_argument(
        '--min_p',
        type=float,
        default=0.0,
        help='Minimum probability threshold'
    )
    sampling_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    sampling_group.add_argument(
        '--max_tokens',
        type=int,
        default=10000,
        help='Maximum tokens to generate'
    )
    
    # Processing Parameters
    processing_group = parser.add_argument_group('Processing Parameters')
    processing_group.add_argument(
        '--language',
        type=str,
        default='English',
        help='Language of the reports'
    )
    processing_group.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Number of samples per batch'
    )
    
    # Input/Output Paths
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        '--input_file',
        type=str,
        default ='./Dataset/test/case_reports.csv',
        # required=True,
        help='Path to input CSV file with columns: CR_id, language, text'
    )
    io_group.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Output directory'
    )
    io_group.add_argument(
        '--output_name',
        type=str,
        default='simulated_ehr',
        help='Output file name (without .csv extension)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    print("=" * 70)
    print("Synthetic EHR Simulator")
    print("=" * 70)
    
    try:
        # Validate input file
        validate_input_file(args.input_file)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        output_file_path = os.path.join(
            args.output_dir,
            f"{args.output_name}.csv"
        )
        
        # Initialize sampling parameters
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            seed=args.seed,
            max_tokens=args.max_tokens
        )
        
        # Print configuration
        print("\nConfiguration:")
        print(f"  Model: {args.model_name}")
        print(f"  Max model length: {args.max_model_len}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Language: {args.language}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-p: {args.top_p}")
        print(f"  Seed: {args.seed}")
        print(f"  Max tokens: {args.max_tokens}")
        
        # Initialize model
        print(f"\nInitializing model: {args.model_name}")
        print("This may take a few minutes...")
        
        llm = LLM(
            model=args.model_name,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.batch_size,
        )
        print("Model loaded successfully!")
        
        print(f"\nInput file: {args.input_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Output file: {output_file_path}")
        
        # Process generation
        success = process_batch_generation(
            args.input_file,
            llm,
            sampling_params,
            output_file_path,
            args.batch_size,
            args.language
        )
        
        if success:
            print("\n" + "=" * 70)
            print("Processing completed successfully!")
            print("=" * 70)
            return 0
        else:
            print("\n" + "=" * 70)
            print("Processing completed with warnings")
            print("=" * 70)
            return 1
    
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        print("\n" + "=" * 70)
        print("Processing failed!")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
