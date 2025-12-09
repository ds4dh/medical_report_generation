"""
EHR Simulator using vLLM

Simulate Electronic Health Records from case reports or medical transcripts.
Supports english and french, zero-shot only.

Usage:
    # Case reports
    python generate.py --task case_report --language english --input_file data.csv
    
    # Medical transcripts
    python generate.py --task transcript --language french --input_file data.csv
"""

import os
import sys
import csv
import logging
import argparse
from pathlib import Path
from datetime import datetime

from vllm import LLM, SamplingParams

from config import *
from prompts import get_prompt_creator
from utils import validate_files, load_case_report_data, load_transcript_data


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_case_report_ehrs(
    input_file: str,
    llm: LLM,
    sampling_params: SamplingParams,
    output_file: str,
    batch_size: int,
    language: str
):
    """Simulate EHRs from case reports."""
    logger.info(f"Simulating EHRs from case reports: {language}")
    
    # Get prompt function
    prompt_func = get_prompt_creator('case_report', language)
    
    # Load input data
    case_reports, cr_ids = load_case_report_data(input_file, language)
    logger.info(f"Processing {len(case_reports)} case reports")
    
    # Create output dir
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Process batches
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TASK_COLUMNS['case_report']['output'])
        writer.writeheader()
        
        for i in range(0, len(case_reports), batch_size):
            batch_end = min(i + batch_size, len(case_reports))
            logger.info(f"Batch {i//batch_size + 1}: samples {i}-{batch_end-1}")
            
            # Create prompts
            messages = []
            for report in case_reports[i:batch_end]:
                sys_prompt, user_prompt = prompt_func(report)
                messages.append([
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ])
            
            # Generate
            outputs = llm.chat(messages, sampling_params)
            
            # Save
            for j, output in enumerate(outputs):
                idx = i + j
                writer.writerow({
                    'id': cr_ids[idx],
                    'language': language,
                    'case_report': case_reports[idx],
                    'simulated_ehr': output.outputs[0].text.strip()
                })
    
    logger.info(f"✓ Results saved to: {output_file}")


def simulate_transcript_ehrs(
    input_file: str,
    llm: LLM,
    sampling_params: SamplingParams,
    output_file: str,
    batch_size: int,
    language: str
):
    """Simulate EHRs from medical transcripts."""
    logger.info(f"Simulating EHRs from transcripts: {language}")
    
    # Get prompt function
    prompt_func = get_prompt_creator('transcript', language)
    
    # Load input data
    reports, specialties, report_types, mt_ids = load_transcript_data(input_file, language)
    logger.info(f"Processing {len(reports)} transcripts")
    
    # Create output dir
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Process batches
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TASK_COLUMNS['transcript']['output'])
        writer.writeheader()
        
        for i in range(0, len(reports), batch_size):
            batch_end = min(i + batch_size, len(reports))
            logger.info(f"Batch {i//batch_size + 1}: samples {i}-{batch_end-1}")
            
            # Create prompts
            messages = []
            for idx in range(i, batch_end):
                sys_prompt, user_prompt = prompt_func(
                    reports[idx],
                    specialties[idx],
                    report_types[idx]
                )
                messages.append([
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ])
            
            # Generate
            outputs = llm.chat(messages, sampling_params)
            
            # Save
            for j, output in enumerate(outputs):
                idx = i + j
                writer.writerow({
                    'id': mt_ids[idx],
                    'medical_report': reports[idx],
                    'specialty': specialties[idx],
                    'report_type': report_types[idx],
                    'simulated_ehr': output.outputs[0].text.strip()
                })
    
    logger.info(f"✓ Results saved to: {output_file}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Simulate EHRs using vLLM')
    
    # Task type
    parser.add_argument('--task', choices=['case_report', 'transcript'], required=True,
                        help='Task type: case_report or transcript')
    parser.add_argument('--language', choices=['english', 'french'], default='english',
                        help='Output language')
    
    # File paths
    parser.add_argument('--input_file', required=True, help='Input CSV file')
    parser.add_argument('--output_dir', default='./outputs', help='Output directory')
    parser.add_argument('--output_name', default=None, help='Output filename (without .csv)')
    
    # Model configuration
    parser.add_argument('--model_name', default=DEFAULT_MODEL, help='Model name')
    parser.add_argument('--max_model_len', type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    
    # Generation parameters
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--top_p', type=float, default=DEFAULT_TOP_P)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS)
    
    args = parser.parse_args()
    
    # Set output name
    if args.output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_name = f'{args.task}_ehr_simulation_{args.language}_{timestamp}'
    
    output_file = os.path.join(args.output_dir, f"{args.output_name}.csv")
    
    try:
        # Validate
        validate_files(args.input_file, args.language, args.task)
        
        # Initialize model
        logger.info(f"Loading model: {args.model_name}")
        llm = LLM(
            model=args.model_name,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.batch_size
        )
        logger.info("✓ Model loaded")
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_tokens=args.max_tokens
        )
        
        # Simulate EHRs
        if args.task == 'case_report':
            simulate_case_report_ehrs(
                args.input_file, llm, sampling_params, output_file,
                args.batch_size, args.language
            )
        else:  # transcript
            simulate_transcript_ehrs(
                args.input_file, llm, sampling_params, output_file,
                args.batch_size, args.language
            )
        
        logger.info("=" * 60)
        logger.info("✓ EHR simulation completed successfully!")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
