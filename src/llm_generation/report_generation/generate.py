"""
Medical Text Generator using vLLM

Generate case reports or medical transcripts from EHR data.
Supports english and french, zero-shot and few-shot prompting.

Usage:
    # Case reports
    python generate.py --task case_report --approach zeroshot --language english
    
    # Medical transcripts
    python generate.py --task transcript --approach fewshot --language french
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
from utils import (
    validate_files, 
    load_case_report_data, 
    load_transcript_data,
    remove_lines_after_condition
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_case_reports(
    input_file: str,
    llm: LLM,
    sampling_params: SamplingParams,
    output_file: str,
    batch_size: int,
    approach: str,
    language: str,
    dev_file: str = None,
    num_shots: int = DEFAULT_NUM_SHOTS
):
    """Generate case reports."""
    logger.info(f"Generating case reports: {approach}, {language}")
    
    # Get prompt functions
    prompt_funcs = get_prompt_creator('case_report', language)
    
    # Load dev set for few-shot
    fewshot_examples = None
    if approach == 'fewshot':
        dev_ehrs, dev_reports, _ = load_case_report_data(dev_file, language)
        fewshot_examples = prompt_funcs['create_fewshot_examples'](
            dev_ehrs, dev_reports, num_shots
        )
    
    # Load input data
    input_ehrs, ref_reports, cr_ids = load_case_report_data(input_file, language)
    logger.info(f"Processing {len(input_ehrs)} samples")
    
    # Create output dir
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Process batches
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TASK_COLUMNS['case_report']['output'])
        writer.writeheader()
        
        for i in range(0, len(input_ehrs), batch_size):
            batch_end = min(i + batch_size, len(input_ehrs))
            logger.info(f"Batch {i//batch_size + 1}: samples {i}-{batch_end-1}")
            
            # Create prompts
            messages = []
            for ehr in input_ehrs[i:batch_end]:
                sys_prompt, user_prompt = prompt_funcs['create_prompt'](ehr, fewshot_examples)
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
                    'ehr': input_ehrs[idx],
                    'reference_report': ref_reports[idx],
                    'generated_report': output.outputs[0].text.strip()
                })
    
    logger.info(f"✓ Results saved to: {output_file}")


def generate_transcripts(
    input_file: str,
    llm: LLM,
    sampling_params: SamplingParams,
    output_file: str,
    batch_size: int,
    approach: str,
    language: str,
    dev_file: str = None,
    num_shots: int = DEFAULT_NUM_SHOTS
):
    """Generate medical transcripts."""
    logger.info(f"Generating transcripts: {approach}, {language}")
    
    # Get prompt functions
    prompt_funcs = get_prompt_creator('transcript', language)
    
    # Load dev set for few-shot
    dev_data = None
    if approach == 'fewshot':
        dev_data = load_transcript_data(dev_file, language)
    
    # Load input data
    input_ehrs, ref_reports, specialties, report_types = load_transcript_data(input_file, language)
    logger.info(f"Processing {len(input_ehrs)} samples")
    
    # Create output dir
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Process batches
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TASK_COLUMNS['transcript']['output'])
        writer.writeheader()
        
        for i in range(0, len(input_ehrs), batch_size):
            batch_end = min(i + batch_size, len(input_ehrs))
            logger.info(f"Batch {i//batch_size + 1}: samples {i}-{batch_end-1}")
            
            # Create prompts
            messages = []
            for idx in range(i, batch_end):
                ehr = input_ehrs[idx]
                
                # Apply preprocessing for zero-shot
                if approach == 'zeroshot':
                    ehr = remove_lines_after_condition(ehr)
                
                # Create prompt
                if approach == 'fewshot':
                    # Extract matching few-shot samples
                    fewshot_examples = prompt_funcs['create_fewshot_examples'](
                        *dev_data,
                        specialties[idx],
                        report_types[idx],
                        num_shots
                    )
                    sys_prompt, user_prompt = prompt_funcs['create_prompt'](
                        ehr, specialties[idx], report_types[idx], fewshot_examples
                    )
                else:
                    sys_prompt, user_prompt = prompt_funcs['create_prompt'](
                        ehr, specialties[idx], report_types[idx]
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
                    'input_ehr': input_ehrs[idx],
                    'reference_report': ref_reports[idx],
                    'generated_report': output.outputs[0].text.strip(),
                    'doctor_specialty': specialties[idx],
                    'report_type': report_types[idx]
                })
    
    logger.info(f"✓ Results saved to: {output_file}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Generate medical text using vLLM')
    
    # Task type
    parser.add_argument('--task', choices=['case_report', 'transcript'], required=True,
                        help='Task type: case_report or transcript')
    parser.add_argument('--approach', choices=['zeroshot', 'fewshot'], default='zeroshot',
                        help='Prompting approach')
    parser.add_argument('--language', choices=['english', 'french'], default='english',
                        help='Output language')
    
    # File paths
    parser.add_argument('--input_file', required=True, help='Input CSV file')
    parser.add_argument('--dev_file', default=None, help='Dev CSV file (for few-shot)')
    parser.add_argument('--output_dir', default='./outputs', help='Output directory')
    parser.add_argument('--output_name', default=None, help='Output filename (without .csv)')
    
    # Model configuration
    parser.add_argument('--model_name', default=DEFAULT_MODEL, help='Model name')
    parser.add_argument('--max_model_len', type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    
    # Generation parameters
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num_shots', type=int, default=DEFAULT_NUM_SHOTS)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--top_p', type=float, default=DEFAULT_TOP_P)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS)
    
    args = parser.parse_args()
    
    # Set output name
    if args.output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_name = f'{args.task}_{args.approach}_{args.language}_{timestamp}'
    
    output_file = os.path.join(args.output_dir, f"{args.output_name}.csv")
    
    try:
        # Validate
        dev_file = args.dev_file if args.approach == 'fewshot' else None
        validate_files(args.input_file, dev_file, args.approach, args.language, args.task)
        
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
        
        # Generate
        if args.task == 'case_report':
            generate_case_reports(
                args.input_file, llm, sampling_params, output_file,
                args.batch_size, args.approach, args.language, dev_file, args.num_shots
            )
        else:  # transcript
            generate_transcripts(
                args.input_file, llm, sampling_params, output_file,
                args.batch_size, args.approach, args.language, dev_file, args.num_shots
            )
        
        logger.info("=" * 60)
        logger.info("✓ Generation completed successfully!")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
