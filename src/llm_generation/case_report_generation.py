"""
Medical case report generator using vLLM (Multilingual: English/French)
This script generates medical case reports from input simulated EHR using zero-shot or few-shot prompting.
"""

import os
import csv
import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import pandas as pd
from vllm import LLM, SamplingParams


# ============================================================================
# PROMPT CREATION - FEW-SHOT (ENGLISH)
# ============================================================================

def create_fewshot_examples_en(
    dev_ehrs: List[str],
    dev_reports: List[str],
    num_shots: int
) -> str:
    """
    Create few-shot examples from development set (English).
    
    Args:
        dev_ehrs (List[str]): List of simulated EHRs from dev set
        dev_reports (List[str]): List of case reports from dev set
        num_shots (int): Number of few-shot examples to include
        
    Returns:
        str: Formatted few-shot examples
    """
    fewshots = []
    for idx in range(min(num_shots, len(dev_ehrs))):
        example = f"""Sample ({idx + 1}):
**Input EHR:**
{dev_ehrs[idx]}

**Generated Case Report:**
{dev_reports[idx]}

"""
        fewshots.append(example)
    
    return '\n'.join(fewshots)


def create_fewshot_prompt_en(
    target_ehr: str,
    fewshot_examples: str
) -> Tuple[str, str]:
    """
    Create a structured prompt for few-shot medical case report generation (English).
    
    Args:
        target_ehr (str): Input EHR for target case
        fewshot_examples (str): Pre-formatted few-shot examples
        
    Returns:
        tuple: (role_description, full_question)
    """
    role_description = (
        'You are an expert in the medical field, specialising in analysing '
        'and writing case reports.'
    )
    question = f"""**** Objective ****
As an experienced physician who specializes in medical documentation, your task is to produce a detailed and structured clinical case report in English based on the provided Electronic Health Record (EHR).

**** Methodology to follow: ****
1. Carefully analyze all EHR data thoroughly
2. Strictly adhere to the chronological order of clinical events
3. Establish logical connections between symptoms, diagnoses, and treatments
4. Incorporate all relevant information without omissions
5. Use precise and appropriate medical terminology
6. Maintain an objective, factual, and professional tone

**** Format Requirements ****
- A single coherent and fluid paragraph
- Concise yet comprehensive writing style (following PubMed case report conventions)
- Technical vocabulary conforming to medical standards
- Sentence structures typical of published case reports
- If information is missing, omit rather than speculate

**** Few-Shot Examples ****
{fewshot_examples}

**** Target Case ****

**Input EHR:**
{target_ehr}

**Generated Case Report:**
"""
    return role_description, question


# ============================================================================
# PROMPT CREATION - ZERO-SHOT (ENGLISH)
# ============================================================================

def create_zeroshot_prompt_en(input_ehr: str) -> Tuple[str, str]:
    """
    Create a structured prompt for zero-shot medical case report generation (English).
    
    Args:
        input_ehr (str): A simulated electronic health record
        
    Returns:
        tuple: (role_description, full_question)
    """
    role_description = (
        'You are an expert in the medical field, specialising in analysing '
        'and writing case reports.'
    )
    question = f'''
**** Objective ****
As an experienced physician who specializes in medical documentation, your task is to produce a detailed and structured clinical case report in English based on the provided Electronic Health Record (EHR).

*** Methodology: ***
1. Carefully analyze all EHR data thoroughly
2. Strictly adhere to the chronological order of clinical events
3. Establish logical connections between symptoms, diagnoses, and treatments
4. Incorporate all relevant information without omissions
5. Use precise and appropriate medical terminology
6. Maintain an objective, factual, and professional tone

*** Format Requirements: ***
- A single coherent and fluid paragraph
- Concise yet comprehensive writing style (following PubMed case report conventions)
- Technical vocabulary conforming to medical standards
- Sentence structures typical of published case reports
- If information is missing, omit rather than speculate

**** Target Case ****

**Input EHR:**
{input_ehr}

**Generated Case Report:**
'''
    
    return role_description, question


# ============================================================================
# PROMPT CREATION - FEW-SHOT (FRENCH)
# ============================================================================

def create_fewshot_examples_fr(
    dev_ehrs: List[str],
    dev_reports: List[str],
    num_shots: int
) -> str:
    """
    Create few-shot examples from development set (French).
    
    Args:
        dev_ehrs (List[str]): List of simulated EHRs from dev set
        dev_reports (List[str]): List of case reports from dev set
        num_shots (int): Number of few-shot examples to include
        
    Returns:
        str: Formatted few-shot examples
    """
    fewshots = []
    for idx in range(min(num_shots, len(dev_ehrs))):
        example = f"""Exemple ({idx + 1}):
**DSE en entrée:**
{dev_ehrs[idx]}

**Rapport de cas généré:**
{dev_reports[idx]}

"""
        fewshots.append(example)
    
    return '\n'.join(fewshots)


def create_fewshot_prompt_fr(
    target_ehr: str,
    fewshot_examples: str
) -> Tuple[str, str]:
    """
    Create a structured prompt for few-shot medical case report generation (French).
    
    Args:
        target_ehr (str): Input EHR for target case
        fewshot_examples (str): Pre-formatted few-shot examples
        
    Returns:
        tuple: (role_description, full_question)
    """
    role_description = (
        'Vous êtes un expert dans le domaine médical, spécialisé dans l\'analyse '
        'et la rédaction de rapports de cas.'
    )
    question = f"""**** Objectif ****
En tant que médecin expérimenté spécialisé dans la documentation médicale, votre tâche consiste à produire un rapport de cas clinique détaillé et structuré en français basé sur le Dossier de Santé Électronique (DSE) fourni.

**** Méthodologie à suivre: ****
1. Analyser soigneusement toutes les données du DSE de manière approfondie
2. Respecter strictement l'ordre chronologique des événements cliniques
3. Établir des connexions logiques entre les symptômes, les diagnostics et les traitements
4. Incorporer toutes les informations pertinentes sans omissions
5. Utiliser une terminologie médicale précise et appropriée
6. Maintenir un ton objectif, factuel et professionnel

**** Exigences de format ****
- Un paragraphe unique cohérent et fluide
- Style d'écriture concis mais complet (suivant les conventions des rapports de cas PubMed)
- Vocabulaire technique conforme aux normes médicales
- Structures de phrases typiques des rapports de cas publiés
- Si des informations manquent, les omettre plutôt que de spéculer

**** Exemples Few-Shot ****
{fewshot_examples}

**** Cas cible ****

**DSE en entrée:**
{target_ehr}

**Rapport de cas généré:**
"""
    return role_description, question


# ============================================================================
# PROMPT CREATION - ZERO-SHOT (FRENCH)
# ============================================================================

def create_zeroshot_prompt_fr(input_ehr: str) -> Tuple[str, str]:
    """
    Create a structured prompt for zero-shot medical case report generation (French).
    
    Args:
        input_ehr (str): A simulated electronic health record
        
    Returns:
        tuple: (role_description, full_question)
    """
    role_description = (
        'Vous êtes un expert dans le domaine médical, spécialisé dans l\'analyse '
        'et la rédaction de rapports de cas.'
    )
    question = f'''
**** Objectif ****
En tant que médecin expérimenté spécialisé dans la documentation médicale, votre tâche consiste à produire un rapport de cas clinique détaillé et structuré en français basé sur le Dossier de Santé Électronique (DSE) fourni.

*** Méthodologie: ***
1. Analyser soigneusement toutes les données du DSE de manière approfondie
2. Respecter strictement l'ordre chronologique des événements cliniques
3. Établir des connexions logiques entre les symptômes, les diagnostics et les traitements
4. Incorporer toutes les informations pertinentes sans omissions
5. Utiliser une terminologie médicale précise et appropriée
6. Maintenir un ton objectif, factuel et professionnel

*** Exigences de format: ***
- Un paragraphe unique cohérent et fluide
- Style d'écriture concis mais complet (suivant les conventions des rapports de cas PubMed)
- Vocabulaire technique conforme aux normes médicales
- Structures de phrases typiques des rapports de cas publiés
- Si des informations manquent, les omettre plutôt que de spéculer

**** Cas cible ****

**DSE en entrée:**
{input_ehr}

**Rapport de cas généré:**
'''
    
    return role_description, question


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def validate_input_files(
    input_file: str,
    dev_file: Optional[str] = None,
    approach: str = 'zeroshot',
    language: str = 'English'
) -> None:
    """
    Validate that input files exist and have required columns.
    
    Args:
        input_file (str): Path to input CSV file
        dev_file (Optional[str]): Path to development set CSV file (required for few-shot)
        approach (str): Learning approach ('zeroshot' or 'fewshot')
        language (str): Language filter ('English' or 'French')
        
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If required columns are missing or dev_file not provided for few-shot
    """
    # Check if dev_file is required for few-shot
    if approach == 'fewshot' and dev_file is None:
        raise ValueError(
            "Development file (--dev_file) is required when using few-shot approach"
        )
    
    # Check input file existence
    if not Path(input_file).exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            f"Please ensure the file exists"
        )
    
    # Check input file columns
    df_input = pd.read_csv(input_file)
    required_input_cols = ['simulated_ehr', 'case_report', 'language']
    
    missing_cols = [col for col in required_input_cols if col not in df_input.columns]
    if missing_cols:
        raise ValueError(
            f"Input CSV is missing required columns: {missing_cols}\n"
            f"Required columns: {required_input_cols}\n"
            f"Found columns: {list(df_input.columns)}"
        )
    
    # Check if there are any rows for the specified language
    df_filtered = df_input[df_input['language'] == language]
    if len(df_filtered) == 0:
        raise ValueError(
            f"No rows found for language: {language}\n"
            f"Available languages in file: {df_input['language'].unique().tolist()}"
        )
    
    # Check dev file if provided (for few-shot)
    if dev_file is not None:
        if not Path(dev_file).exists():
            raise FileNotFoundError(
                f"Development file not found: {dev_file}\n"
                f"Please ensure the file exists"
            )
        
        df_dev = pd.read_csv(dev_file)
        required_dev_cols = ['simulated_ehr', 'case_report', 'language']
        
        missing_cols = [col for col in required_dev_cols if col not in df_dev.columns]
        if missing_cols:
            raise ValueError(
                f"Development CSV is missing required columns: {missing_cols}\n"
                f"Required columns: {required_dev_cols}\n"
                f"Found columns: {list(df_dev.columns)}"
            )
        
        # Check if there are any rows for the specified language in dev set
        df_dev_filtered = df_dev[df_dev['language'] == language]
        if len(df_dev_filtered) == 0:
            raise ValueError(
                f"No rows found for language '{language}' in development file\n"
                f"Available languages in dev file: {df_dev['language'].unique().tolist()}"
            )


def load_data(
    file_path: str,
    language: str = 'English'
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load EHRs, reports, and CR_ids from CSV file, filtered by language.
    
    Args:
        file_path (str): Path to CSV file
        language (str): Language to filter ('English' or 'French')
        
    Returns:
        Tuple[List[str], List[str], List[str]]: (simulated_ehr, reports, cr_ids)
    """
    df = pd.read_csv(file_path)
    
    # Filter by language
    df = df[df['language'] == language]
    
    ehrs = df['simulated_ehr'].tolist()
    reports = df['case_report'].tolist()
    cr_ids = df['cr_id'].tolist()
    
    return ehrs, reports, cr_ids


def process_batch_generation(
    input_file_path: str,
    llm: LLM,
    sampling_params: SamplingParams,
    output_file_path: str,
    batch_size: int,
    approach: str,
    language: str,
    dev_file_path: Optional[str] = None,
    num_shots: int = 3
) -> bool:
    """
    Process medical report generation in batches.
    
    Args:
        input_file_path (str): Path to input CSV file with EHRs
        llm (LLM): Initialized vLLM model
        sampling_params (SamplingParams): Sampling parameters for generation
        output_file_path (str): Path to save output CSV file
        batch_size (int): Number of samples to process in each batch
        approach (str): Learning approach ('zeroshot' or 'fewshot')
        language (str): Language for processing ('English' or 'French')
        dev_file_path (Optional[str]): Path to development set CSV file (for few-shot)
        num_shots (int): Number of few-shot examples to use (for few-shot)
        
    Returns:
        bool: True if successful
        
    Raises:
        Exception: If processing fails
    """
    try:
        # Select appropriate prompt functions based on language
        if language == 'French':
            create_fewshot_examples = create_fewshot_examples_fr
            create_fewshot_prompt = create_fewshot_prompt_fr
            create_zeroshot_prompt = create_zeroshot_prompt_fr
        else:  # English
            create_fewshot_examples = create_fewshot_examples_en
            create_fewshot_prompt = create_fewshot_prompt_en
            create_zeroshot_prompt = create_zeroshot_prompt_en
        
        # Initialize few-shot examples if needed
        fewshot_examples = None
        if approach == 'fewshot':
            if dev_file_path is None:
                raise ValueError("Development file is required for few-shot approach")
            
            print(f"Loading development set from: {dev_file_path}")
            dev_ehrs, dev_reports, _ = load_data(dev_file_path, language=language)
            
            print(f"Creating {num_shots} few-shot examples in {language}...")
            fewshot_examples = create_fewshot_examples(
                dev_ehrs,
                dev_reports,
                num_shots
            )
        
        # Load input data
        print(f"Loading input data from: {input_file_path}")
        print(f"Filtering for language: {language}")
        input_EHRs, reference_reports, cr_ids = load_data(input_file_path, language=language)
        
        total_samples = len(input_EHRs)
        print(f"Total samples to process ({language}): {total_samples}")
        
        if total_samples == 0:
            print(f"Warning: No samples found in input file for language: {language}")
            return False
        
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'CR_id',
                'language',
                'simulated_ehr',
                'reference_report',
                'generated_report',
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
                EHR_batch = input_EHRs[batch_idx:batch_end]
                reference_batch = reference_reports[batch_idx:batch_end]
                cr_id_batch = cr_ids[batch_idx:batch_end]
                
                # Create prompts for batch based on approach
                all_messages = []
                for ehr in EHR_batch:
                    if approach == 'fewshot':
                        role, query = create_fewshot_prompt(ehr, fewshot_examples)
                    else:  # zeroshot
                        role, query = create_zeroshot_prompt(ehr)
                    
                    chat_message = [
                        {"role": "system", "content": role},
                        {"role": "user", "content": query},
                    ]
                    all_messages.append(chat_message)
                
                # Generate reports
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
                                'language': language,
                                'simulated_ehr': EHR_batch[j],
                                'reference_report': reference_batch[j],
                                'generated_report': generated_text,
                            }
                            
                            csv_writer.writerow(instance)
                    
                    except Exception as e:
                        print(f"Error processing batch {current_batch}: {str(e)}")
                        raise
        
        print(f"\nResults saved to: {output_file_path}")
        print(f"Successfully processed {total_samples} samples in {language}")
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
        description='Medical Case Report Generator using vLLM (Multilingual)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Approach Selection
    approach_group = parser.add_argument_group('Approach Selection')
    approach_group.add_argument(
        '--approach',
        type=str,
        choices=['zeroshot', 'fewshot'],
        default='zeroshot',
        help='Prompting approach: zeroshot or fewshot'
    )
    
    # Language Selection
    language_group = parser.add_argument_group('Language Selection')
    language_group.add_argument(
        '--language',
        type=str,
        choices=['English', 'French'],
        default='English',
        help='Language for processing (English or French)'
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
        default=0.85,
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
        default=12000,
        help='Maximum tokens to generate'
    )
    
    # Processing Parameters
    processing_group = parser.add_argument_group('Processing Parameters')
    processing_group.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Number of samples per batch'
    )
    processing_group.add_argument(
        '--num_shots',
        type=int,
        default=3,
        help='Number of few-shot examples to include in prompt (only for fewshot approach)'
    )
    
    # Input/Output Paths
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        '--input_file',
        type=str,
        default='./data/processed/test/case_reports.csv',
        help='Path to input CSV file'
    )
    io_group.add_argument(
        '--dev_file',
        type=str,
        default='./data/processed/dev/case_reports.csv',
        help='Path to development set CSV file (required for fewshot approach)'
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
        default=None,
        help='Output file name (without .csv extension). If not provided, will use generated_reports_{approach}_{language}'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set default output name based on approach and language if not provided
    if args.output_name is None:
        args.output_name = f'generated_reports_{args.approach}_{args.language}'
    
    print("=" * 70)
    print(f"Medical Case Report Generator ({args.approach.upper()} Learning - {args.language})")
    print("=" * 70)
    
    try:
        # Validate input files
        dev_file_for_validation = args.dev_file if args.approach == 'fewshot' else None
        validate_input_files(
            args.input_file,
            dev_file_for_validation,
            args.approach,
            args.language
        )
        
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
        print(f"  Approach: {args.approach}")
        print(f"  Language: {args.language}")
        print(f"  Model: {args.model_name}")
        print(f"  Max model length: {args.max_model_len}")
        print(f"  Batch size: {args.batch_size}")
        if args.approach == 'fewshot':
            print(f"  Number of few-shot examples: {args.num_shots}")
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
        if args.approach == 'fewshot':
            print(f"Development file: {args.dev_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Output file: {output_file_path}")
        
        # Process generation
        dev_file_for_processing = args.dev_file if args.approach == 'fewshot' else None
        success = process_batch_generation(
            args.input_file,
            llm,
            sampling_params,
            output_file_path,
            args.batch_size,
            args.approach,
            args.language,
            dev_file_for_processing,
            args.num_shots
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
