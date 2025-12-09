"""
Utility functions for data loading and preprocessing.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd

from config import TASK_COLUMNS, SUPPORTED_TASKS, SUPPORTED_LANGUAGES, SUPPORTED_APPROACHES


logger = logging.getLogger(__name__)


def remove_lines_after_condition(text: str) -> str:
    """
    Remove lines after conclusion/summary headers (for transcripts).
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    lines = text.split('\n')
    result = []
    
    for line in lines:
        if 'conclusion' in line.lower() and '###' in line:
            break
        if 'summary' in line.lower() and '###' in line:
            break
        result.append(line)
    
    return '\n'.join(result)


def validate_files(
    input_file: str,
    dev_file: Optional[str],
    approach: str,
    language: str,
    task: str
) -> None:
    """
    Validate input files and configuration.
    
    Args:
        input_file: Path to input CSV
        dev_file: Path to dev CSV (required for few-shot)
        approach: 'zeroshot' or 'fewshot'
        language: 'English' or 'French'
        task: 'case_report' or 'transcript'
    """
    # Validate config
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Invalid task: {task}. Use {SUPPORTED_TASKS}")
    if approach not in SUPPORTED_APPROACHES:
        raise ValueError(f"Invalid approach: {approach}. Use {SUPPORTED_APPROACHES}")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Invalid language: {language}. Use {SUPPORTED_LANGUAGES}")
    
    # Check dev file requirement
    if approach == 'fewshot' and dev_file is None:
        raise ValueError("Dev file required for few-shot approach")
    
    # Validate files
    _validate_csv(input_file, task, language, "Input file")
    if dev_file:
        _validate_csv(dev_file, task, language, "Dev file")


def _validate_csv(file_path: str, task: str, language: str, file_desc: str) -> None:
    """Validate a single CSV file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{file_desc} not found: {file_path}")
    
    df = pd.read_csv(file_path)
    required_cols = TASK_COLUMNS[task]['required']
    
    # For transcript task, check language-specific columns
    if task == 'transcript':
        lang_prefix = 'eng' if language == 'english' else 'fre'
        lang_cols = [f'{lang_prefix}_ehr', f'{lang_prefix}_report', 
                    f'{lang_prefix}_specialty', f'{lang_prefix}_report_type']
        missing = [col for col in lang_cols if col not in df.columns]
    else:
        # For case_report, check language column exists
        missing = [col for col in required_cols if col not in df.columns]
        if 'language' in required_cols and len(df[df['language'] == language]) == 0:
            available = df['language'].unique().tolist()
            raise ValueError(f"No {language} data in {file_desc}. Available: {available}")
    
    if missing:
        raise ValueError(f"{file_desc} missing columns: {missing}")
    
    logger.info(f"✓ {file_desc} validated")


def load_case_report_data(file_path: str, language: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load case report data.
    
    Args:
        file_path: Path to CSV
        language: Language filter
        
    Returns:
        (ehrs, reports, cr_ids)
    """
    df = pd.read_csv(file_path)
    df = df[df['language'] == language]
    
    ehrs = df['ehr'].fillna('').tolist()
    reports = df['case_report'].fillna('').tolist()
    cr_ids = df['id'].fillna('').tolist()
    
    return ehrs, reports, cr_ids


def load_transcript_data(
    file_path: str, 
    language: str
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load transcript data.
    
    Args:
        file_path: Path to CSV
        language: Language filter
        
    Returns:
        (ehrs, reports, specialties, report_types)
    """
    lang_prefix = 'eng' if language == 'english' else 'fre'
    
    df = pd.read_csv(file_path)
    
    ehrs = df[f'{lang_prefix}_ehr'].fillna('').tolist()
    reports = df[f'{lang_prefix}_report'].fillna('').tolist()
    specialties = df[f'{lang_prefix}_specialty'].fillna('').tolist()
    report_types = df[f'{lang_prefix}_report_type'].fillna('').tolist()
    
    return ehrs, reports, specialties, report_types
