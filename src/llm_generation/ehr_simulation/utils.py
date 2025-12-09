"""
Utility functions for data loading and validation.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from config import TASK_COLUMNS, SUPPORTED_TASKS, SUPPORTED_LANGUAGES


logger = logging.getLogger(__name__)


def validate_files(
    input_file: str,
    language: str,
    task: str
) -> None:
    """
    Validate input file and configuration.
    
    Args:
        input_file: Path to input CSV
        language: 'english' or 'french'
        task: 'case_report' or 'transcript'
    """
    # Validate config
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Invalid task: {task}. Use {SUPPORTED_TASKS}")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Invalid language: {language}. Use {SUPPORTED_LANGUAGES}")
    
    # Validate file
    _validate_csv(input_file, task, language, "Input file")


def _validate_csv(file_path: str, task: str, language: str, file_desc: str) -> None:
    """Validate a single CSV file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{file_desc} not found: {file_path}")
    
    df = pd.read_csv(file_path)
    required_cols = TASK_COLUMNS[task]['required']
    
    # For transcript task, check language-specific columns
    if task == 'transcript':
        lang_prefix = 'eng' if language == 'english' else 'fre'
        lang_cols = [f'{lang_prefix}_report', f'{lang_prefix}_specialty', 
                    f'{lang_prefix}_report_type']
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


def load_case_report_data(file_path: str, language: str) -> Tuple[List[str], List[str]]:
    """
    Load case report data.
    
    Args:
        file_path: Path to CSV
        language: Language filter
        
    Returns:
        (case_reports, cr_ids)
    """
    df = pd.read_csv(file_path)
    df = df[df['language'] == language]
    
    case_reports = df['case_report'].fillna('').tolist()
    cr_ids = df['id'].tolist()
    
    return case_reports, cr_ids


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
        (reports, specialties, report_types, mt_ids)
    """
    lang_prefix = 'eng' if language == 'english' else 'fre'
    
    df = pd.read_csv(file_path)
    
    reports = df[f'{lang_prefix}_report'].fillna('').tolist()
    specialties = df[f'{lang_prefix}_specialty'].fillna('').tolist()
    report_types = df[f'{lang_prefix}_report_type'].fillna('').tolist()
    mt_ids = df['id'].tolist()
    
    return reports, specialties, report_types, mt_ids
