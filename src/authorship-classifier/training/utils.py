"""
Utility functions for data loading and validation.
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd


logger = logging.getLogger(__name__)


def validate_split_files(data_folder: str) -> None:
    """
    Validate that all required split files exist.
    
    Args:
        data_folder: Path to folder containing train/dev/test CSV files
        
    Raises:
        FileNotFoundError: If any required file is missing
    """
    required_files = ['train.csv', 'dev.csv', 'test.csv']
    
    for filename in required_files:
        file_path = os.path.join(data_folder, filename)
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    logger.info(f"✓ All required split files found in: {data_folder}")


def load_splits(data_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-saved train/dev/test splits from CSV files.
    
    Args:
        data_folder: Path to folder containing CSV files
        
    Returns:
        (train_data, dev_data, test_data) as pandas DataFrames
        
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If required columns are missing
    """
    logger.info("=" * 80)
    logger.info("LOADING DATA SPLITS")
    logger.info("=" * 80)
    
    # Validate files exist
    validate_split_files(data_folder)
    
    # Load splits
    train_data = pd.read_csv(os.path.join(data_folder, 'train.csv'))
    dev_data = pd.read_csv(os.path.join(data_folder, 'dev.csv'))
    test_data = pd.read_csv(os.path.join(data_folder, 'test.csv'))
    
    # Validate required columns
    required_columns = ['text', 'label']
    for name, df in [('train', train_data), ('dev', dev_data), ('test', test_data)]:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{name}.csv missing required columns: {missing}")
    
    # Print statistics
    _print_split_stats(train_data, dev_data, test_data)
    
    return train_data, dev_data, test_data


def _print_split_stats(
    train_data: pd.DataFrame, 
    dev_data: pd.DataFrame, 
    test_data: pd.DataFrame
) -> None:
    """Print statistics for loaded splits."""
    
    def get_label_counts(df):
        return sum(df['label'] == 0), sum(df['label'] == 1)
    
    train_machine, train_human = get_label_counts(train_data)
    dev_machine, dev_human = get_label_counts(dev_data)
    test_machine, test_human = get_label_counts(test_data)
    
    logger.info(f"\n✓ Train split:")
    logger.info(f"  - Total: {len(train_data)}")
    logger.info(f"  - Machine (0): {train_machine}")
    logger.info(f"  - Human (1): {train_human}")
    
    logger.info(f"\n✓ Dev split:")
    logger.info(f"  - Total: {len(dev_data)}")
    logger.info(f"  - Machine (0): {dev_machine}")
    logger.info(f"  - Human (1): {dev_human}")
    
    logger.info(f"\n✓ Test split:")
    logger.info(f"  - Total: {len(test_data)}")
    logger.info(f"  - Machine (0): {test_machine}")
    logger.info(f"  - Human (1): {test_human}")
    
    # Overall statistics
    total = len(train_data) + len(dev_data) + len(test_data)
    logger.info(f"\n{'=' * 80}")
    logger.info("TOTAL DATA")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total samples: {total}")
    logger.info(f"Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    logger.info(f"Dev:   {len(dev_data)} ({len(dev_data)/total*100:.1f}%)")
    logger.info(f"Test:  {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    logger.info(f"{'=' * 80}\n")


def save_predictions(
    predictions_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'predictions.csv'
) -> None:
    """
    Save predictions to CSV.
    
    Args:
        predictions_df: DataFrame with predictions
        output_dir: Output directory
        filename: Output filename
    """
    output_path = os.path.join(output_dir, filename)
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved predictions to: {output_path}")


def save_metrics(
    metrics: dict,
    output_dir: str,
    filename: str = 'metrics.csv'
) -> None:
    """
    Save metrics dictionary to CSV.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Output directory
        filename: Output filename
    """
    output_path = os.path.join(output_dir, filename)
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    logger.info(f"✓ Saved metrics to: {output_path}")
