"""
Inference and prediction functions.
"""

import os
import torch
import logging
import pandas as pd
from typing import Tuple
from torch.utils.data import DataLoader

from config import EVAL_BATCH_SIZE, RESULTS_FILES
from dataset import TextClassificationDataset
from evaluation import evaluate_model, analyze_misclassifications
from utils import save_predictions, save_metrics


logger = logging.getLogger(__name__)


def run_inference(
    model,
    test_data: pd.DataFrame,
    tokenizer,
    max_length: int,
    device: torch.device,
    output_dir: str
) -> Tuple:
    """
    Run inference on test set and save predictions.
    
    Args:
        model: Trained model
        test_data: Test DataFrame
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        device: Device to run on
        output_dir: Directory to save outputs
        
    Returns:
        (accuracy, precision, recall, f1, predictions_df)
    """
    logger.info("=" * 80)
    logger.info("INFERENCE ON TEST SET")
    logger.info("=" * 80)
    
    # Create test dataset and loader
    test_dataset = TextClassificationDataset(
        test_data['text'].tolist(),
        test_data['label'].tolist(),
        tokenizer,
        max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    
    # Run evaluation
    test_acc, test_precision, test_recall, test_f1, predictions, probabilities = evaluate_model(
        model, test_loader, device, "TEST SET"
    )
    
    # Prepare predictions dataframe
    predictions_df = _prepare_predictions_df(
        test_data, predictions, probabilities
    )
    
    # Save all predictions
    save_predictions(
        predictions_df,
        output_dir,
        RESULTS_FILES['test_predictions']
    )
    
    # Analyze and print results
    _print_inference_results(predictions_df)
    
    # Save misclassified samples
    misclassified = predictions_df[~predictions_df['correct']]
    if len(misclassified) > 0:
        save_predictions(
            misclassified,
            output_dir,
            RESULTS_FILES['test_misclassified']
        )
        _print_misclassification_analysis(misclassified)
    
    # Save detailed metrics
    metrics = {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'total_samples': len(predictions_df),
        'correct_predictions': sum(predictions_df['correct']),
        'incorrect_predictions': sum(~predictions_df['correct']),
        'machine_as_human': sum((predictions_df['label'] == 0) & 
                               (predictions_df['predicted_label'] == 1)),
        'human_as_machine': sum((predictions_df['label'] == 1) & 
                               (predictions_df['predicted_label'] == 0))
    }
    save_metrics(metrics, output_dir, RESULTS_FILES['test_metrics'])
    
    return test_acc, test_precision, test_recall, test_f1, predictions_df


def _prepare_predictions_df(
    test_data: pd.DataFrame,
    predictions: list,
    probabilities: list
) -> pd.DataFrame:
    """Prepare predictions dataframe with probabilities."""
    predictions_df = test_data.copy()
    predictions_df['predicted_label'] = predictions
    predictions_df['prob_machine'] = [prob[0] for prob in probabilities]
    predictions_df['prob_human'] = [prob[1] for prob in probabilities]
    predictions_df['correct'] = (
        predictions_df['label'] == predictions_df['predicted_label']
    )
    return predictions_df


def _print_inference_results(predictions_df: pd.DataFrame) -> None:
    """Print inference results."""
    total = len(predictions_df)
    correct = sum(predictions_df['correct'])
    incorrect = total - correct
    
    logger.info(f"\nInference Results:")
    logger.info(f"  Total samples: {total}")
    logger.info(f"  Correct: {correct} ({correct/total*100:.2f}%)")
    logger.info(f"  Incorrect: {incorrect} ({incorrect/total*100:.2f}%)")


def _print_misclassification_analysis(misclassified: pd.DataFrame) -> None:
    """Print misclassification analysis."""
    total_misc = len(misclassified)
    machine_as_human = sum(
        (misclassified['label'] == 0) & 
        (misclassified['predicted_label'] == 1)
    )
    human_as_machine = sum(
        (misclassified['label'] == 1) & 
        (misclassified['predicted_label'] == 0)
    )
    
    logger.info(f"\nMisclassification Analysis:")
    logger.info(f"  Total misclassified: {total_misc}")
    logger.info(f"  Machine as Human: {machine_as_human} "
               f"({machine_as_human/total_misc*100:.1f}%)")
    logger.info(f"  Human as Machine: {human_as_machine} "
               f"({human_as_machine/total_misc*100:.1f}%)")
