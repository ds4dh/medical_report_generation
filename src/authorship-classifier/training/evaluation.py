"""
Evaluation functions for binary classification.
"""

import torch
import logging
import numpy as np
from typing import Tuple, List
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from config import LABEL_NAMES


logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    dataloader,
    device,
    dataset_name: str = ""
) -> Tuple[float, float, float, float, List[int], List[np.ndarray]]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        dataset_name: Name for logging
        
    Returns:
        (accuracy, precision, recall, f1, predictions, probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Prepare model arguments
            model_args = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            
            # Conditionally add token_type_ids
            if 'token_type_ids' in batch:
                model_args['token_type_ids'] = batch['token_type_ids'].to(device)
            
            # Forward pass
            outputs = model(**model_args)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    # Print results
    _print_evaluation_results(
        dataset_name, accuracy, precision, recall, f1,
        all_labels, all_preds
    )
    
    return accuracy, precision, recall, f1, all_preds, all_probs


def _print_evaluation_results(
    dataset_name: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    true_labels: List[int],
    predictions: List[int]
) -> None:
    """Print detailed evaluation results."""
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{dataset_name} RESULTS")
    logger.info(f"{'=' * 80}")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                Predicted")
    logger.info(f"              Machine  Human")
    logger.info(f"Actual Machine  {cm[0][0]:6d}  {cm[0][1]:5d}")
    logger.info(f"       Human    {cm[1][0]:6d}  {cm[1][1]:5d}")
    
    logger.info(f"\nClassification Report:")
    report = classification_report(
        true_labels, predictions,
        target_names=[f'{LABEL_NAMES[0].capitalize()} (0)', 
                     f'{LABEL_NAMES[1].capitalize()} (1)']
    )
    logger.info(f"\n{report}")


def analyze_misclassifications(
    true_labels: List[int],
    predictions: List[int]
) -> dict:
    """
    Analyze misclassification patterns.
    
    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        
    Returns:
        Dictionary with misclassification statistics
    """
    correct = [t == p for t, p in zip(true_labels, predictions)]
    incorrect = [not c for c in correct]
    
    machine_as_human = sum(
        (t == 0 and p == 1) for t, p in zip(true_labels, predictions)
    )
    human_as_machine = sum(
        (t == 1 and p == 0) for t, p in zip(true_labels, predictions)
    )
    
    total_incorrect = sum(incorrect)
    
    analysis = {
        'total_samples': len(true_labels),
        'correct': sum(correct),
        'incorrect': total_incorrect,
        'accuracy': sum(correct) / len(true_labels),
        'machine_as_human': machine_as_human,
        'human_as_machine': human_as_machine
    }
    
    if total_incorrect > 0:
        analysis['machine_as_human_pct'] = (machine_as_human / total_incorrect) * 100
        analysis['human_as_machine_pct'] = (human_as_machine / total_incorrect) * 100
    
    return analysis
