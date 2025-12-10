"""
Training functions for binary classification.
"""

import os
import torch
import logging
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from config import *
from dataset import TextClassificationDataset
from evaluation import evaluate_model


logger = logging.getLogger(__name__)


def train_classifier(
    train_data: pd.DataFrame,
    dev_data: pd.DataFrame,
    output_dir: str,
    model_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    device: torch.device
) -> Tuple:
    """
    Train binary classifier.
    
    Args:
        train_data: Training DataFrame
        dev_data: Development DataFrame
        output_dir: Directory to save outputs
        model_name: HuggingFace model identifier
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_length: Maximum sequence length
        device: Device to train on
        
    Returns:
        (tokenizer, best_epoch, best_dev_acc, results_df)
    """
    logger.info("=" * 80)
    logger.info("INITIALIZING TRAINING")
    logger.info("=" * 80)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        trust_remote_code=True
    )
    model.to(device)
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        train_data['text'].tolist(),
        train_data['label'].tolist(),
        tokenizer,
        max_length
    )
    dev_dataset = TextClassificationDataset(
        dev_data['text'].tolist(),
        dev_data['label'].tolist(),
        tokenizer,
        max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=DEFAULT_WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Print configuration
    _print_training_config(
        model_name, num_epochs, batch_size, learning_rate,
        max_length, total_steps, device
    )
    
    # Training loop
    best_dev_acc = 0
    best_epoch = 0
    results = []
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logger.info(f"{'=' * 80}")
        
        # Training
        train_loss = _train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch + 1
        )
        
        # Evaluation on dev set
        dev_acc, dev_precision, dev_recall, dev_f1, _, _ = evaluate_model(
            model, dev_loader, device, "DEV SET"
        )
        
        # Save results
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'dev_accuracy': dev_acc,
            'dev_precision': dev_precision,
            'dev_recall': dev_recall,
            'dev_f1': dev_f1
        })
        
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Dev Accuracy: {dev_acc:.4f}")
        logger.info(f"  Dev F1: {dev_f1:.4f}")
        
        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch + 1
            model_save_path = os.path.join(output_dir, CHECKPOINT_DIR)
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logger.info(f"  ✓ New best model saved! (Accuracy: {dev_acc:.4f})")
    
    # Save training results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, RESULTS_FILES['training'])
    results_df.to_csv(results_path, index=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Best model: Epoch {best_epoch} (Dev Accuracy: {best_dev_acc:.4f})")
    
    return tokenizer, best_epoch, best_dev_acc, results_df


def _train_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    train_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Prepare model arguments
        model_args = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # Conditionally add token_type_ids
        if 'token_type_ids' in batch:
            model_args['token_type_ids'] = batch['token_type_ids'].to(device)
        
        # Forward pass
        model.zero_grad()
        outputs = model(**model_args)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
    
    return train_loss / len(train_loader)


def _print_training_config(
    model_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    total_steps: int,
    device: torch.device
) -> None:
    """Print training configuration."""
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Max length: {max_length}")
    logger.info(f"  Total training steps: {total_steps}")
    logger.info(f"  Gradient clip norm: {GRADIENT_CLIP_NORM}")
