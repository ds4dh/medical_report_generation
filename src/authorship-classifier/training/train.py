"""
Binary Classification Training Script

Train a transformer model to classify text as machine-generated or human-written.
Supports multilingual models like EuroBERT.

Usage:
    python train.py --data_folder ./splits --output_dir ./outputs
"""

import os
import sys
import torch
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForSequenceClassification

from config import *
from utils import load_splits, save_metrics
from trainer import train_classifier
from inference import run_inference


# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train binary classifier for machine vs human text',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument('--data_folder', required=True,
                        help='Folder containing train.csv, dev.csv, test.csv')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (auto-generated if not provided)')
    
    # Model configuration
    parser.add_argument('--model_name', default=DEFAULT_MODEL,
                        help='HuggingFace model identifier')
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH,
                        help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
                        help='Learning rate')
    
    # GPU configuration
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU ID to use (e.g., "0" or "0,1")')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    return parser.parse_args()


def setup_device(gpu_id: str, no_cuda: bool) -> torch.device:
    """
    Setup device for training.
    
    Args:
        gpu_id: GPU ID string
        no_cuda: Whether to disable CUDA
        
    Returns:
        torch.device
    """
    if no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        logger.info("Using device: CPU")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        device = torch.device('cuda')
        logger.info(f"Using device: CUDA (GPU {gpu_id})")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    return device


def create_output_dir(base_dir: str, model_name: str, num_epochs: int) -> str:
    """
    Create output directory with timestamp.
    
    Args:
        base_dir: Base directory (None for auto)
        model_name: Model name
        num_epochs: Number of epochs
        
    Returns:
        Output directory path
    """
    if base_dir is None:
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        model_str = model_name.replace('/', '-')
        base_dir = f"outputs/{model_str}_{num_epochs}ep_{timestamp}"
    
    os.makedirs(base_dir, exist_ok=True)
    logger.info(f"Output directory: {base_dir}")
    
    return base_dir


def save_final_summary(
    output_dir: str,
    model_name: str,
    args: argparse.Namespace,
    best_epoch: int,
    best_dev_acc: float,
    test_acc: float,
    test_precision: float,
    test_recall: float,
    test_f1: float,
    train_samples: int,
    dev_samples: int,
    test_samples: int,
    predictions_df
) -> None:
    """Save final summary of training and evaluation."""
    
    summary = {
        'model': model_name,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'best_epoch': best_epoch,
        'best_dev_accuracy': best_dev_acc,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'train_samples': train_samples,
        'dev_samples': dev_samples,
        'test_samples': test_samples,
        'test_correct': sum(predictions_df['correct']),
        'test_incorrect': sum(~predictions_df['correct'])
    }
    
    save_metrics(summary, output_dir, RESULTS_FILES['final_summary'])
    
    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Best Dev Accuracy:  {best_dev_acc:.4f} (Epoch {best_epoch})")
    logger.info(f"Test Accuracy:      {test_acc:.4f}")
    logger.info(f"Test Precision:     {test_precision:.4f}")
    logger.info(f"Test Recall:        {test_recall:.4f}")
    logger.info(f"Test F1 Score:      {test_f1:.4f}")
    
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nSaved files:")
    logger.info(f"  📁 {CHECKPOINT_DIR}/ (trained model and tokenizer)")
    logger.info(f"  📊 {RESULTS_FILES['training']} (per-epoch metrics)")
    logger.info(f"  📊 {RESULTS_FILES['final_summary']} (complete results)")
    logger.info(f"  📊 {RESULTS_FILES['test_predictions']} (all predictions with probabilities)")
    logger.info(f"  📊 {RESULTS_FILES['test_misclassified']} (misclassified samples)")
    logger.info(f"  📊 {RESULTS_FILES['test_metrics']} (detailed test metrics)")
    logger.info("=" * 80)


def main():
    """Main execution."""
    
    logger.info("=" * 80)
    logger.info("BINARY CLASSIFICATION TRAINING")
    logger.info("Machine vs Human Text Classification")
    logger.info("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup device
    device = setup_device(args.gpu_id, args.no_cuda)
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, args.model_name, args.num_epochs)
    
    try:
        # Load data splits
        train_data, dev_data, test_data = load_splits(args.data_folder)
        
        # Train model
        tokenizer, best_epoch, best_dev_acc, training_results = train_classifier(
            train_data,
            dev_data,
            output_dir,
            args.model_name,
            args.num_epochs,
            args.batch_size,
            args.learning_rate,
            args.max_length,
            device
        )
        
        # Load best model for test inference
        logger.info("\n" + "=" * 80)
        logger.info("LOADING BEST MODEL FOR TEST INFERENCE")
        logger.info("=" * 80)
        
        best_model_path = os.path.join(output_dir, CHECKPOINT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            best_model_path,
            trust_remote_code=True
        )
        model.to(device)
        logger.info(f"✓ Loaded best model from epoch {best_epoch}")
        
        # Run inference on test set
        test_acc, test_precision, test_recall, test_f1, predictions_df = run_inference(
            model,
            test_data,
            tokenizer,
            args.max_length,
            device,
            output_dir
        )
        
        # Save final summary
        save_final_summary(
            output_dir,
            args.model_name,
            args,
            best_epoch,
            best_dev_acc,
            test_acc,
            test_precision,
            test_recall,
            test_f1,
            len(train_data),
            len(dev_data),
            len(test_data),
            predictions_df
        )
        
        logger.info("\n✓ ALL COMPLETE!")
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
