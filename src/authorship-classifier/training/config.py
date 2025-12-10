"""
Configuration for binary classification (machine vs human text).
"""

# Model settings
DEFAULT_MODEL = 'EuroBERT/EuroBERT-210m'
DEFAULT_NUM_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_LENGTH = 5000 
DEFAULT_WARMUP_STEPS = 0

# Classification settings
NUM_LABELS = 2
LABEL_NAMES = ['machine', 'human']

# Training settings
GRADIENT_CLIP_NORM = 1.0
EVAL_BATCH_SIZE = 32

# Output settings
CHECKPOINT_DIR = 'best_model'
RESULTS_FILES = {
    'training': 'training_results.csv',
    'test_predictions': 'test_predictions.csv',
    'test_misclassified': 'test_misclassified.csv',
    'test_metrics': 'test_metrics.csv',
    'final_summary': 'final_summary.csv'
}
