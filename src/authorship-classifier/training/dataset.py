"""
PyTorch Dataset for text classification.
"""

import torch
from torch.utils.data import Dataset
from typing import List


class TextClassificationDataset(Dataset):
    """
    Dataset for binary text classification.
    
    Handles tokenization and prepares data for model input.
    Dynamically handles token_type_ids based on tokenizer capabilities.
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer, 
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of labels (0 or 1)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with input_ids, attention_mask, labels, and optionally token_type_ids
        """
        # Tokenize text
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Prepare item
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        # Conditionally add token_type_ids if tokenizer produces them
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
        
        return item
