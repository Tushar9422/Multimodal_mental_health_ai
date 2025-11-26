#!/usr/bin/env python3
"""
Text preprocessing and tokenization for sentiment analysis.
Prepares data for DistilBERT fine-tuning.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import get_project_logger, ensure_dir

logger = get_project_logger("text_processor")
config = get_config()


class TextDataLoader:
    """
    Load and preprocess text data for sentiment analysis.
    
    Handles:
    - Loading from CSV (all_processed_texts.csv)
    - Label mapping (5 categories → numeric)
    - Train/validation split
    - Dataset statistics
    """
    
    def __init__(self, max_length=128):
        """
        Initialize text data loader.
        
        Args:
            max_length: Maximum sequence length for tokenization
        """
        self.max_length = max_length
        self.text_csv = config.DATA_DIR / "text" / "processed" / "all_processed_texts.csv"
        
        print(f"💬 Text Data Loader Initialized")
        print(f"   Max sequence length: {max_length}")
        print(f"   Source: {self.text_csv}")
    
    def load_data(self):
        """Load text data from CSV."""
        print("\n📂 Loading text data...")
        
        if not self.text_csv.exists():
            raise FileNotFoundError(f"Text CSV not found: {self.text_csv}")
        
        # Load CSV
        df = pd.read_csv(self.text_csv)
        
        print(f"✅ Loaded {len(df)} text samples")
        print(f"   Columns: {list(df.columns)}")
        
        # Check required columns
        required = ['text', 'category']
        if not all(col in df.columns for col in required):
            raise ValueError(f"CSV must contain columns: {required}")
        
        return df
    
    def create_label_mapping(self, df):
        """
        Create label mapping for categories.
        
        Expected categories (from your data validation):
        - very_negative
        - negative
        - neutral
        - positive
        - very_positive
        """
        # Get unique categories (sorted for consistency)
        categories = sorted(df['category'].unique())
        
        # Create mappings
        category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}
        
        print(f"\n📋 Label Mapping:")
        for cat, idx in category_to_idx.items():
            count = len(df[df['category'] == cat])
            print(f"   {idx}: {cat:15s} ({count:5d} samples)")
        
        return category_to_idx, idx_to_category, categories
    
    def prepare_dataset(self, df, category_to_idx, test_size=0.2, random_state=42):
        """
        Prepare train/validation split.
        
        Args:
            df: DataFrame with text and categories
            category_to_idx: Category to index mapping
            test_size: Validation split ratio
            random_state: Random seed for reproducibility
        """
        print(f"\n🔄 Preparing dataset (test_size={test_size})...")
        
        # Extract texts and labels
        texts = df['text'].values
        categories = df['category'].values
        labels = np.array([category_to_idx[cat] for cat in categories])
        
        # Split into train/validation
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels  # Maintain category distribution
        )
        
        print(f"✅ Dataset split complete:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Calculate class distribution
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        print(f"\n📊 Training set distribution:")
        for idx, count in zip(unique_train, counts_train):
            cat = self.idx_to_category[idx]
            pct = count / len(y_train) * 100
            print(f"   {cat:15s}: {count:5d} ({pct:5.1f}%)")
        
        return X_train, X_val, y_train, y_val
    
    def save_processed_data(self, X_train, X_val, y_train, y_val, 
                           category_to_idx, idx_to_category, categories):
        """Save processed data for training."""
        print("\n💾 Saving processed data...")
        
        output_dir = ensure_dir(config.DATA_DIR / "processed_features")
        
        # Save data splits
        np.save(output_dir / "X_text_train.npy", X_train)
        np.save(output_dir / "X_text_val.npy", X_val)
        np.save(output_dir / "y_text_train.npy", y_train)
        np.save(output_dir / "y_text_val.npy", y_val)
        
        # Save mappings
        metadata = {
            'category_to_idx': category_to_idx,
            'idx_to_category': idx_to_category,
            'categories': categories,
            'num_classes': len(categories),
            'max_length': self.max_length,
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }
        
        with open(output_dir / "text_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Data saved to: {output_dir}")
        print(f"   - X_text_train.npy, X_text_val.npy")
        print(f"   - y_text_train.npy, y_text_val.npy")
        print(f"   - text_metadata.json")


def prepare_text_dataset():
    """Prepare complete text dataset for training."""
    print("💬 Preparing Text Sentiment Dataset")
    print("="*60)
    
    # Initialize loader
    loader = TextDataLoader(max_length=128)
    
    # Load data
    df = loader.load_data()
    
    # Create label mapping
    category_to_idx, idx_to_category, categories = loader.create_label_mapping(df)
    loader.idx_to_category = idx_to_category  # Store for later use
    
    # Prepare dataset
    X_train, X_val, y_train, y_val = loader.prepare_dataset(
        df, category_to_idx, test_size=0.2
    )
    
    # Save processed data
    loader.save_processed_data(
        X_train, X_val, y_train, y_val,
        category_to_idx, idx_to_category, categories
    )
    
    print("\n✅ Text dataset preparation complete!")
    print("   Ready for DistilBERT fine-tuning")
    
    return True


if __name__ == "__main__":
    success = prepare_text_dataset()
    
    if success:
        print("\n✅ Text dataset ready for model training!")
    else:
        print("\n❌ Text dataset preparation failed")
        sys.exit(1)
