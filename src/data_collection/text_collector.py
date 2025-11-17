#!/usr/bin/env python3
"""
Text data collection for mental health sentiment analysis.
Fixed version with proper file saving.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from textblob import TextBlob
from transformers import pipeline
import nltk
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import get_project_logger, ensure_dir, Timer

logger = get_project_logger("text_collector")
config = get_config()


class TextDataCollector:
    """Collect and preprocess text data for mental health analysis."""
    
    def __init__(self):
        self.config = config
        self.raw_text_dir = config.TEXT_DATA_DIR / "raw"
        self.processed_text_dir = config.TEXT_DATA_DIR / "processed"
        
        # Initialize sentiment analyzer with feedback
        print("\n🤖 Initializing sentiment analyzer (downloading model if first time)...")
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            print("✅ Sentiment analyzer initialized successfully!")
        except Exception as e:
            print(f"⚠️  Could not load sentiment analyzer: {e}")
            print("   Continuing without advanced sentiment analysis...")
            self.sentiment_analyzer = None
        
        # Mental health keywords
        self.mental_health_keywords = {
            'depression': ['depressed', 'depression', 'sad', 'hopeless', 'worthless'],
            'anxiety': ['anxious', 'anxiety', 'worried', 'panic', 'nervous'],
            'stress': ['stressed', 'stress', 'overwhelmed', 'pressure'],
            'positive': ['happy', 'joy', 'excited', 'grateful', 'blessed'],
            'negative': ['hate', 'angry', 'frustrated', 'annoyed', 'upset']
        }
        
        ensure_dir(self.raw_text_dir)
        ensure_dir(self.processed_text_dir)
        
        print(f"📁 Raw text directory: {self.raw_text_dir}")
        print(f"📁 Processed text directory: {self.processed_text_dir}")
    
    def check_available_datasets(self):
        """Check which datasets are available and provide download instructions."""
        print("\n🔍 Checking for available datasets...")
        
        datasets = {
            'mental_health_reddit.csv': {
                'name': 'Mental Health Reddit',
                'url': 'https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus',
                'found': False
            },
            'emotion_text.csv': {
                'name': 'Emotion in Text',
                'url': 'https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp',
                'found': False
            },
            'sentiment140.csv': {
                'name': 'Sentiment140',
                'url': 'https://www.kaggle.com/datasets/kazanova/sentiment140',
                'found': False,
                'optional': True
            }
        }
        
        found_count = 0
        for filename, info in datasets.items():
            filepath = self.raw_text_dir / filename
            if filepath.exists():
                info['found'] = True
                found_count += 1
                print(f"   ✅ Found: {info['name']} ({filename})")
            else:
                optional = " (optional)" if info.get('optional') else ""
                print(f"   ❌ Missing: {info['name']} ({filename}){optional}")
        
        if found_count == 0:
            print("\n⚠️  No datasets found in data/text/raw/")
            print("\n💡 To download datasets:")
            for filename, info in datasets.items():
                if not info.get('optional'):
                    print(f"\n{info['name']}:")
                    print(f"  1. Download from: {info['url']}")
                    print(f"  2. Place as: {self.raw_text_dir / filename}")
            print("\n📝 Or the script will generate synthetic data automatically")
        
        return found_count, datasets
    
    def clean_text(self, text):
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short texts
        if len(text) < 10:
            return ""
        
        return text
    
    def extract_text_features(self, text):
        """Extract comprehensive text features."""
        try:
            features = {}
            
            # Basic statistics
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            features['char_count'] = len(text)
            features['sentence_count'] = len(text.split('.'))
            
            # Sentiment analysis with TextBlob
            blob = TextBlob(text)
            features['textblob_polarity'] = blob.sentiment.polarity
            features['textblob_subjectivity'] = blob.sentiment.subjectivity
            
            # Advanced sentiment with Transformers (if available)
            if self.sentiment_analyzer:
                try:
                    sentiment_scores = self.sentiment_analyzer(text[:512])  # Limit for model
                    for score in sentiment_scores[0]:
                        features[f"roberta_{score['label'].lower()}"] = score['score']
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for text: {e}")
            
            # Mental health keyword analysis
            text_lower = text.lower()
            for category, keywords in self.mental_health_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                features[f'mh_keywords_{category}'] = count
            
            # Emotional indicators
            features['exclamation_count'] = text.count('!')
            features['question_count'] = text.count('?')
            features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
            
            # Readability (simplified)
            words = text.split()
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                features['avg_word_length'] = avg_word_length
            else:
                features['avg_word_length'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {}
    
    def categorize_mental_health_text(self, text, features):
        """Categorize text based on mental health indicators."""
        # Simple rule-based categorization
        depression_score = features.get('mh_keywords_depression', 0)
        anxiety_score = features.get('mh_keywords_anxiety', 0)
        stress_score = features.get('mh_keywords_stress', 0)
        positive_score = features.get('mh_keywords_positive', 0)
        
        polarity = features.get('textblob_polarity', 0)
        
        if depression_score > 0 or polarity < -0.3:
            return 'very_negative'
        elif anxiety_score > 0 or stress_score > 0 or polarity < -0.1:
            return 'negative'
        elif positive_score > 0 or polarity > 0.3:
            return 'positive'
        elif polarity > 0.1:
            return 'very_positive'
        else:
            return 'neutral'
    
    def process_mental_health_reddit(self):
        """Process Reddit mental health dataset."""
        print("\n📱 Processing Mental Health Reddit dataset...")
        
        reddit_file = self.raw_text_dir / "mental_health_reddit.csv"
        
        if not reddit_file.exists():
            print(f"   ⚠️  Dataset not found: {reddit_file}")
            logger.warning(f"Reddit dataset not found at {reddit_file}")
            return []
        
        try:
            df = pd.read_csv(reddit_file)
            print(f"   📊 Loaded {len(df)} rows from Reddit dataset")
            print(f"   📋 Columns: {df.columns.tolist()}")
            
            # Find text column
            text_column = None
            for col in ['text', 'post', 'comment', 'content', 'body', 'selftext']:
                if col in df.columns:
                    text_column = col
                    print(f"   ✅ Using '{col}' as text column")
                    break
            
            if text_column is None:
                print(f"   ❌ Could not find text column. Available: {df.columns.tolist()}")
                return []
            
        except Exception as e:
            print(f"   ❌ Error loading dataset: {e}")
            logger.error(f"Error loading Reddit dataset: {e}")
            return []
        
        processed_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Processing Reddit"):
            try:
                text = self.clean_text(str(row.get(text_column, '')))
                
                if not text:
                    continue
                
                features = self.extract_text_features(text)
                if not features:
                    continue
                
                category = self.categorize_mental_health_text(text, features)
                
                sample_data = {
                    'file_id': f"reddit_{idx:06d}",
                    'dataset': 'Reddit_MentalHealth',
                    'text': text,
                    'category': category,
                    'source': row.get('subreddit', 'unknown'),
                    **features
                }
                
                processed_data.append(sample_data)
                
            except Exception as e:
                logger.error(f"Error processing Reddit sample {idx}: {e}")
                continue
        
        print(f"   ✅ Successfully processed {len(processed_data)} Reddit samples")
        return processed_data
    
    def process_emotion_text_dataset(self):
        """Process general emotion text dataset."""
        print("\n😊 Processing Emotion Text dataset...")
        
        emotion_file = self.raw_text_dir / "emotion_text.csv"
        
        if not emotion_file.exists():
            print(f"   ⚠️  Dataset not found: {emotion_file}")
            logger.warning(f"Emotion text dataset not found at {emotion_file}")
            return []
        
        try:
            df = pd.read_csv(emotion_file)
            print(f"   📊 Loaded {len(df)} rows from Emotion dataset")
            print(f"   📋 Columns: {df.columns.tolist()}")
            
            # Validate required columns
            if 'text' not in df.columns:
                print(f"   ❌ 'text' column not found. Available: {df.columns.tolist()}")
                return []
            
        except Exception as e:
            print(f"   ❌ Error loading dataset: {e}")
            logger.error(f"Error loading Emotion dataset: {e}")
            return []
        
        processed_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Processing Emotion"):
            try:
                text = self.clean_text(str(row.get('text', '')))
                if not text:
                    continue
                
                features = self.extract_text_features(text)
                if not features:
                    continue
                
                # Get emotion label (could be 'emotion', 'label', 'sentiment')
                emotion_label = row.get('emotion') or row.get('label') or row.get('sentiment', 'neutral')
                
                sample_data = {
                    'file_id': f"emotion_{idx:06d}",
                    'dataset': 'EmotionText',
                    'text': text,
                    'category': str(emotion_label).lower(),
                    'original_label': str(emotion_label),
                    **features
                }
                
                processed_data.append(sample_data)
                
            except Exception as e:
                logger.error(f"Error processing emotion text {idx}: {e}")
                continue
        
        print(f"   ✅ Successfully processed {len(processed_data)} Emotion samples")
        return processed_data
    
    def process_sentiment140_dataset(self):
        """Process Sentiment140 Twitter dataset (optional - large)."""
        print("\n🐦 Processing Sentiment140 dataset...")
        
        sentiment_file = self.raw_text_dir / "sentiment140.csv"
        
        if not sentiment_file.exists():
            print(f"   ⚠️  Dataset not found: {sentiment_file} (optional)")
            return []
        
        try:
            # Sentiment140 has no header, specific column structure
            df = pd.read_csv(sentiment_file, encoding='latin-1', header=None)
            print(f"   📊 Loaded {len(df)} rows from Sentiment140")
            
            # Only process first 10,000 to keep it manageable
            max_samples = 10000
            df = df.head(max_samples)
            print(f"   ⚙️  Processing first {max_samples} samples")
            
        except Exception as e:
            print(f"   ❌ Error loading dataset: {e}")
            logger.error(f"Error loading Sentiment140: {e}")
            return []
        
        sentiment_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
        processed_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Processing Sentiment140"):
            try:
                text = self.clean_text(str(row[5]))  # Column 5 is text
                if not text:
                    continue
                
                features = self.extract_text_features(text)
                if not features:
                    continue
                
                sentiment = sentiment_map.get(row[0], 'neutral')
                
                sample_data = {
                    'file_id': f"sentiment140_{idx:06d}",
                    'dataset': 'Sentiment140',
                    'text': text,
                    'category': sentiment,
                    'original_sentiment': int(row[0]),
                    **features
                }
                
                processed_data.append(sample_data)
                
            except Exception as e:
                continue
        
        print(f"   ✅ Successfully processed {len(processed_data)} Sentiment140 samples")
        return processed_data
    
    def create_synthetic_examples(self, n_samples=1000):
        """Create synthetic text examples for mental health categories."""
        print(f"\n🎨 Generating {n_samples} synthetic text examples...")
        
        synthetic_templates = {
            'very_negative': [
                "I feel so hopeless and don't see any point in continuing",
                "Everything seems pointless and I can't find any motivation",
                "I'm completely overwhelmed and can't cope anymore",
                "The emptiness inside is unbearable and nothing brings me joy",
                "I feel isolated and alone, nobody understands what I'm going through",
            ],
            'negative': [
                "Having a really tough day and feeling quite stressed",
                "Things have been difficult lately and I'm struggling",
                "Feeling anxious about upcoming challenges",
                "Not doing great today, feeling a bit down",
                "The pressure is getting to me and I'm finding it hard to manage",
            ],
            'neutral': [
                "Just had a regular day, nothing special happened",
                "Working on various tasks and getting things done",
                "Had some meetings and completed normal activities",
                "Things are going okay, maintaining my normal schedule",
                "Another typical day, worked on projects and did errands",
            ],
            'positive': [
                "Had a good day today and accomplished my goals",
                "Feeling optimistic about upcoming opportunities",
                "Made some progress and feeling satisfied",
                "Things are looking up and I'm feeling more positive",
                "Enjoyed quality time with friends and family today",
            ],
            'very_positive': [
                "Feeling amazing and grateful for all the good things",
                "Had an incredible day full of joy and excitement",
                "So happy and blessed to have such wonderful experiences",
                "Life is beautiful and I feel blessed beyond measure",
                "Everything is falling into place and I couldn't be more thrilled",
            ]
        }
        
        synthetic_data = []
        samples_per_category = n_samples // len(synthetic_templates)
        
        for category, templates in synthetic_templates.items():
            for i in range(samples_per_category):
                text = np.random.choice(templates)
                features = self.extract_text_features(text)
                
                sample_data = {
                    'file_id': f"synthetic_{category}_{i:04d}",
                    'dataset': 'Synthetic',
                    'text': text,
                    'category': category,
                    'source': 'generated',
                    **features
                }
                
                synthetic_data.append(sample_data)
        
        print(f"   ✅ Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data
    
    def save_processed_samples(self, processed_data):
        """Save processed text samples to files."""
        print("\n💾 Saving processed text samples...")
        
        if not processed_data:
            print("   ⚠️  No data to save!")
            return False
        
        # Ensure processed directory exists
        ensure_dir(self.processed_text_dir)
        print(f"   📁 Directory: {self.processed_text_dir}")
        
        # Save as consolidated CSV for easy access
        csv_file = self.processed_text_dir / "all_processed_texts.csv"
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(processed_data)
            
            # Save to CSV
            df.to_csv(csv_file, index=False)
            print(f"   ✅ Saved CSV: {csv_file}")
            print(f"      📊 {len(df)} samples, {len(df.columns)} features")
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            print(f"   ❌ Failed to save CSV: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Also save as JSON for detailed feature preservation
        json_file = self.processed_text_dir / "all_processed_texts.json"
        
        try:
            with open(json_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            print(f"   ✅ Saved JSON: {json_file}")
            
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            print(f"   ❌ Failed to save JSON: {e}")
        
        # Save split files by dataset for easier management
        datasets = {}
        for sample in processed_data:
            dataset_name = sample['dataset']
            if dataset_name not in datasets:
                datasets[dataset_name] = []
            datasets[dataset_name].append(sample)
        
        print(f"\n   📊 Saving {len(datasets)} dataset splits...")
        for dataset_name, samples in datasets.items():
            split_file = self.processed_text_dir / f"{dataset_name.lower()}_processed.csv"
            try:
                df_split = pd.DataFrame(samples)
                df_split.to_csv(split_file, index=False)
                print(f"      ✅ {dataset_name}: {len(samples)} samples → {split_file.name}")
            except Exception as e:
                print(f"      ❌ {dataset_name}: Failed - {e}")
        
        print(f"\n   💾 All processed samples saved to: {self.processed_text_dir}")
        return True
    
    def create_text_metadata(self, processed_data):
        """Create text processing metadata."""
        metadata_file = config.DATA_DIR / "metadata" / "text_metadata.json"
        ensure_dir(metadata_file.parent)
        
        summary = {
            'total_samples': len(processed_data),
            'datasets': {},
            'categories': {},
            'feature_stats': {}
        }
        
        # Count by dataset and category
        for sample in processed_data:
            dataset = sample['dataset']
            category = sample['category']
            
            summary['datasets'][dataset] = summary['datasets'].get(dataset, 0) + 1
            summary['categories'][category] = summary['categories'].get(category, 0) + 1
        
        # Feature statistics
        numerical_features = ['text_length', 'word_count', 'textblob_polarity']
        for feature in numerical_features:
            values = [sample.get(feature, 0) for sample in processed_data if feature in sample]
            if values:
                summary['feature_stats'][feature] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump({
                'summary': summary,
                'samples': processed_data
            }, f, indent=2)
        
        logger.info(f"Text metadata saved to {metadata_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Text Dataset Summary:")
        print(f"  Total samples: {summary['total_samples']:,}")
        print(f"\n  Dataset distribution:")
        for dataset, count in summary['datasets'].items():
            print(f"    {dataset}: {count:,}")
        print(f"\n  Category distribution:")
        for category, count in sorted(summary['categories'].items()):
            print(f"    {category}: {count:,}")
        print("="*60)
        
        return summary
    
    def collect_and_process(self):
        """Main text data collection method."""
        print("\n🚀 Starting Text Data Collection and Processing")
        print("="*60)
        
        # Check available datasets
        found_count, datasets = self.check_available_datasets()
        
        all_processed_data = []
        
        # Process Reddit mental health data
        reddit_data = self.process_mental_health_reddit()
        all_processed_data.extend(reddit_data)
        
        # Process emotion text dataset
        emotion_data = self.process_emotion_text_dataset()
        all_processed_data.extend(emotion_data)
        
        # Process Sentiment140 (optional)
        sentiment140_data = self.process_sentiment140_dataset()
        all_processed_data.extend(sentiment140_data)
        
        # Generate synthetic examples
        synthetic_data = self.create_synthetic_examples(1000)
        all_processed_data.extend(synthetic_data)
        
        # Check if we have data
        if not all_processed_data:
            print("❌ No text data was processed")
            logger.error("No text data processed")
            return False
        
        print(f"\n✅ Total samples collected: {len(all_processed_data)}")
        
        # Debug information
        print(f"\n🔍 DEBUG INFO:")
        print(f"   Total samples to save: {len(all_processed_data)}")
        if all_processed_data:
            print(f"   First sample keys: {list(all_processed_data[0].keys())[:5]}...")
            print(f"   Sample text: {all_processed_data[0].get('text', 'N/A')[:50]}...")
            print(f"   Processed directory: {self.processed_text_dir}")
            print(f"   Directory exists: {self.processed_text_dir.exists()}")
        
        # Save processed samples (THIS IS THE KEY LINE)
        print(f"\n📝 Calling save_processed_samples()...")
        save_success = self.save_processed_samples(all_processed_data)
        
        if not save_success:
            print("⚠️  File saving had issues, but continuing...")
        
        # Create metadata
        summary = self.create_text_metadata(all_processed_data)
        
        print("\n✅ Text data collection completed successfully!")
        return True


def main():
    """Run text data collection."""
    print("💬 Text Sentiment Data Collection")
    print("="*60)
    
    collector = TextDataCollector()
    success = collector.collect_and_process()
    
    if success:
        print("\n✅ Text data collection completed!")
    else:
        print("\n❌ Text data collection failed")
        return False
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
