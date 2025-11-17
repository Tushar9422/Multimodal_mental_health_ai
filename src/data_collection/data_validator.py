#!/usr/bin/env python3
"""
Data validation and quality control for multimodal datasets.
Updated to match RAVDESS (audio), FER-2013 (facial), and text datasets.
"""

import json
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import get_project_logger, ensure_dir

logger = get_project_logger("data_validator")
config = get_config()

class DataValidator:
    """Validate and analyze collected multimodal data."""
    
    def __init__(self):
        self.config = config
        self.validation_results = {}
        
        print("🔍 Data Validator Initialized")
        print("="*60)
    
    def validate_audio_data(self):
        """Validate RAVDESS audio dataset."""
        print("\n🎵 Validating Audio Data (RAVDESS)...")
        
        metadata_file = config.DATA_DIR / "metadata" / "audio_metadata.json"
        if not metadata_file.exists():
            print("   ❌ Audio metadata not found")
            logger.error("Audio metadata not found")
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        audio_summary = metadata['summary']
        audio_samples = metadata.get('samples', [])
        
        # Validation checks
        validation_results = {
            'total_samples': audio_summary['total_samples'],
            'emotion_distribution': audio_summary['emotions'],
            'dataset_distribution': audio_summary.get('datasets', {}),
            'issues': []
        }
        
        print(f"   📊 Total samples: {audio_summary['total_samples']}")
        
        # Check minimum samples per emotion
        min_samples = 50
        for emotion, count in audio_summary['emotions'].items():
            if count < min_samples:
                validation_results['issues'].append(
                    f"Low sample count for {emotion}: {count} (minimum: {min_samples})"
                )
                print(f"   ⚠️  {emotion}: {count} samples (below minimum)")
            else:
                print(f"   ✅ {emotion}: {count} samples")
        
        # Check file existence (sample check)
        missing_files = 0
        check_limit = min(100, len(audio_samples))
        
        for sample in audio_samples[:check_limit]:
            if 'processed_path' in sample:
                if not Path(sample['processed_path']).exists():
                    missing_files += 1
        
        if missing_files > 0:
            validation_results['issues'].append(
                f"Missing {missing_files}/{check_limit} processed audio files"
            )
            print(f"   ⚠️  Missing {missing_files} processed files (checked {check_limit})")
        else:
            print(f"   ✅ All checked files exist ({check_limit} samples)")
        
        self.validation_results['audio'] = validation_results
        
        if len(validation_results['issues']) == 0:
            print("   ✅ Audio validation passed!")
        else:
            print(f"   ⚠️  Audio validation: {len(validation_results['issues'])} issues found")
        
        return True
    
    def validate_facial_data(self):
        """Validate FER-2013 facial expression dataset."""
        print("\n👤 Validating Facial Data (FER-2013)...")
        
        metadata_file = config.DATA_DIR / "metadata" / "facial_metadata.json"
        if not metadata_file.exists():
            print("   ❌ Facial metadata not found")
            logger.error("Facial metadata not found")
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        facial_summary = metadata['summary']
        facial_samples = metadata.get('samples', [])
        
        validation_results = {
            'total_samples': facial_summary['total_samples'],
            'emotion_distribution': facial_summary['emotions'],
            'split_distribution': facial_summary.get('splits', {}),
            'issues': []
        }
        
        print(f"   📊 Total samples: {facial_summary['total_samples']}")
        
        # Check emotion balance
        emotion_counts = facial_summary['emotions']
        if emotion_counts:
            max_count = max(emotion_counts.values())
            min_count = min(emotion_counts.values())
            
            for emotion, count in emotion_counts.items():
                print(f"   📈 {emotion}: {count} samples")
            
            if max_count / min_count > 10:  # Imbalance threshold
                validation_results['issues'].append(
                    f"Emotion imbalance detected. Max: {max_count}, Min: {min_count}"
                )
                print(f"   ⚠️  Class imbalance: {max_count}/{min_count} ratio")
            else:
                print(f"   ✅ Emotion distribution balanced")
        
        # Check train/test split
        splits = facial_summary.get('splits', {})
        if splits:
            print(f"\n   Split distribution:")
            for split, count in splits.items():
                print(f"      {split}: {count} samples")
        
        # Check image quality (sample check)
        corrupted_images = 0
        check_limit = min(100, len(facial_samples))
        
        for sample in facial_samples[:check_limit]:
            if 'processed_path' in sample:
                try:
                    img_path = Path(sample['processed_path'])
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is None:
                            corrupted_images += 1
                        elif img.shape[:2] != config.IMAGE_SIZE:
                            corrupted_images += 1
                except:
                    corrupted_images += 1
        
        if corrupted_images > 0:
            validation_results['issues'].append(
                f"Found {corrupted_images}/{check_limit} corrupted/invalid images"
            )
            print(f"   ⚠️  {corrupted_images} corrupted images (checked {check_limit})")
        else:
            print(f"   ✅ All checked images valid ({check_limit} samples)")
        
        self.validation_results['facial'] = validation_results
        
        if len(validation_results['issues']) == 0:
            print("   ✅ Facial validation passed!")
        else:
            print(f"   ⚠️  Facial validation: {len(validation_results['issues'])} issues found")
        
        return True
    
    def validate_text_data(self):
        """Validate text sentiment dataset."""
        print("\n💬 Validating Text Data...")
        
        metadata_file = config.DATA_DIR / "metadata" / "text_metadata.json"
        if not metadata_file.exists():
            print("   ❌ Text metadata not found")
            logger.error("Text metadata not found")
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        text_summary = metadata['summary']
        text_samples = metadata.get('samples', [])
        
        validation_results = {
            'total_samples': text_summary['total_samples'],
            'category_distribution': text_summary['categories'],
            'dataset_distribution': text_summary.get('datasets', {}),
            'feature_stats': text_summary.get('feature_stats', {}),
            'issues': []
        }
        
        print(f"   📊 Total samples: {text_summary['total_samples']}")
        
        # Check dataset sources
        datasets = text_summary.get('datasets', {})
        print(f"\n   Dataset sources:")
        for dataset, count in datasets.items():
            print(f"      {dataset}: {count} samples")
        
        # Check category distribution
        categories = text_summary['categories']
        print(f"\n   Category distribution:")
        for category, count in sorted(categories.items()):
            print(f"      {category}: {count} samples")
        
        # Check text quality
        empty_texts = 0
        short_texts = 0
        
        check_limit = min(1000, len(text_samples))
        for sample in text_samples[:check_limit]:
            text = sample.get('text', '')
            if not text or len(text.strip()) == 0:
                empty_texts += 1
            elif len(text.split()) < 3:
                short_texts += 1
        
        if empty_texts > 0:
            validation_results['issues'].append(f"Found {empty_texts} empty texts")
            print(f"   ⚠️  {empty_texts} empty texts")
        
        if short_texts > 0:
            validation_results['issues'].append(f"Found {short_texts} very short texts")
            print(f"   ⚠️  {short_texts} very short texts")
        
        if empty_texts == 0 and short_texts == 0:
            print(f"   ✅ All checked texts valid ({check_limit} samples)")
        
        # Check category diversity
        if len(categories) < 3:
            validation_results['issues'].append("Insufficient category diversity")
            print(f"   ⚠️  Only {len(categories)} categories found")
        else:
            print(f"   ✅ Good category diversity: {len(categories)} categories")
        
        # Check processed files exist
        processed_csv = config.DATA_DIR / "text" / "processed" / "all_processed_texts.csv"
        if processed_csv.exists():
            print(f"   ✅ Processed CSV exists: {processed_csv}")
        else:
            validation_results['issues'].append("Processed CSV file not found")
            print(f"   ⚠️  Processed CSV not found")
        
        self.validation_results['text'] = validation_results
        
        if len(validation_results['issues']) == 0:
            print("   ✅ Text validation passed!")
        else:
            print(f"   ⚠️  Text validation: {len(validation_results['issues'])} issues found")
        
        return True
    
    def create_data_summary_report(self):
        """Create comprehensive data summary report with visualizations."""
        print("\n📊 Creating Data Summary Report...")
        
        # Create visualization directory
        viz_dir = ensure_dir(config.DATA_DIR / "visualizations")
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create summary plots
        fig = plt.figure(figsize=(18, 12))
        
        # Audio emotion distribution
        if 'audio' in self.validation_results:
            ax1 = plt.subplot(2, 3, 1)
            emotions = list(self.validation_results['audio']['emotion_distribution'].keys())
            counts = list(self.validation_results['audio']['emotion_distribution'].values())
            
            ax1.bar(emotions, counts, color='skyblue', edgecolor='navy')
            ax1.set_title('Audio Emotion Distribution (RAVDESS)', fontweight='bold')
            ax1.set_xlabel('Emotions')
            ax1.set_ylabel('Sample Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(counts):
                ax1.text(i, v + 5, str(v), ha='center', va='bottom')
        
        # Facial emotion distribution
        if 'facial' in self.validation_results:
            ax2 = plt.subplot(2, 3, 2)
            emotions = list(self.validation_results['facial']['emotion_distribution'].keys())
            counts = list(self.validation_results['facial']['emotion_distribution'].values())
            
            ax2.bar(emotions, counts, color='lightcoral', edgecolor='darkred')
            ax2.set_title('Facial Emotion Distribution (FER-2013)', fontweight='bold')
            ax2.set_xlabel('Emotions')
            ax2.set_ylabel('Sample Count')
            ax2.tick_params(axis='x', rotation=45)
            
            for i, v in enumerate(counts):
                ax2.text(i, v + 100, str(v), ha='center', va='bottom')
        
        # Text category distribution
        if 'text' in self.validation_results:
            ax3 = plt.subplot(2, 3, 3)
            categories = list(self.validation_results['text']['category_distribution'].keys())
            counts = list(self.validation_results['text']['category_distribution'].values())
            
            ax3.bar(categories, counts, color='lightgreen', edgecolor='darkgreen')
            ax3.set_title('Text Category Distribution', fontweight='bold')
            ax3.set_xlabel('Categories')
            ax3.set_ylabel('Sample Count')
            ax3.tick_params(axis='x', rotation=45)
            
            for i, v in enumerate(counts):
                ax3.text(i, v + 50, str(v), ha='center', va='bottom')
        
        # Dataset distribution pie charts
        if 'audio' in self.validation_results:
            ax4 = plt.subplot(2, 3, 4)
            datasets = self.validation_results['audio'].get('dataset_distribution', {'RAVDESS': self.validation_results['audio']['total_samples']})
            ax4.pie(datasets.values(), labels=datasets.keys(), autopct='%1.1f%%', startangle=90)
            ax4.set_title('Audio Dataset Sources', fontweight='bold')
        
        if 'facial' in self.validation_results:
            ax5 = plt.subplot(2, 3, 5)
            splits = self.validation_results['facial'].get('split_distribution', {})
            if splits:
                ax5.pie(splits.values(), labels=splits.keys(), autopct='%1.1f%%', startangle=90)
                ax5.set_title('Facial Data Splits (Train/Test)', fontweight='bold')
        
        if 'text' in self.validation_results:
            ax6 = plt.subplot(2, 3, 6)
            datasets = self.validation_results['text'].get('dataset_distribution', {})
            if datasets:
                ax6.pie(datasets.values(), labels=datasets.keys(), autopct='%1.1f%%', startangle=90)
                ax6.set_title('Text Dataset Sources', fontweight='bold')
        
        plt.suptitle('Multimodal Mental Health Dataset Summary', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        summary_plot = viz_dir / "dataset_summary.png"
        plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
        print(f"   ✅ Summary plot saved: {summary_plot}")
        plt.close()
        
        # Save detailed report
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'validation_results': self.validation_results,
            'summary': {
                'total_audio_samples': self.validation_results.get('audio', {}).get('total_samples', 0),
                'total_facial_samples': self.validation_results.get('facial', {}).get('total_samples', 0),
                'total_text_samples': self.validation_results.get('text', {}).get('total_samples', 0),
                'total_issues': sum(len(result.get('issues', [])) for result in self.validation_results.values())
            }
        }
        
        # Calculate total samples
        total_samples = (report['summary']['total_audio_samples'] + 
                        report['summary']['total_facial_samples'] + 
                        report['summary']['total_text_samples'])
        report['summary']['total_samples'] = total_samples
        
        report_file = config.DATA_DIR / "data_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ Validation report saved: {report_file}")
        
        # Print summary table
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Samples: {total_samples:,}")
        print(f"  • Audio (RAVDESS):  {report['summary']['total_audio_samples']:,}")
        print(f"  • Facial (FER-2013): {report['summary']['total_facial_samples']:,}")
        print(f"  • Text:              {report['summary']['total_text_samples']:,}")
        print(f"\nTotal Issues Found: {report['summary']['total_issues']}")
        print("="*60)
        
        return report
    
    def run_complete_validation(self):
        """Run complete validation pipeline."""
        print("\n🚀 Starting Complete Data Validation")
        print("="*60)
        
        success = True
        
        # Validate each modality
        try:
            if not self.validate_audio_data():
                success = False
        except Exception as e:
            print(f"   ❌ Audio validation error: {e}")
            success = False
        
        try:
            if not self.validate_facial_data():
                success = False
        except Exception as e:
            print(f"   ❌ Facial validation error: {e}")
            success = False
        
        try:
            if not self.validate_text_data():
                success = False
        except Exception as e:
            print(f"   ❌ Text validation error: {e}")
            success = False
        
        # Create summary report
        try:
            report = self.create_data_summary_report()
        except Exception as e:
            print(f"   ❌ Report creation error: {e}")
            success = False
            report = None
        
        # Final verdict
        print("\n" + "="*60)
        if success and report and report['summary']['total_issues'] == 0:
            print("✅ ALL VALIDATIONS PASSED!")
            print("🎉 Your multimodal dataset is ready for Phase 3!")
        elif success:
            print("⚠️  VALIDATION COMPLETED WITH WARNINGS")
            print("   Review issues above before proceeding to Phase 3")
        else:
            print("❌ VALIDATION FAILED")
            print("   Fix errors before proceeding to Phase 3")
        print("="*60)
        
        return success and (report['summary']['total_issues'] == 0 if report else False)


def main():
    """Run data validation."""
    print("🔍 Multimodal Data Validation")
    print("="*60)
    
    validator = DataValidator()
    success = validator.run_complete_validation()
    
    if success:
        print("\n✅ Data validation completed successfully!")
        print("\n📁 Check these files:")
        print("   • data/data_validation_report.json")
        print("   • data/visualizations/dataset_summary.png")
        print("\n🚀 Ready to proceed to Phase 3: Model Training!")
    else:
        print("\n⚠️  Data validation completed with issues")
        print("   Review error messages and fix issues before training")
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
