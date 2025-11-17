#!/usr/bin/env python3
"""
Master script for collecting all multimodal data.
Runs RAVDESS (audio), FER-2013 (facial), and text collection in sequence.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_collection.audio_collector_ravdess import RAVDESSAudioCollector
from src.data_collection.facial_collector_fer2013 import FER2013FacialCollector
from src.data_collection.text_collector import TextDataCollector
from src.data_collection.data_validator import DataValidator
from src.utils import Timer

def print_header(text, char="="):
    """Print a formatted header."""
    print("\n" + char * 70)
    print(text.center(70))
    print(char * 70)

def print_task_header(task_num, task_name):
    """Print task header."""
    print(f"\n{'='*70}")
    print(f"  Task {task_num}: {task_name}")
    print(f"{'='*70}")

def main():
    """Master data collection pipeline."""
    print_header("🚀 MULTIMODAL MENTAL HEALTH AI - DATA COLLECTION PIPELINE", "=")
    print("\nThis script will collect and process:")
    print("  1. 🎵 Audio emotion data (RAVDESS)")
    print("  2. 👤 Facial expression data (FER-2013)")
    print("  3. 💬 Text sentiment data (Multiple sources)")
    print("  4. 🔍 Data validation & quality control")
    
    input("\nPress ENTER to start data collection...")
    
    success_count = 0
    total_tasks = 4
    results = {}
    
    # ========================================
    # Task 1: Audio Data Collection (RAVDESS)
    # ========================================
    print_task_header(1, "Audio Data Collection (RAVDESS)")
    
    with Timer("Audio data collection"):
        try:
            audio_collector = RAVDESSAudioCollector()
            if audio_collector.collect_and_process():
                print("✅ Audio data collection completed successfully")
                success_count += 1
                results['audio'] = 'success'
            else:
                print("❌ Audio data collection failed")
                results['audio'] = 'failed'
        except Exception as e:
            print(f"❌ Audio collection error: {e}")
            results['audio'] = 'error'
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Task 2: Facial Data Collection (FER-2013)
    # ========================================
    print_task_header(2, "Facial Expression Data Collection (FER-2013)")
    
    with Timer("Facial data collection"):
        try:
            facial_collector = FER2013FacialCollector()
            if facial_collector.collect_and_process():
                print("✅ Facial data collection completed successfully")
                success_count += 1
                results['facial'] = 'success'
            else:
                print("❌ Facial data collection failed")
                results['facial'] = 'failed'
        except Exception as e:
            print(f"❌ Facial collection error: {e}")
            results['facial'] = 'error'
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Task 3: Text Data Collection
    # ========================================
    print_task_header(3, "Text Sentiment Data Collection")
    
    with Timer("Text data collection"):
        try:
            text_collector = TextDataCollector()
            if text_collector.collect_and_process():
                print("✅ Text data collection completed successfully")
                success_count += 1
                results['text'] = 'success'
            else:
                print("❌ Text data collection failed")
                results['text'] = 'failed'
        except Exception as e:
            print(f"❌ Text collection error: {e}")
            results['text'] = 'error'
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Task 4: Data Validation
    # ========================================
    print_task_header(4, "Data Validation & Quality Control")
    
    with Timer("Data validation"):
        try:
            validator = DataValidator()
            if validator.run_complete_validation():
                print("✅ Data validation passed")
                success_count += 1
                results['validation'] = 'success'
            else:
                print("⚠️  Data validation completed with issues")
                success_count += 0.5  # Partial success
                results['validation'] = 'partial'
        except Exception as e:
            print(f"❌ Data validation error: {e}")
            results['validation'] = 'error'
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Final Summary
    # ========================================
    print_header("📊 DATA COLLECTION PIPELINE SUMMARY", "=")
    
    print(f"\nTasks Completed: {success_count}/{total_tasks}")
    print("\nResults by Task:")
    print(f"  1. Audio (RAVDESS):  {results.get('audio', 'not run').upper()}")
    print(f"  2. Facial (FER-2013): {results.get('facial', 'not run').upper()}")
    print(f"  3. Text Sentiment:    {results.get('text', 'not run').upper()}")
    print(f"  4. Validation:        {results.get('validation', 'not run').upper()}")
    
    # Overall status
    print("\n" + "="*70)
    if success_count == total_tasks:
        print("🎉 ALL DATA COLLECTION TASKS COMPLETED SUCCESSFULLY!")
        print("\n✅ Your multimodal dataset is ready!")
        print("\n🚀 Next Steps:")
        print("   1. Review data/data_validation_report.json")
        print("   2. Check data/visualizations/dataset_summary.png")
        print("   3. Proceed to Phase 3: Individual Model Development")
        print("\nRecommended command:")
        print("   python3 initialize_phase3.py")
        
    elif success_count >= total_tasks * 0.75:
        print("✅ DATA COLLECTION MOSTLY SUCCESSFUL")
        print("\n⚠️  Some minor issues detected")
        print("\n📋 Action Items:")
        print("   1. Review validation report for warnings")
        print("   2. Fix any issues before proceeding to Phase 3")
        print("   3. Re-run failed tasks if needed")
        
    else:
        print("❌ DATA COLLECTION ENCOUNTERED SIGNIFICANT ISSUES")
        print("\n🔧 Troubleshooting Steps:")
        print("   1. Check error messages above")
        print("   2. Verify datasets are properly downloaded and placed")
        print("   3. Ensure all dependencies are installed")
        print("   4. Re-run individual collectors:")
        print("      • python3 src/data_collection/audio_collector_ravdess.py")
        print("      • python3 src/data_collection/facial_collector_fer2013.py")
        print("      • python3 src/data_collection/text_collector.py")
    
    print("="*70)
    
    # Create completion marker file
    if success_count >= total_tasks * 0.75:
        marker_file = Path("data/.phase2_complete")
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_text(f"Phase 2 completed: {success_count}/{total_tasks} tasks successful\n")
        print(f"\n✅ Phase 2 completion marker created: {marker_file}")
    
    return success_count >= total_tasks * 0.75


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
