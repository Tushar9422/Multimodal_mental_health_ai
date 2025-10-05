#!/usr/bin/env python3
"""
WSL2 Ubuntu project initialization for Multimodal Mental Health AI.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Initialize project in WSL2 Ubuntu environment."""
    print("🚀 Initializing WSL2 Multimodal Mental Health AI Project")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow Version: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not found. Please install requirements.txt")
        return False
    
    # Import configuration
    try:
        from src.config import get_config
        from src.utils import get_project_logger, check_wsl2_gpu
        
        config = get_config()
        print(f"✅ Configuration loaded: {config.__class__.__name__}")
    except ImportError as e:
        print(f"❌ Failed to import project modules: {e}")
        return False
    
    # Create directories
    print("\n📁 Creating project directories...")
    config.create_directories()
    
    # Setup TensorFlow GPU
    print("\n🔧 Configuring TensorFlow GPU for WSL2...")
    gpu_info = config.setup_tensorflow_gpu()
    
    # Check WSL2 GPU status
    print("\n🎯 WSL2 GPU Status:")
    wsl2_gpu = check_wsl2_gpu()
    
    if wsl2_gpu['available']:
        print(f"✅ WSL2 GPU Access: {wsl2_gpu['count']} device(s)")
        for device in wsl2_gpu['devices']:
            print(f"   🎮 {device}")
    else:
        print("❌ WSL2 GPU not accessible")
        if 'error' in wsl2_gpu:
            print(f"   Error: {wsl2_gpu['error']}")
    
    # Test imports
    print("\n🧪 Testing key library imports...")
    test_imports = [
        ('cv2', 'OpenCV'),
        ('librosa', 'Librosa'),
        ('transformers', 'Transformers'),
        ('streamlit', 'Streamlit'),
        ('fastapi', 'FastAPI'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib')
    ]
    
    successful_imports = 0
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
            successful_imports += 1
        except ImportError:
            print(f"   ❌ {name}")
    
    # Summary
    print(f"\n📊 Setup Summary:")
    print(f"   Libraries: {successful_imports}/{len(test_imports)} imported successfully")
    print(f"   GPU Support: {'✅' if gpu_info['gpu_available'] else '❌'}")
    print(f"   Environment: WSL2 Ubuntu")
    print(f"   Python: {sys.version.split()[0]}")
    
    if successful_imports == len(test_imports) and gpu_info['gpu_available']:
        print("\n🎉 WSL2 setup completed successfully!")
        print("\n🚀 Ready for Phase 2: Data Collection & Preparation")
        print("\nNext steps:")
        print("1. Start Jupyter Lab: jupyter lab --ip=0.0.0.0")
        print("2. Or open VS Code: code .")
        print("3. Begin building your multimodal AI models!")
        return True
    else:
        print("\n⚠️ Setup completed with some issues - check errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
