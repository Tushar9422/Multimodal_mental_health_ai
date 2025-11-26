#!/usr/bin/env python3
"""Verify GPU setup for TensorFlow."""

import os

# Set environment FIRST
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np

print("="*60)
print("🔍 TensorFlow GPU Verification")
print("="*60)

print(f"\n1. TensorFlow version: {tf.__version__}")
print(f"2. Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"3. GPU devices: {tf.config.list_physical_devices('GPU')}")

# Test GPU computation
try:
    print("\n4. Testing GPU computation...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        result = c.numpy()
    print("   ✅ GPU computation successful!")
    
except Exception as e:
    print(f"   ❌ GPU computation failed: {e}")
    print("   💡 Recommendation: Use CPU mode for now")

print("\n" + "="*60)
