#!/usr/bin/env python3
"""
CUDA Compatibility Test for TagGUI

This script tests whether your system has PyTorch installed with support
for newer CUDA compute capabilities, including Blackwell architecture (sm_120).

Run this script after installing TagGUI requirements to verify compatibility
with NVIDIA RTX PRO 6000 Blackwell Workstation Edition and similar GPUs.
"""

import sys

def test_pytorch_cuda_support():
    """Test PyTorch installation and CUDA support."""
    print("TagGUI CUDA Compatibility Test")
    print("=" * 50)
    
    # Test PyTorch import and version
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.version.cuda:
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("! PyTorch compiled without CUDA support")
            
    except ImportError:
        print("❌ PyTorch not installed!")
        print("Please install requirements.txt first:")
        print("  pip install -r requirements.txt")
        return False
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available on this system: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            compute_capability = f"{props.major}.{props.minor}"
            print(f"  Device {i}: {props.name}")
            print(f"    Compute capability: {compute_capability}")
            
            # Check for Blackwell support
            if props.major >= 12:
                print(f"    ✓ Supports Blackwell architecture (sm_120+)")
            elif props.major >= 9:
                print(f"    ✓ Supports Ada Lovelace/Hopper architecture")
            elif props.major >= 8:
                print(f"    ✓ Supports Ampere architecture") 
            elif props.major >= 7:
                print(f"    ✓ Supports Turing/Volta architecture")
            else:
                print(f"    ⚠ Older architecture (may have limited support)")
    else:
        print("No CUDA devices detected or CUDA drivers not installed")
    
    # Test basic tensor operations
    try:
        x = torch.randn(100, 100)
        y = torch.mm(x, x.t())
        print("✓ Basic CPU tensor operations work")
        
        if cuda_available:
            x_cuda = x.cuda()
            y_cuda = torch.mm(x_cuda, x_cuda.t())
            print("✓ Basic CUDA tensor operations work")
    except Exception as e:
        print(f"❌ Error with tensor operations: {e}")
        return False
    
    # Check PyTorch version for sm_120 support
    major, minor = map(int, torch.__version__.split('.')[:2])
    if major > 2 or (major == 2 and minor >= 8):
        print("✓ PyTorch version supports Blackwell architecture (sm_120)")
    else:
        print(f"⚠ PyTorch {torch.__version__} may not support sm_120")
        print("  Consider updating to PyTorch 2.8.0+ for full Blackwell support")
    
    print("=" * 50)
    print("Test completed successfully!")
    
    if cuda_available:
        print("Your system is ready for GPU-accelerated TagGUI operations.")
    else:
        print("TagGUI will run in CPU mode on this system.")
    
    return True

if __name__ == "__main__":
    success = test_pytorch_cuda_support()
    sys.exit(0 if success else 1)