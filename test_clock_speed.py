#!/usr/bin/env python3
"""Simple test script for ti.clock_speed_hz()"""

import gstaichi as ti

# Test with CUDA
print("Testing ti.clock_speed_hz() with CUDA backend...")
try:
    ti.init(arch=ti.cuda)
    clock_rate_hz = ti.clock_speed_hz()
    print(f"✓ CUDA clock speed: {clock_rate_hz / 1e6:.2f} MHz ({clock_rate_hz / 1e9:.3f} GHz)")
    
    if clock_rate_hz > 0:
        print("✓ Clock speed is valid (> 0)")
    else:
        print("✗ Clock speed should be > 0 for CUDA")
    
    # Verify it's a float
    if isinstance(clock_rate_hz, float):
        print("✓ Clock speed is returned as float")
    else:
        print(f"✗ Clock speed should be float, got {type(clock_rate_hz)}")
except Exception as e:
    print(f"✗ CUDA test failed: {e}")

# Test with CPU
print("\nTesting ti.clock_speed_hz() with CPU backend...")
try:
    ti.reset()
    ti.init(arch=ti.cpu)
    clock_rate_hz = ti.clock_speed_hz()
    print(f"✓ CPU clock speed: {clock_rate_hz} Hz")
    
    if clock_rate_hz == 0.0:
        print("✓ Clock speed correctly returns 0.0 for CPU (not supported)")
    else:
        print(f"✗ Clock speed should be 0.0 for CPU, got {clock_rate_hz}")
except Exception as e:
    print(f"✗ CPU test failed: {e}")

print("\nDone!")

