#!/usr/bin/env python3

from data import data

def test_generators():
    print("Available generators:", data.list_generators())
    
    print("\n=== Testing SHM Generator ===")
    shm_gen = data.get('damped_shm')
    x, y = shm_gen.get_data(seq_len=4)
    print(f"SHM data shapes: x={x.size()}, y={y.size()}")
    
    print("\n=== Testing Sin Generator ===")
    sin_gen = data.get('sin', frequency=2.0, amplitude=0.8)
    x, y = sin_gen.get_data(seq_len=6)
    print(f"Sin data shapes: x={x.size()}, y={y.size()}")
    
    print("\n=== Testing Linear Generator ===")
    linear_gen = data.get('linear', slope=0.5, intercept=1.0)
    x, y = linear_gen.get_data(seq_len=8)
    print(f"Linear data shapes: x={x.size()}, y={y.size()}")
    
    print("\n=== Testing Exponential Generator ===")
    exp_gen = data.get('exp', growth_rate=0.05, initial_value=2.0)
    x, y = exp_gen.get_data(seq_len=3)
    print(f"Exp data shapes: x={x.size()}, y={y.size()}")
    
    print("\n=== Testing Bessel J_2 Generator ===")
    bessel_gen = data.get('bessel_j2', amplitude=1.0, x_scale=1.0, x_max=20)
    x, y = bessel_gen.get_data(seq_len=5)
    print(f"Bessel J_2 data shapes: x={x.size()}, y={y.size()}")
    
    # Test raw data range
    t, raw_data = bessel_gen.generate_raw_data()
    print(f"Bessel J_2 raw data range: [{raw_data.min():.4f}, {raw_data.max():.4f}]")
    
    print("\n✅ All tests passed!")

if __name__ == '__main__':
    test_generators()