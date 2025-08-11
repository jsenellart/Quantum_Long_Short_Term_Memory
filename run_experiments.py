#!/usr/bin/env python3
import os

from QLSTM_v0 import run_experiment

def example_experiments():
    print("Running various experiment configurations...")

    # Example 1: Classical LSTM with SHM data
    print("\n" + "="*60)
    print("EXPERIMENT 1: Classical LSTM with SHM data")
    print("="*60)
    run_experiment(
        model_type='lstm',
        epochs=100,
        generator_name='damped_shm'
    )

    # Example 2: Default QLSTM with SHM data
    print("\n" + "="*60)
    print("EXPERIMENT 2:"
          " Default QLSTM with SHM data")
    print("="*60)
    run_experiment(epochs=100, generator_name='damped_shm')

    # Example 3: QLSTM with Sine wave data
    print("\n" + "="*60)
    print("EXPERIMENT 3: QLSTM with Sine wave data")
    print("="*60)
    run_experiment(
        generator_name='sin',
        model_type='qlstm',
        exp_name='QLSTM_TS_MODEL_SIN',
        epochs=10,
        frequency=2.0,
        amplitude=1.0
    )
    
    # Example 4: LSTM with Linear data, different hyperparameters
    print("\n" + "="*60)
    print("EXPERIMENT 4: LSTM with Linear data, larger hidden size")
    print("="*60)
    run_experiment(
        generator_name='linear',
        model_type='lstm',
        hidden_size=10,
        seq_length=8,
        batch_size=20,
        epochs=10,
        learning_rate=0.005,
        exp_name='LSTM_TS_MODEL_LINEAR_LARGE',
        slope=0.5,
        intercept=1.0
    )
    
    # Example 5: QLSTM with Exponential data, deeper VQC
    print("\n" + "="*60)
    print("EXPERIMENT 5: QLSTM with Exponential data, deep VQC")
    print("="*60)
    run_experiment(
        generator_name='exp',
        model_type='qlstm',
        vqc_depth=8,
        hidden_size=8,
        seq_length=6,
        epochs=10,
        exp_name='QLSTM_TS_MODEL_EXP_DEEP',
        growth_rate=0.05,
        initial_value=1.0
    )

def compare_models():
    """Compare QLSTM vs LSTM on the same dataset"""
    print("\n" + "="*60)
    print("MODEL COMPARISON: QLSTM vs LSTM on Sine Wave")
    print("="*60)
    
    common_params = {
        'generator_name': 'sin',
        'hidden_size': 6,
        'seq_length': 5,
        'epochs': 30,
        'batch_size': 15,
        'learning_rate': 0.01,
        'frequency': 1.5,
        'amplitude': 0.8,
        'seed': 42
    }
    
    print("Training QLSTM...")
    qlstm_model, qlstm_train_loss, qlstm_test_loss = run_experiment(
        model_type='qlstm',
        exp_name='COMPARE_QLSTM_SIN',
        vqc_depth=4,
        **common_params
    )
    
    print("\nTraining LSTM...")
    lstm_model, lstm_train_loss, lstm_test_loss = run_experiment(
        model_type='lstm',
        exp_name='COMPARE_LSTM_SIN',
        **common_params
    )
    
    print(f"\nFinal Results:")
    print(f"QLSTM - Train Loss: {qlstm_train_loss[-1]:.6f}, Test Loss: {qlstm_test_loss[-1]:.6f}")
    print(f"LSTM  - Train Loss: {lstm_train_loss[-1]:.6f}, Test Loss: {lstm_test_loss[-1]:.6f}")

if __name__ == '__main__':
    # Run individual examples
    example_experiments()
    
    # Compare models
    compare_models()
    
    print("\nAll experiments completed!")