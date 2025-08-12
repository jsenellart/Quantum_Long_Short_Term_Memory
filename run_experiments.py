#!/usr/bin/env python3

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import os
import glob

from QLSTM_v0 import run_experiment

def example_lstm_experiments():
    print("Running various experiment configurations...")

    generators = ('damped_shm', 'sin', 'linear', 'exp', 'bessel_j2')

    results = []
    for idx, generator in enumerate(generators, start=1):
        print("\n" + "="*60)
        print(f"EXPERIMENT {idx}: Classical LSTM with {generator.upper()} data")
        print("="*60)
        model, train_loss, test_loss, exp_name, run_datetime, last_epoch = run_experiment(
            model_type='lstm',
            epochs=100,
            batch_size=32,
            hidden_size=5,
            generator_name=generator,
            exp_name=f'experiments/LSTM_TS_MODEL_{generator.upper()}_HIDDEN_5',
            fformat="png"
        )
        results.append((exp_name, run_datetime, generator, last_epoch))

        # Create the summary figure
    create_summary_figure(results, filename="experiments/summary_lstm_hidden_5_experiments.png")


def create_summary_figure(results, filename="experiments/summary_lstm_experiments.png"):
    """
        results: list of tuples (exp_name, run_datetime, generator, last_epoch)
        filename: output file name (png)
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    for col, (exp_name, run_datetime, generator, last_epoch) in enumerate(results):
        base_exp_name = os.path.basename(os.path.normpath(exp_name))
        # Find the loss and simulation files for the last epoch with the correct timestamp
        pattern_loss = os.path.join(exp_name, f"{base_exp_name}_NO_*_Epoch_{last_epoch}_loss_{run_datetime}.png")
        pattern_sim = os.path.join(exp_name, f"{base_exp_name}_NO_*_Epoch_{last_epoch}_simulation_{run_datetime}.png")
        loss_files = sorted(glob.glob(pattern_loss))
        sim_files = sorted(glob.glob(pattern_sim))
        if loss_files:
            img_loss = mpimg.imread(loss_files[-1])
            axes[0, col].imshow(img_loss)
            axes[0, col].set_title(f"{generator} - Loss")
            axes[0, col].axis('off')
        else:
            axes[0, col].set_visible(False)
        if sim_files:
            img_sim = mpimg.imread(sim_files[-1])
            axes[1, col].imshow(img_sim)
            axes[1, col].set_title(f"{generator} - Simulation")
            axes[1, col].axis('off')
        else:
            axes[1, col].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def lstm_progression_figure(generator, filename, epochs_to_plot=None, epochs=100, **run_experiment_kwargs):
    """
    Trains an LSTM on the specified generator and generates a figure with the simulations at selected epochs
    and the final loss curve on the right.
    """
    if epochs_to_plot is None:
        epochs_to_plot = [1, 15, 30, 100]
    exp_name = f'experiments/LSTM_TS_MODEL_{generator.upper()}_HIDDEN_5'
    # Run the training (run_datetime will be used for all files)
    model, train_loss, test_loss, exp_name, run_datetime, last_epoch = run_experiment(
        model_type='lstm',
        epochs=epochs,
        batch_size=32,
        hidden_size=5,
        generator_name=generator,
        exp_name=exp_name,
        fformat="png",
        **run_experiment_kwargs
    )

    base_exp_name = os.path.basename(os.path.normpath(exp_name))

    # Prepare the figure: 1 row, len(epochs_to_plot)+1 columns (simulations + loss)
    ncols = len(epochs_to_plot) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4))
    for i, epoch in enumerate(epochs_to_plot):
        sim_file = os.path.join(exp_name, f"{base_exp_name}_NO_1_Epoch_{epoch}_simulation_{run_datetime}.png")
        if os.path.exists(sim_file):
            img_sim = mpimg.imread(sim_file)
            axes[i].imshow(img_sim)
            axes[i].set_title(f"Epoch {epoch}")
            axes[i].axis('off')
        else:
            axes[i].set_visible(False)
    # Final loss curve (last epoch)
    loss_file = os.path.join(exp_name, f"{base_exp_name}_NO_1_Epoch_{epochs}_loss_{run_datetime}.png")
    if os.path.exists(loss_file):
        img_loss = mpimg.imread(loss_file)
        axes[-1].imshow(img_loss)
        axes[-1].set_title(f"Loss (Epoch {epochs})")
        axes[-1].axis('off')
    else:
        axes[-1].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Add top margin
    print(f"Saving figure to {filename}")
    plt.savefig(filename, dpi=300)
    plt.close()  # Close the figure to free memory

def lstm_sin_progression_figure(filename="experiments/lstm_sin_progression.png"):
    """
    Trains an LSTM on sin and generates a figure with the simulations at epochs 1, 15, 30, 100
    and the final loss curve on the right.
    """

    epochs_to_plot = [1, 15, 30, 100]
    epochs = 100
    generator = 'sin'
    exp_name = f'experiments/LSTM_TS_MODEL_{generator.upper()}_HIDDEN_5'
    lstm_progression_figure(generator, filename, epochs_to_plot=epochs_to_plot, epochs=epochs)

def lstm_airline_passengers_figure(filename="experiments/lstm_airline_passengers.png"):
    """
    Train an LSTM on the airline-passengers dataset and generate a figure with the loss and simulation results at epochs 1, 15, 30, 100.
    """

    epochs_to_plot = [1, 15, 30, 100]
    epochs = 100
    generator = 'airline_passengers'
    exp_name = f'experiments/LSTM_TS_MODEL_{generator.upper()}_HIDDEN_5'

    lstm_progression_figure(generator, filename, epochs_to_plot=epochs_to_plot, epochs=epochs)

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
        exp_name='experiments/COMPARE_QLSTM_SIN',
        vqc_depth=4,
        **common_params
    )
    
    print("\nTraining LSTM...")
    lstm_model, lstm_train_loss, lstm_test_loss = run_experiment(
        model_type='lstm',
        exp_name='experiments/COMPARE_LSTM_SIN',
        **common_params
    )
    
    print(f"\nFinal Results:")
    print(f"QLSTM - Train Loss: {qlstm_train_loss[-1]:.6f}, Test Loss: {qlstm_test_loss[-1]:.6f}")
    print(f"LSTM  - Train Loss: {lstm_train_loss[-1]:.6f}, Test Loss: {lstm_test_loss[-1]:.6f}")

if __name__ == '__main__':
    lstm_airline_passengers_figure()
    lstm_sin_progression_figure()
    #example_lstm_experiments()
    #compare_models()