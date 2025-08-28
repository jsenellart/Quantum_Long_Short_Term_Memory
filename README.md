# QLSTM for Time-Series Forecasting with Quantum Circuits

This repository implements a Quantum Long Short-Term Memory (QLSTM) model for time-series forecasting. It integrates parameterized quantum circuits (VQCs) into a classical LSTM architecture using [PennyLane](https://pennylane.ai/) and [PyTorch](https://pytorch.org/). A classical LSTM baseline, multiple pluggable data generators, experiment aggregation utilities, and multi-seed robustness analysis are provided.

---

## 🚀 Highlights
- Custom QLSTMCell with VQC-based gates (input, forget, cell, output)
- Classical LSTM baseline for comparison
- Pluggable time-series generators (synthetic, real, quantum-inspired)
- Progression figures (predictions at selected epochs + final loss)
- Summary grids across generators (loss + simulation)
- Multi-seed loss envelopes (mean, std band, min/max)
- Reproducible timestamped artifacts under `experiments/`
- Quantum population inversion generators (collapse & revival)

---

## Repository Structure (excerpt)
```
data/
  generators.py        # All data generators
dataset/               # External CSV datasets (e.g. airline_passengers.csv)
experiments/           # Auto-generated figures
run_experiments.py     # Orchestration utilities
QLSTM_v0.py            # Core training (run_experiment)
```

---

## 🧪 Data Generators
Implemented in `data/generators.py`:
- damped_shm
- sin, linear, exp
- bessel_j2
- airline_passengers (CSV in `dataset/`)
- population_inversion
- population_inversion_collapse_revival (Jaynes–Cummings collapse & revival)

Add a new generator by creating a class with a `generate()` method returning a 1-D numpy array (float32) and passing its registered name to `run_experiment`. Register the generator in `DataFactory` (`data/__init__.py`)

Skeleton:
```python
class MyGenerator(BaseGenerator):
    def generate(self):
        return series.astype(np.float32)
```

---

## Running a Single Experiment
CLI:
```bash
python QLSTM_v0.py
```

Running the script will by default:
- Train a QLSTM model for 100 epochs on the default `damped_shm` generator
- Save plots and model checkpoint files under a folder like: `experiments/QLSTM_TS_MODEL_DAMPED_SHM`
- Generate the following visualizations:
  - Training & testing loss curves
  - Ground-truth vs. predicted output (simulation) plot



Programmatic:
```python
from QLSTM_v0 import run_experiment
model, train_loss, test_loss, exp_name, run_dt, last_epoch = run_experiment(
    model_type='qlstm',
    generator_name='damped_shm',
    epochs=100,
    hidden_size=6,
    seq_length=5
)
```

Artifact naming (shared timestamp):
```
<EXP_NAME>/<EXP_NAME>_NO_<run>_Epoch_<epoch>_{loss|simulation}_<timestamp>.png
```

---

## Progression Figures
Show predictions at selected epochs + final loss (example from `run_experiments.py`):
```python
from run_experiments import lstm_progression_figure
lstm_progression_figure(
    generator='bessel_j2',
    filename='experiments/lstm_bessel_progression.png',
    epochs_to_plot=[1, 15, 30, 100],
    epochs=100
)
```

---

## Summary Grids Across Generators
Run multiple LSTM experiments and aggregate their final loss & simulation images:
```python
from run_experiments import example_lstm_experiments
example_lstm_experiments()
```
Output: `experiments/summary_lstm_hidden_5_experiments.png`.

---

## Multi-Seed Stability Analysis
Evaluate robustness:
```python
from run_experiments import lstm_random_seeds
lstm_random_seeds(40, generator='sin', epochs=100)
```
Generates `experiments/LSTM_RANDOM_SEEDS_MODEL_SIN_loss_envelope.png` with:
- Mean loss curve
- ±1 std deviation band
- Min/max envelope

---

## QLSTM vs LSTM Comparison
```python
from run_experiments import compare_models
compare_models()
```
Prints final train/test losses for both architectures.

---

## Reproducibility
A single timestamp captured at training start is reused for all artifacts of that run, enabling deterministic grouping (used by summary and progression figure utilities).

---

## (Optional) Future Robustness Metrics
Potential additions:
- Monotonicity score
- Spike intensity (sum of upward jumps / total downward progress)
- Total variation (sum |Δloss|)

---

## Citation
If you use this repository, please cite:
```bibtex
@inproceedings{chen2022quantum,
  title={Quantum long short-term memory},
  author={Chen, Samuel Yen-Chi and Yoo, Shinjae and Fang, Yao-Lung L},
  booktitle={ICASSP 2022 - IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={8622--8626},
  year={2022},
  organization={IEEE}
}
```

---

## Next Steps

Contributions (new generators, metrics, models) are welcome.

---

## API: run_experiment

The core utility for training either an LSTM or QLSTM on any registered generator.

Signature (see detailed docstring in `QLSTM_v0.py`):
```
run_experiment(
  generator_name='damped_shm',
  model_type='qlstm',
  seq_length=4,
  hidden_size=5,
  batch_size=10,
  epochs=100,
  learning_rate=0.01,
  train_split=0.67,
  vqc_depth=5,
  exp_name=None,
  exp_index=1,
  seed=0,
  fformat='pdf',
  run_datetime=None,
  save_only_last_progress=True,
  **generator_kwargs
)
```

Returns:
`(model, train_loss_list, test_loss_list, exp_name, run_datetime, last_epoch)`.

Key behaviors:
- Creates experiment folder (if missing)
- Saves loss & simulation figures each epoch (or only final, depending on `save_only_last_progress`)
- Embeds a single timestamp `run_datetime` into all artifact filenames for deterministic grouping
- Stores pickled loss arrays, simulation outputs, and the model state dict

Typical examples:
```python
run_experiment(model_type='lstm', generator_name='sin', epochs=50)
run_experiment(model_type='qlstm', generator_name='population_inversion_collapse_revival', vqc_depth=4, epochs=80)
run_experiment(model_type='lstm', generator_name='airline_passengers', seq_length=12, epochs=120)
```
Combine with multi-seed outer loops or higher-level helpers in `run_experiments.py` as needed.
