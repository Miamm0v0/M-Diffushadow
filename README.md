# M-Diffushadow

M-Diffushadow is a multimodal Diffushadow project for learning quantum classical-shadow measurement data with a masked-token Transformer.  The current code focuses on two one-dimensional spin-chain settings:

- **TFI**: transverse-field Ising model.
- **XXZ**: periodic XXZ spin chain, with both single-site shadow samples and nearest-neighbor two-site product samples.

The main idea is to train a model to complete masked measurement outcomes conditioned on the physical parameter and measurement bases.  In the multimodal setting, the model sees two complementary modalities:

- **B / single modality**: single-site Pauli measurements.
- **C / pair modality**: nearest-neighbor two-site Pauli product measurements.

## Repository Layout

```text
.
|-- Qmultimodel.py                     # Multimodal Transformer model
|-- train_multi.py                     # Multimodal training script
|-- eval_new.py                        # Evaluation and sample generation
|-- eval_utils.py                      # Hamiltonians, exact values, shadow estimators
|-- generate_xxz_multimodal_dataset.py # XXZ multimodal dataset generation
|-- eval_multimodal_partial.py         # Partial-mask FM consistency evaluation
|-- cal_xxz_multimodal_errors.py       # Error metrics for XXZ .npz outputs
|-- draw_fig_properties.py             # Plotting for TFI/J1J2/ANNNI-style outputs
|-- draw_fig_properties_xxz.py         # Plotting for XXZ multimodal outputs
`-- draw_loss.py                       # Training loss visualization
```

Some scripts also keep compatibility hooks for older single-modal baselines and other spin models.  The main multimodal path is `Qmultimodel.py` + `train_multi.py` + `eval_new.py`.

## Data Format

The multimodal model uses two JSON files with matching physical-parameter groups.

### Single / B Modality

Each row has length `1 + 2N`:

```text
[g_or_Delta, P1, ..., PN, b1, ..., bN]
```

For XXZ, the scalar condition is `Delta`.  For TFI, it is usually the transverse-field parameter `g` or `h`, depending on the dataset naming.

### Pair / C Modality

Each row has length `1 + 3N`:

```text
[g_or_Delta, r1_1, r2_1, c1, ..., r1_N, r2_N, cN]
```

For XXZ, item `i` represents the periodic nearest-neighbor bond `(i, i+1)`.

Token convention:

```text
Pauli basis: 2 = X, 3 = Y, 4 = Z
Outcome:     0 = -1, 1 = +1
Mask token: -1
```

The current model and training script are written for `N = 10` qubits by default.

## Installation

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

For CUDA training, install the PyTorch build that matches your CUDA version from the official PyTorch instructions, then install the remaining packages from `requirements.txt`.

## Generate XXZ Multimodal Data

Example:

```bash
python generate_xxz_multimodal_dataset.py \
  --num-qubits 10 \
  --samples-per-delta 10000 \
  --deltas -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 \
  --pair-basis-mode diagonal \
  --seq-json-out data/xxz_train_seq.json \
  --pair-json-out data/xxz_train_phys.json \
  --meta-json-out data/xxz_train_meta.json
```

`--pair-basis-mode diagonal` samples only `XX`, `YY`, and `ZZ` pair products, which matches the XXZ Hamiltonian terms.  Use `full` to sample all nine Pauli-product bases.

## Train M-Diffushadow

Train both modalities:

```bash
python train_multi.py \
  --seq_data_path data/xxz_train_seq.json \
  --phys_data_path data/xxz_train_phys.json \
  --train_target both \
  --epochs 100 \
  --batch_size 256 \
  --model_save_path checkpoints/m_diffushadow_xxz.pth \
  --loss_save_path logs/loss_multimodal_xxz.json
```

`--train_target` can be:

- `b`: train only the single-site B head.
- `c`: train only the pair-product C head.
- `both`: train both heads jointly.

The training script randomly masks B/C outcomes and computes cross-entropy loss only on the masked positions.

## Evaluate

Example XXZ evaluation:

```bash
python eval_new.py \
  --multimodal \
  --predict_model xxz \
  --generate_target both \
  --model_path checkpoints/m_diffushadow_xxz.pth \
  --save_data_path outputs/xxz_eval.npz \
  --num_qubits 10 \
  --h_length 41 \
  --sample_size_per_h 10000 \
  --repeat_times 6 \
  --diffusion_steps 6 \
  --exact_value
```

The saved `.npz` contains generated-shadow estimates and, when `--exact_value` is enabled, exact reference curves for comparison.

## Plot and Error Analysis

Plot XXZ multimodal properties:

```bash
python draw_fig_properties_xxz.py \
  --file outputs/xxz_eval.npz \
  --output_dir plots/xxz \
  --modalities both \
  --show_phase_boundaries
```

Compute XXZ error metrics:

```bash
python cal_xxz_multimodal_errors.py \
  --file outputs/xxz_eval.npz \
  --output_json outputs/xxz_eval_errors.json
```

Plot training loss:

```bash
python draw_loss.py \
  --json_path logs/loss_multimodal_xxz.json \
  --output plots/loss_xxz.png
```

## Notes

- The code assumes grouped training data.  `train_multi.py` currently groups training rows in blocks of `10000` samples per physical-parameter value.
- `Qmultimodel.py` uses fixed sequence lengths for `N = 10`: single length `21`, pair length `31`, and combined Transformer length `41`.
- TFI datasets should follow the same B/C JSON conventions as above.
- `eval_new.py` contains legacy support for additional models and baseline architectures; keep the corresponding model files in the repository if you plan to expose those paths.
