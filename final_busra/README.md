#  True Gene Classification with GiG network

This repository contains code for MLMI Praktikum Course project, training a multi-class gene classifier using GiG network with PyTorch Geometric. The model processes graphs with 2405 classes.

---

## Dataset

The dataset is too large to be stored in this GitHub repo. It is available via Google Drive.

 **[Download the processed dataset as pkl files here](https://drive.google.com/drive/folders/1m42kNkKKzllYybEwC3xRH9wZXZUFmIha?usp=drive_link)**  
Place the files in the following directory structure:

```
final_busra/
├── main.py
├── main_clip.py
├── models.py
├── DataLast/
│   └── corrected_datasets/
│       ├── train_shuffled_y.pkl
│       ├── val_shuffled_y.pkl
│       └── test_shuffled_y.pkl
```

---

## Code Structure

- **`main.py`**  
  Main script that handles:
  - Loading datasets (`.pkl` files)
  - Model definition (`GeneClassifier`)
  - Training with `train()`
  - Validation and testing with `evaluate()`
  - Logging using Weights & Biases (`wandb`)

- **`models.py`**  
  Contains the model components:
  - `NodeConvolution`: node-level encoder with GNN layers (GAT, GCN, etc.)
  - `LGL` / `LGLKL`: population-level embedding module
  - `GNN`: final graph-level processor
  - `Classifier`: fully connected classification head
  - `GiG`: full model wrapper combining the modules

---

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Kai-Ze612/Gragh-in-Graph-for-rare-dieases-diagonsis.git
   ```

2. If using wandb, log in:
   ```bash
   wandb login
   ```

---

## Training

To train the model:

```bash
python main.py
```

Trained models are saved in `last_models/run_<timestamp>/`.

---

## Evaluation

The test set is evaluated at the end of training, and metrics (loss and accuracy) are printed and optionally logged to Weights & Biases.

---

##  Configuration

Model and training hyperparameters are defined in the `config` dictionary in `main.py`:
```python
config = {
    "num_node_features": 128,
    "output_dim": 2405,
    "node_level_module": "GAT",
    ...
}
```

Modify these as needed for experiments.

---


