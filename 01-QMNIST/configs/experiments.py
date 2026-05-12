# configs/experiments.py
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENTS = {
    "512": {
        "project_name": "Layers",
        "batch_size": 64,
        "lr": 1e-1,
        "epochs": 10,
        "hidden_layers": [512], # Array de capas
        "activation": ["ReLU"],
        "device": DEVICE
    }
}
