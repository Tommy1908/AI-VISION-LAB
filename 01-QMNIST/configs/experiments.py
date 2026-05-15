# configs/experiments.py
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENTS = {
    "1_layer": {
        "project_name": "layers",
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 15,
        "hidden_layers": [1],
        "activation": ["ReLU"],
        "device": DEVICE
    },
    "2_layer": {
        "project_name": "layers",
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 15,
        "hidden_layers": [2],
        "activation": ["ReLU"],
        "device": DEVICE
    }
}