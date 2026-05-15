import torch.nn as nn
import torch

ACTIVATION_MAP = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid
    }

class QMNIST_MLP(nn.Module):
    def __init__(self,
                hidden_layers:list[int]=[512],
                activation:list[str]=["ReLu"]
                ) -> None:
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation

        #Validate and convert activation
        activations_names:list[str] = activation
        activations: list[type[nn.Module]] = []
        if len(activation) == 1:
            activations_names *= len(hidden_layers)
        for act in activations_names:
            act = act.lower()
            if act not in ACTIVATION_MAP:
                raise ValueError(f"'{act}' is not valid, use: {list(ACTIVATION_MAP.keys())}")
            activations.append(ACTIVATION_MAP[act])

        layers:list[nn.Module] = []
        layers.append(nn.Flatten())
        input_dimension = 28*28 # QMNIST input size
        for i, neurons in enumerate(hidden_layers):
            layers.append(nn.Linear(input_dimension, neurons))
            layers.append(activations[i]())
            input_dimension = neurons # input of next layer is the current output
        layers.append(nn.Linear(input_dimension, 10)) #10 for the usual output 0-9. Could try a "NaN"?

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def save_model(self, path):
        checkpoint = {
            "state_dict": self.state_dict(),
            "hidden_layers": self.hidden_layers,
            "activation": self.activation
        }
        torch.save(checkpoint, path)

    @classmethod 
    def load_model(cls, path):
        checkpoint = torch.load(path)
        model = cls(
            hidden_layers=checkpoint['hidden_layers'],
            activation=checkpoint['activation']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model