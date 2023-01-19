import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.eval()

    def forward(self, x: torch.Tensor, add_sigmoid: bool = True) -> torch.Tensor:
        with torch.no_grad():
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            if add_sigmoid:
                x = torch.sigmoid(x)

            return x

    def init_weights_xavier(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


if __name__ == '__main__':
    net = Network(2, 16, 1)
