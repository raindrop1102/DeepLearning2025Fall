import torch
import torch.nn as nn


# x = torch.randn(64, 784)

modle = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features= 1024, out_features= 256),
    nn.ReLU(),
    nn.Linear(in_features= 256, out_features= 128),
    nn.ReLU(),
    nn.Linear(in_features= 128, out_features= 10),
    nn.Softmax(dim=1)
)

# out = modle(x)
# print(out.shape)