import torch
from torch.nn import functional as F
import torch.nn as nn

class TestLayers(nn.Sequential):
    def __init__(self, test_input_dim, test_output_dim):
        super().__init__(
            nn.Linear(test_input_dim, test_output_dim),
            nn.ReLU()
        )
        

test_input = torch.randn(64, 10, 10)
test_output = TestLayers(10, 10)(test_input)

print(test_output.shape)

test_expand = test_output[(..., ) + (None, ) * 2]
print(test_expand.shape)

print(test_output.unsqueeze(-1).unsqueeze(-1).shape)

test_output

