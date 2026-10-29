"""Compose two existing CPU-native Product2 modules as a prepared field."""
import torch

from cellerator.experimental import ExecutionField
from cellerator.torch import Product2Module


pairwise = Product2Module(
    torch.tensor([0.5, -1.25], dtype=torch.float32),
    torch.tensor([0, 1], dtype=torch.int64),
    torch.tensor([1, 2], dtype=torch.int64),
)
combine = Product2Module(
    torch.tensor([2.0], dtype=torch.float32),
    torch.tensor([0], dtype=torch.int64),
    torch.tensor([1], dtype=torch.int64),
)
field = ExecutionField((("pairwise", pairwise), ("combine", combine)), input_extent=3)
binding = field.bind(torch.tensor([1.5, -2.0, 4.0], dtype=torch.float32))

print("prepared native handles:", field.prepared_handles)
print("field output:", field(binding))
