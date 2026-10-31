"""CPU proof that a Cellerator native autograd operator composes with nn modules."""
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


def _source_product2_module():
    import cellerator

    try:
        importlib.import_module("cellerator._native")
    except ModuleNotFoundError as exc:
        if exc.name != "cellerator._native":
            raise
        installed_package = Path(
            importlib.metadata.distribution("cellerator").locate_file("cellerator")
        )
        if installed_package.is_dir() and str(installed_package) not in cellerator.__path__:
            cellerator.__path__.append(str(installed_package))
        importlib.import_module("cellerator._native")

    source_path = Path(__file__).resolve().parents[2] / "python/cellerator/torch/product2.py"
    spec = importlib.util.spec_from_file_location(
        "cellerator._nn_foundation_source_product2", source_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load the source Product2 Torch adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Product2Module


class NativeCompositionTests(unittest.TestCase):
    def test_product2_native_autograd_trains_alongside_ordinary_torch_layers(self):
        Product2Module = _source_product2_module()
        torch.manual_seed(16)
        native = Product2Module(
            torch.tensor([0.7], dtype=torch.float32),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
        )
        model = nn.Sequential(
            nn.Linear(2, 2, bias=False),
            native,
            nn.Linear(1, 1, bias=False),
        ).float()
        sample = torch.tensor([0.6, -0.8], dtype=torch.float32, requires_grad=True)
        target = torch.tensor([-0.4], dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        optimizer.zero_grad(set_to_none=True)
        loss = (model(sample) - target).square().mean()
        loss.backward()

        self.assertIsNotNone(sample.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        optimizer.step()

        after = list(model.parameters())
        self.assertTrue(any(not torch.equal(old, new.detach()) for old, new in zip(before, after)))
        self.assertTrue(torch.isfinite(model(sample.detach())).all())


if __name__ == "__main__":
    unittest.main()
