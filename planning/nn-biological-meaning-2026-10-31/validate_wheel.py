"""Validate an extracted wheel without changing the installed package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT_CHECK = r'''
import sys
import cellerator
from cellerator import Cell, FeatureBinding, IdentityCodec, PreparedAlignment, Prediction, PredictionQuery
assert "torch" not in sys.modules
binding = FeatureBinding.from_ids(("gene-A", "gene-B"), (2, 7), model_capacity=8)
assert binding.prepare(("gene-B", "gene-A"), binding.identities).source_positions == (1, 0)
assert Cell(3).bind(lambda data: data + 1).predict() == 4
print(cellerator.__file__)
'''

NN_CHECK = r'''
import json
import torch
from torch import nn
from cellerator import FeatureBinding, Prediction, PredictionQuery
from cellerator import nn as cnn
from cellerator.torch import nn as tnn
from cellerator.adapters.torch import GatedInteractor
torch.set_num_threads(1)
torch.manual_seed(911)
names = ("FeatureEncoder", "PreparedRelation", "RelationAttention", "InteractionPrograms",
         "FeatureReadout", "ModelWithViews", "grouped_linear", "GroupedLinear",
         "biological_manifest", "validate_biological_manifest", "save_biological_manifest",
         "load_biological_manifest", "align_prediction_targets", "categorical_target_columns",
         "reduce_prediction_loss")
for name in names:
    assert getattr(cnn, name) is getattr(tnn, name), name
genes = FeatureBinding.from_ids(("gA", "gB", "gC"), (2, 7, 9), model_capacity=10)
encoder = cnn.FeatureEncoder(10, 4, binding=genes)
alignment = genes.map_occurrences(("gA", "gA", "gB"))
features = encoder.encode_bound(alignment, torch.tensor([0.1, 0.2, 0.3]))
assert alignment.to_indices() is alignment.to_indices()
relation = cnn.PreparedRelation.from_csr((0, 2, 3), (0, 1, 2),
    source_order=("gA", "gA", "gB"), target_order=("pA", "pB"))
programs = cnn.InteractionPrograms(4, 2, program_ids=("pA", "pB"), context_size=3)
program_result = programs(features, relation=relation, feature_order=relation.source_order,
    context=torch.randn(3))
program_result["feature_updates"].square().sum().backward()
assert programs.program_queries.grad is not None
peaks = FeatureBinding.from_ids(("peak-A", "peak-B"), (3, 8), domain="peak", model_capacity=9)
model = cnn.ModelWithViews(nn.Linear(4, 4))
model.add_view("atac", cnn.FeatureReadout(peaks, 4), output_binding=peaks, queryable=True)
x = torch.randn(2, 3, 4, requires_grad=True)
prediction = model.query(x, PredictionQuery("atac", ("peak-B",), ("cell-1", "cell-2")))
assert prediction.feature_ids == ("peak-B",) and prediction.values.shape == (2, 1)
loss, count = cnn.reduce_prediction_loss(prediction.values.square(), unit="observation")
assert count.item() == 2
loss.backward()
assert x.grad is not None and model.base.weight.grad is not None
manifest = cnn.biological_manifest(model)
cnn.validate_biological_manifest(model, manifest)
columns = cnn.categorical_target_columns(Prediction(torch.zeros(2, 2), (7, 2)), (2, 7))
assert columns.tolist() == [1, 0]
native = cnn.Product2Module(torch.ones(1), torch.tensor([0]), torch.tensor([1]))
native_input = torch.tensor([2.0, 3.0], requires_grad=True)
native(native_input).sum().backward()
assert native.coefficients.grad is not None and native_input.grad is not None
grouped = cnn.GroupedLinear(2, 4, 3)
grouped(torch.randn(3, 4), (0, 1, 3)).square().mean().backward()
assert grouped.weight.grad is not None
conditioner = nn.Sequential(nn.Linear(3, 1), nn.Unflatten(-1, (1, 1)))
interactor = GatedInteractor(4, 8, 4, conditioner=conditioner)
gate_context = torch.randn(2, 3, requires_grad=True)
interactor(torch.randn(2, 3, 4), context=gate_context).sum().backward()
assert gate_context.grad is not None and conditioner[0].weight.grad is not None
print(json.dumps({"torch": torch.__version__, "facade_exports": len(names),
                  "query_shape": list(prediction.values.shape), "native_cpu_gradients": True,
                  "registered_gate_conditioner_gradients": True}))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    wheel = args.wheel.resolve()
    with tempfile.TemporaryDirectory(prefix="cellerator-biological-wheel-") as temporary:
        installation = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(installation)
        package = installation / "cellerator"
        packaged = sorted(package.rglob("*.py"))
        mismatches = [str(path.relative_to(installation)) for path in packaged
                      if path.read_bytes() != (repository / "python" / path.relative_to(installation)).read_bytes()]
        source_count = len(list((repository / "python/cellerator").rglob("*.py")))
        if mismatches or source_count != len(packaged):
            raise RuntimeError(f"wheel/source mismatch: {mismatches}; counts {len(packaged)}/{source_count}")
        environment = dict(os.environ, PYTHONPATH=str(installation), CUDA_VISIBLE_DEVICES="",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

        def run(arguments: list[str]) -> str:
            result = subprocess.run([sys.executable, *arguments], cwd=installation,
                                    env=environment, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            return result.stdout + result.stderr

        root = run(["-c", ROOT_CHECK])
        neural = json.loads(run(["-c", NN_CHECK]))
        compatibility = run([str(repository / "tests/bindings/python/test_python_surface.py")])
        record = {"schema": "cellerator.biological-meaning.wheel-validation.v1",
                  "wheel": wheel.name, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  "packaged_python_files": len(packaged), "source_bytes_match": not mismatches,
                  "torch_free_root": True, "root_import": root.strip(),
                  "neural_smoke": neural, "compatibility_suite": compatibility.strip(),
                  "limits": "Isolated CPU installation; no CUDA or performance qualification."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
