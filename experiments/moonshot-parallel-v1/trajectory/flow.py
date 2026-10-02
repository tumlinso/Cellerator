"""CPU Torch reference for the local law dX/dt = L X + X R (M03).

Learning computes fresh matrix exponentials on every call. Inference stores
detached exponentials and checks actual parameter values as well as metadata.
This block is not a nonlinear trajectory solver or a performance measurement.
"""

import math

import torch


class SylvesterFlow:
    """Exact local flow with caller-owned L/R tensors and explicit generations.

    Call ``flow(X, dt, L=..., R=...)`` to override the stored factors. Training
    defaults to torch's grad mode; explicit inference detaches all operands.
    A differentiable dt cannot be explicitly sent through inference.
    """

    def __init__(self, L, R, *, definition_id, structure_generation=0,
                 parameter_generation=0):
        if not isinstance(definition_id, str) or not definition_id:
            raise ValueError("definition_id must be a nonempty string")
        for generation in (structure_generation, parameter_generation):
            if type(generation) is not int or generation < 0:
                raise ValueError("generations must be nonnegative integers")
        self._validate_factor("L", L)
        self._validate_factor("R", R)
        if L.shape[0] != L.shape[1] or R.shape[0] != R.shape[1]:
            raise ValueError("L and R must be square")
        if L.dtype != R.dtype:
            raise ValueError("L and R must have the same dtype")
        self.L, self.R = L, R
        self.definition_id = definition_id
        self.structure_generation = structure_generation
        self.parameter_generation = parameter_generation
        self._cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _validate_factor(name, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cpu":
            raise ValueError("this reference supports CPU tensors only")
        if tensor.dtype not in (torch.float32, torch.float64):
            raise ValueError("this reference supports float32/float64 only")
        if tensor.ndim != 2 or min(tensor.shape) == 0:
            raise ValueError(f"{name} must be a nonempty matrix")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must be finite")

    @classmethod
    def _validate(cls, X, L, R, dt):
        for name, tensor in (("X", X), ("L", L), ("R", R)):
            cls._validate_factor(name, tensor)
        if L.shape != (X.shape[0], X.shape[0]):
            raise ValueError("L must be square and match X rows")
        if R.shape != (X.shape[1], X.shape[1]):
            raise ValueError("R must be square and match X columns")
        if X.dtype != L.dtype or X.dtype != R.dtype:
            raise ValueError("X, L and R must have the same dtype")
        if isinstance(dt, torch.Tensor):
            if dt.ndim != 0 or dt.device != X.device or dt.dtype != X.dtype:
                raise ValueError("dt must be scalar with X's dtype and device")
            dt_value = dt.detach().item()
        else:
            dt_value = float(dt)
        if not math.isfinite(dt_value):
            raise ValueError("dt must be finite")
        return dt_value

    @staticmethod
    def _signature(tensor):
        # Full values also catch replacement tensors and mutations bypassing
        # PyTorch's version counter; a metadata generation is not sufficient.
        snapshot = tensor.detach().contiguous().clone()
        try:
            version = tensor._version
        except RuntimeError:  # tensors created in torch.inference_mode
            version = None
        return (tuple(tensor.shape), version, snapshot.numpy().tobytes())

    def __call__(self, X, dt, *, L=None, R=None, training=None):
        L = self.L if L is None else L
        R = self.R if R is None else R
        dt_value = self._validate(X, L, R, dt)
        training = torch.is_grad_enabled() if training is None else training
        if training:
            # Never retain an autograd graph in the inference cache.
            return torch.matrix_exp(dt * L) @ X @ torch.matrix_exp(dt * R)
        if isinstance(dt, torch.Tensor) and dt.requires_grad:
            raise ValueError("differentiable dt requires training=True")
        key = (self.definition_id, self.structure_generation,
               self.parameter_generation, dt_value, X.dtype, X.device,
               self._signature(L), self._signature(R))
        if key in self._cache:
            self.cache_hits += 1
            left, right = self._cache[key]
        else:
            self.cache_misses += 1
            with torch.no_grad():
                left = torch.matrix_exp(dt_value * L.detach())
                right = torch.matrix_exp(dt_value * R.detach())
            # Bound prototype memory to one prepared parameter/dt context.
            self._cache = {key: (left, right)}
        with torch.no_grad():
            return left @ X.detach() @ right

    def cache_info(self):
        return {"entries": len(self._cache), "hits": self.cache_hits,
                "misses": self.cache_misses}

    def checkpoint(self):
        """JSON-compatible snapshots; cached graphs/values are excluded."""
        self._validate_factor("L", self.L)
        self._validate_factor("R", self.R)
        return {"schema_version": 1,
                "definition_id": self.definition_id,
                "structure_generation": self.structure_generation,
                "parameter_generation": self.parameter_generation,
                "dtype": str(self.L.dtype).removeprefix("torch."),
                "L": self.L.detach().tolist(), "R": self.R.detach().tolist(),
                "L_requires_grad": self.L.requires_grad,
                "R_requires_grad": self.R.requires_grad}

    @classmethod
    def from_checkpoint(cls, payload):
        expected_keys = {"schema_version", "definition_id", "structure_generation",
                         "parameter_generation", "dtype", "L", "R",
                         "L_requires_grad", "R_requires_grad"}
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError("checkpoint must be a dict with exactly the expected fields")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported flow checkpoint schema")
        dtype = {"float32": torch.float32, "float64": torch.float64}.get(payload["dtype"])
        if dtype is None:
            raise ValueError("unsupported checkpoint dtype")
        for key in ("L_requires_grad", "R_requires_grad"):
            if type(payload[key]) is not bool:
                raise ValueError("checkpoint requires_grad flags must be boolean")
        L = torch.tensor(payload["L"], dtype=dtype).requires_grad_(payload["L_requires_grad"])
        R = torch.tensor(payload["R"], dtype=dtype).requires_grad_(payload["R_requires_grad"])
        return cls(L, R, definition_id=payload["definition_id"],
                   structure_generation=payload["structure_generation"],
                   parameter_generation=payload["parameter_generation"])
