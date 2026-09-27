"""Biological identity, mechanism construction, and guarded Torch updates."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence
import weakref

import torch
from torch import Tensor, nn


def _identity(value: "Identity | tuple[int, int]") -> "Identity":
    if isinstance(value, Identity):
        return value
    if len(value) != 2:
        raise ValueError("an identity is a (low, high) uint64 pair")
    return Identity(int(value[0]), int(value[1]))


def _signed_word(word: int) -> int:
    """Encode uint64 in Torch's signed int64 list schema without losing bits."""
    return word if word < (1 << 63) else word - (1 << 64)


@dataclass(frozen=True, order=True)
class Identity:
    """Stable 128-bit biological or execution identity, encoded low then high."""

    low: int
    high: int

    def __post_init__(self) -> None:
        limit = (1 << 64) - 1
        if not (0 <= self.low <= limit and 0 <= self.high <= limit):
            raise ValueError("identity words must fit uint64")
        if self.low == 0 and self.high == 0:
            raise ValueError("zero is not a valid persistent identity")

    @property
    def words(self) -> tuple[int, int]:
        return self.low, self.high


@dataclass(frozen=True)
class Axis:
    """A biological axis identity plus its logical extent.

    Shape is deliberately separate from identity: two equally sized axes with
    different domain, order, geometry, or partition are not interchangeable.
    """

    domain: Identity
    order: Identity
    geometry: Identity
    partition: Identity
    extent: int

    def __post_init__(self) -> None:
        for name in ("domain", "order", "geometry", "partition"):
            object.__setattr__(self, name, _identity(getattr(self, name)))
        if self.extent <= 0:
            raise ValueError("axis extent must be positive")

    @property
    def words(self) -> list[int]:
        return [_signed_word(word) for identity in
                (self.domain, self.order, self.geometry, self.partition)
                for word in identity.words]


@dataclass(frozen=True)
class Incidence:
    """One ordered argument slot referring to an index on a biological axis."""

    slot: int
    role: Identity
    index: int
    axis: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identity(self.role))
        if self.slot < 0 or self.axis < 0 or self.index < 0:
            raise ValueError("incidence slot, axis, and index must be nonnegative")


@dataclass(frozen=True)
class OutputContribution:
    """A declared additive contribution to an output coordinate."""

    slot: int
    role: Identity
    index: int
    assembly: Identity
    scale: float = 1.0
    axis: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identity(self.role))
        object.__setattr__(self, "assembly", _identity(self.assembly))
        if self.slot < 0 or self.axis < 0 or self.index < 0:
            raise ValueError("output slot, axis, and index must be nonnegative")
        if not torch.isfinite(torch.tensor(float(self.scale))):
            raise ValueError("output contribution scale must be finite")


@dataclass(frozen=True)
class Mechanism:
    """One product law with ordered, possibly repeated arguments."""

    identity: Identity
    coefficient: Identity
    arguments: tuple[Incidence, ...]
    outputs: tuple[OutputContribution, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _identity(self.identity))
        object.__setattr__(self, "coefficient", _identity(self.coefficient))
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        slots = sorted(x.slot for x in self.arguments)
        if slots != list(range(len(slots))):
            raise ValueError("argument slots must be unique and contiguous from zero")
        if not self.outputs:
            raise ValueError("each mechanism must declare at least one output")


@dataclass(frozen=True)
class MechanismSpec:
    """Typed biological axes and a set of prepared product mechanisms."""

    input_axis: Axis
    output_axis: Axis
    coefficient_axis: Axis
    coefficient_ids: tuple[Identity, ...]
    mechanisms: tuple[Mechanism, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient_ids",
                           tuple(_identity(i) for i in self.coefficient_ids))
        object.__setattr__(self, "mechanisms", tuple(self.mechanisms))
        if len(self.coefficient_ids) != self.coefficient_axis.extent:
            raise ValueError("coefficient identity count must equal coefficient axis extent")
        if len(set(self.coefficient_ids)) != len(self.coefficient_ids):
            raise ValueError("coefficient logical identities must be unique")
        if not self.mechanisms:
            raise ValueError("at least one mechanism is required")
        if len({m.identity for m in self.mechanisms}) != len(self.mechanisms):
            raise ValueError("mechanism identities must be unique")
        coefficients = set(self.coefficient_ids)
        for m in self.mechanisms:
            if m.coefficient not in coefficients:
                raise ValueError(f"mechanism {m.identity} binds an unknown coefficient")
            for arg in m.arguments:
                if arg.axis != 0 or arg.index >= self.input_axis.extent:
                    raise ValueError("argument incidence is outside the declared input axis")
            for out in m.outputs:
                if out.axis != 0 or out.index >= self.output_axis.extent:
                    raise ValueError("output contribution is outside the declared output axis")
        # Multiple writes to a coordinate require one explicit assembly identity.
        writers: dict[int, set[Identity]] = {}
        for m in self.mechanisms:
            for out in m.outputs:
                writers.setdefault(out.index, set()).add(out.assembly)
        if any(len(owners) != 1 for owners in writers.values()):
            raise ValueError("additive writers to an output coordinate need one assembly identity")


_loaded_library: Optional[Path] = None


def _load_native_library() -> None:
    global _loaded_library
    raw = os.environ.get("CELLERATORCH_NATIVE_LIBRARY")
    if not raw:
        raise RuntimeError(
            "set CELLERATORCH_NATIVE_LIBRARY to the CelleraTorch mechanism "
            "operator shared library built against this PyTorch installation"
        )
    path = Path(raw).expanduser().resolve()
    if _loaded_library == path:
        return
    if not path.is_file():
        raise FileNotFoundError(f"CelleraTorch native library not found: {path}")
    torch.ops.load_library(str(path))
    _loaded_library = path


def _flatten_identity(items: Iterable[Identity]) -> list[int]:
    return [_signed_word(word) for identity in items for word in identity.words]


def _constructor_data(spec: MechanismSpec) -> tuple:
    arg_offsets = [0]
    arg_slots: list[int] = []
    arg_roles: list[Identity] = []
    arg_axes: list[int] = []
    arg_indices: list[int] = []
    output_offsets = [0]
    output_slots: list[int] = []
    output_roles: list[Identity] = []
    output_axes: list[int] = []
    output_indices: list[int] = []
    output_assemblies: list[Identity] = []
    output_scales: list[float] = []
    bindings = {logical_id: index for index, logical_id in enumerate(spec.coefficient_ids)}
    coefficient_bindings: list[int] = []
    for mechanism in spec.mechanisms:
        for arg in mechanism.arguments:
            arg_slots.append(arg.slot)
            arg_roles.append(arg.role)
            arg_axes.append(arg.axis)
            arg_indices.append(arg.index)
        arg_offsets.append(len(arg_slots))
        coefficient_bindings.append(bindings[mechanism.coefficient])
        for out in mechanism.outputs:
            output_slots.append(out.slot)
            output_roles.append(out.role)
            output_axes.append(out.axis)
            output_indices.append(out.index)
            output_assemblies.append(out.assembly)
            output_scales.append(float(out.scale))
        output_offsets.append(len(output_slots))
    axis_words = spec.input_axis.words + spec.output_axis.words + spec.coefficient_axis.words
    axis_extents = [spec.input_axis.extent, spec.output_axis.extent,
                    spec.coefficient_axis.extent]
    return (
        axis_words, axis_extents, _flatten_identity(spec.coefficient_ids),
        _flatten_identity(m.identity for m in spec.mechanisms),
        arg_offsets, arg_slots, _flatten_identity(arg_roles), arg_axes, arg_indices,
        coefficient_bindings, output_offsets, output_slots,
        _flatten_identity(output_roles), output_axes, output_indices,
        _flatten_identity(output_assemblies), output_scales,
    )


_owner_parameters: weakref.WeakValueDictionary[int, nn.Parameter] = weakref.WeakValueDictionary()


def _parameter_for(handle: object) -> nn.Parameter:
    view = handle.coefficients()
    key = int(view.data_ptr())
    saved = _owner_parameters.get(key)
    if saved is not None:
        if getattr(saved, "_cellerator_native_handle", None) is not handle:
            raise RuntimeError("native coefficient storage address is already registered to another owner")
        return saved
    parameter = nn.Parameter(view, requires_grad=True)
    parameter._cellerator_native_handle = handle
    parameter._cellerator_owner_key = key
    parameter._cellerator_expected_version = parameter._version
    _owner_parameters[key] = parameter
    return parameter


class BiologicalTensor:
    """Tensor paired with the biological axis identity it inhabits."""

    def __init__(self, tensor: Tensor, axis: Axis):
        if not isinstance(tensor, Tensor):
            raise TypeError("BiologicalTensor.tensor must be a torch.Tensor")
        self.tensor = tensor
        self.axis = axis


class MechanismModule(nn.Module):
    """Torch module backed by a prepared native product-mechanism program."""

    def __init__(self, spec: MechanismSpec, initial_coefficients: Tensor,
                 *, max_batch: int, max_live_forwards: int = 8,
                 precision: str = "f32", _handle: object | None = None):
        super().__init__()
        if precision not in ("f32", "mixed_f16"):
            raise ValueError("precision must be 'f32' or 'mixed_f16'")
        if max_batch <= 0 or max_live_forwards <= 0:
            raise ValueError("prepared batch and live-forward capacities must be positive")
        if initial_coefficients.dtype != torch.float32 or not initial_coefficients.is_cuda:
            raise ValueError("native coefficients require a CUDA float32 tensor")
        if initial_coefficients.ndim != 1 or initial_coefficients.numel() != len(spec.coefficient_ids):
            raise ValueError("coefficient tensor shape must match the logical coefficient axis")
        if not initial_coefficients.is_contiguous():
            raise ValueError("initial coefficient tensor must be contiguous")
        if precision == "mixed_f16":
            finite = bool(torch.isfinite(initial_coefficients).all().item())
            in_half_range = bool((initial_coefficients.abs() < 65520.0).all().item())
            if not finite or not in_half_range:
                raise ValueError("mixed coefficients must be finite and round to finite FP16 values")
        if _handle is None:
            _load_native_library()
            precision_code = 0 if precision == "f32" else 1
            self._native = torch.classes.celleratorch.Mechanism(
                initial_coefficients, *_constructor_data(spec), int(max_batch),
                int(max_live_forwards), precision_code,
            )
        else:
            self._native = _handle
        self.spec = spec
        self.precision = precision
        self.max_batch = int(max_batch)
        self.max_live_forwards = int(max_live_forwards)
        self.coefficients = _parameter_for(self._native)
        self._owner_key = int(self.coefficients.data_ptr())
        # A version baseline detects ordinary in-place mutation outside the guard.
        self.coefficients._cellerator_expected_version = self.coefficients._version

    def forward(self, x: Tensor | BiologicalTensor) -> Tensor:
        if isinstance(x, BiologicalTensor):
            if x.axis != self.spec.input_axis:
                raise ValueError("input biological axis identity does not match prepared mechanism")
            x = x.tensor
        if x.ndim != 2 or x.shape[1] != self.spec.input_axis.extent:
            raise ValueError("input must have shape [batch, declared input-axis extent]")
        if x.shape[0] > self.max_batch:
            raise ValueError("batch exceeds the prepared mechanism capacity")
        return torch.ops.celleratorch.mechanism_apply(
            x, self.coefficients, self._native, self.spec.input_axis.words
        )

    @classmethod
    def shared_view(cls, owner: "MechanismModule") -> "MechanismModule":
        """Create another module view that reuses the exact prepared owner/program."""
        return cls(owner.spec, owner.coefficients, max_batch=owner.max_batch,
                   max_live_forwards=owner.max_live_forwards,
                   precision=owner.precision, _handle=owner._native)

    def _apply(self, fn, recurse: bool = True):  # noqa: ANN001
        raise RuntimeError("native coefficient storage cannot be moved or recast with Module.to/_apply")

    def _load_from_state_dict(self, *args, **kwargs):
        state_dict = args[0] if args else kwargs.get("state_dict", {})
        prefix = args[1] if len(args) > 1 else kwargs.get("prefix", "")
        if prefix + "coefficients" in state_dict:
            raise RuntimeError("restore native mechanisms with load_checkpoint(), not assign/copy state_dict")
        return super()._load_from_state_dict(*args, **kwargs)


def _unique_modules(modules: MechanismModule | nn.Module | Iterable[nn.Module]) -> list[MechanismModule]:
    roots = [modules] if isinstance(modules, nn.Module) else list(modules)
    result: list[MechanismModule] = []
    seen: set[int] = set()
    for root in roots:
        for module in root.modules():
            if isinstance(module, MechanismModule) and id(module._native) not in seen:
                seen.add(id(module._native))
                result.append(module)
    return result


def _validate_optimizer(optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError("guarded_step currently supports torch.optim.Adam only")
    for group in optimizer.param_groups:
        # Torch's automatic (None) selection can choose foreach/fused kernels
        # on CUDA. Resolve that default to the supported single-tensor path.
        for key in ("foreach", "fused"):
            if group.get(key, False) is None:
                group[key] = False
        if group.get("weight_decay", 0) != 0:
            raise ValueError("guarded native Adam requires weight_decay=0")
        for key in ("foreach", "fused", "capturable", "differentiable"):
            if group.get(key, False) is not False:
                raise ValueError(f"guarded native Adam does not support {key}=True")
        if group.get("maximize", False):
            raise ValueError("guarded native Adam does not support maximize=True")
        if group.get("amsgrad", False):
            raise ValueError("guarded native Adam does not support AMSGrad")
        if isinstance(group.get("lr"), Tensor):
            raise ValueError("guarded native Adam requires a scalar Python learning rate")
    params = [p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in params}) != len(params):
        raise ValueError("optimizer contains duplicate parameter ownership")
    storage_keys = [
        (p.device.type, p.device.index, p.untyped_storage().data_ptr(),
         p.storage_offset(), tuple(p.shape), tuple(p.stride()))
        for p in params
    ]
    if len(set(storage_keys)) != len(storage_keys):
        raise ValueError("optimizer contains aliased parameter storage")
    if any(p.grad is not None and p.grad.is_sparse for p in params):
        raise ValueError("guarded native Adam does not support sparse gradients")
    return params


def guarded_step(modules: MechanismModule | nn.Module | Iterable[nn.Module],
                 optimizer: torch.optim.Optimizer, *, scaler=None,
                 closure=None) -> bool:
    """Perform one stock Adam step and publish each changed native owner once.

    Returns ``False`` when GradScaler found non-finite gradients. It unscales
    once, skips Adam and native publication, then updates the scaler's scale.
    Ordinary Torch parameters in the optimizer are stepped in the same call.
    """
    if closure is not None:
        raise ValueError("guarded_step does not support optimizer closures")
    native_modules = _unique_modules(modules)
    parameters = _validate_optimizer(optimizer)
    parameter_set = {id(p) for p in parameters}
    native_by_parameter = {id(m.coefficients): m for m in native_modules}
    for module in native_modules:
        if id(module.coefficients) not in parameter_set:
            raise ValueError("optimizer must include every native coefficient owner")
        if (module.coefficients._cellerator_owner_key != module._owner_key
                or module.coefficients._cellerator_native_handle is not module._native):
            raise RuntimeError("native coefficient view lost its registered owner")
        if module.coefficients._version != module.coefficients._cellerator_expected_version:
            raise RuntimeError("native coefficients changed outside guarded_step")
        module._native.validate_coefficients(module.coefficients)
        module._native.preflight_update()
    # Reject native parameter views hidden in the optimizer when their owner
    # was omitted from the guarded module set. This check is native because a
    # tensor alias can be created without going through this Python registry.
    try:
        is_native_coefficient = torch.ops.celleratorch.is_native_coefficient
        for parameter in parameters:
            if bool(is_native_coefficient(parameter)) and id(parameter) not in native_by_parameter:
                raise ValueError(
                    "optimizer contains a native coefficient whose owner is absent from modules"
                )
    except AttributeError as exc:
        raise RuntimeError("CelleraTorch native coefficient ownership op is not registered") from exc
    if not any(parameter.grad is not None for parameter in parameters):
        return True
    if scaler is not None:
        if not scaler.is_enabled():
            scaler = None
        else:
            scaler.unscale_(optimizer)
    finite = all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in parameters)
    if not finite:
        if scaler is None:
            raise FloatingPointError("non-finite gradient rejected before optimizer mutation")
        scaler.update()
        return False

    changed = [m for m in native_modules if m.coefficients.grad is not None]
    begun: list[MechanismModule] = []
    try:
        for module in changed:
            module._native.begin_update()
            begun.append(module)
        optimizer.step()
        for module in begun:
            if not bool(torch.isfinite(module.coefficients).all().item()):
                raise FloatingPointError("Adam produced non-finite native coefficients")
            if (module.precision == "mixed_f16"
                    and not bool(torch.isfinite(module.coefficients.to(torch.float16)).all().item())):
                raise FloatingPointError("Adam produced coefficients outside finite FP16 storage range")
        for module in begun:
            module._native.publish_update()
            module.coefficients._cellerator_expected_version = module.coefficients._version
        if scaler is not None:
            scaler.update()
        return True
    except Exception:
        for module in native_modules:
            try:
                module._native.poison()
            except Exception:
                pass
        raise


def _identity_data(identity: Identity) -> list[int]:
    return [identity.low, identity.high]


def _axis_data(axis: Axis) -> dict:
    return {
        "domain": _identity_data(axis.domain), "order": _identity_data(axis.order),
        "geometry": _identity_data(axis.geometry), "partition": _identity_data(axis.partition),
        "extent": axis.extent,
    }


def _spec_data(spec: MechanismSpec) -> dict:
    return {
        "input_axis": _axis_data(spec.input_axis),
        "output_axis": _axis_data(spec.output_axis),
        "coefficient_axis": _axis_data(spec.coefficient_axis),
        "coefficient_ids": [_identity_data(i) for i in spec.coefficient_ids],
        "mechanisms": [{
            "identity": _identity_data(m.identity),
            "coefficient": _identity_data(m.coefficient),
            "arguments": [{"slot": a.slot, "role": _identity_data(a.role),
                           "axis": a.axis, "index": a.index} for a in m.arguments],
            "outputs": [{"slot": o.slot, "role": _identity_data(o.role),
                         "axis": o.axis, "index": o.index,
                         "assembly": _identity_data(o.assembly), "scale": o.scale}
                        for o in m.outputs],
        } for m in spec.mechanisms],
    }


def _axis_from_data(value: dict) -> Axis:
    return Axis(Identity(*value["domain"]), Identity(*value["order"]),
                Identity(*value["geometry"]), Identity(*value["partition"]),
                int(value["extent"]))


def _spec_from_data(value: dict) -> MechanismSpec:
    mechanisms = tuple(Mechanism(
        Identity(*m["identity"]), Identity(*m["coefficient"]),
        tuple(Incidence(int(a["slot"]), Identity(*a["role"]),
                        int(a["index"]), int(a["axis"])) for a in m["arguments"]),
        tuple(OutputContribution(int(o["slot"]), Identity(*o["role"]),
                                 int(o["index"]), Identity(*o["assembly"]),
                                 float(o["scale"]), int(o["axis"])) for o in m["outputs"]),
    ) for m in value["mechanisms"])
    return MechanismSpec(
        _axis_from_data(value["input_axis"]), _axis_from_data(value["output_axis"]),
        _axis_from_data(value["coefficient_axis"]),
        tuple(Identity(*i) for i in value["coefficient_ids"]), mechanisms,
    )


def _cpu_value(value):
    if isinstance(value, Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {k: _cpu_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_value(v) for v in value)
    if isinstance(value, list):
        return [_cpu_value(v) for v in value]
    return value


def _model_mechanisms(model: nn.Module) -> tuple[list[tuple[str, MechanismModule]], set[int]]:
    modules = [(name, child) for name, child in model.named_modules(remove_duplicate=False)
               if isinstance(child, MechanismModule)]
    if not modules:
        raise ValueError("checkpoint model must contain at least one MechanismModule")
    by_owner: dict[int, tuple[str, MechanismModule]] = {}
    for name, module in modules:
        by_owner.setdefault(module._owner_key, (name, module))
    return list(by_owner.values()), {id(module.coefficients) for _, module in modules}


def _named_parameters_all(model: nn.Module) -> dict[str, nn.Parameter]:
    return dict(model.named_parameters(remove_duplicate=False))


def _optimizer_named_groups(model: nn.Module, optimizer: torch.optim.Optimizer):
    _validate_optimizer(optimizer)
    names_by_parameter: dict[int, str] = {}
    for name, parameter in _named_parameters_all(model).items():
        names_by_parameter.setdefault(id(parameter), name)
    groups = []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            name = names_by_parameter.get(id(parameter))
            if name is None:
                raise ValueError("every optimizer parameter must be registered by the checkpoint model")
            names.append(name)
        groups.append((names, {k: _cpu_value(v) for k, v in group.items() if k != "params"}))
    owners, _ = _model_mechanisms(model)
    optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]
    counts = {id(p): sum(id(candidate) == id(p) for candidate in optimizer_params)
              for _, module in owners for p in (module.coefficients,)}
    if any(count != 1 for count in counts.values()):
        raise ValueError("checkpoint optimizer must contain each native coefficient owner exactly once")
    return groups, names_by_parameter


def save_checkpoint(path: str | os.PathLike, model: nn.Module,
                    optimizer: torch.optim.Optimizer) -> None:
    """Save an entire composed model, native logical owners, and named Adam state."""
    owners, native_parameter_ids = _model_mechanisms(model)
    groups, names_by_parameter = _optimizer_named_groups(model, optimizer)
    model_state = model.state_dict()
    native_names = {name for name, parameter in _named_parameters_all(model).items()
                    if id(parameter) in native_parameter_ids}
    ordinary_state = {name: _cpu_value(value) for name, value in model_state.items()
                      if name not in native_names}
    native_records = []
    for name, module in owners:
        module._native.preflight_update()
        values = module._native.snapshot().detach().to(device="cpu", dtype=torch.float32)
        native_records.append({
            "module_name": name,
            "spec": _spec_data(module.spec),
            "precision": module.precision,
            "coefficient_ids": [_identity_data(i) for i in module.spec.coefficient_ids],
            "coefficients": values,
        })
    named_optimizer_state = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names_by_parameter[id(parameter)]
            if parameter in optimizer.state:
                named_optimizer_state[name] = _cpu_value(optimizer.state[parameter])
    torch.save({
        "format": "celleratorch-training-v2",
        "ordinary_model": ordinary_state,
        "native_owners": native_records,
        "optimizer_groups": [{"parameter_names": names, "options": options}
                             for names, options in groups],
        "optimizer_state": named_optimizer_state,
    }, path)


def _validate_spec_match(saved: MechanismSpec, target: MechanismSpec) -> tuple[list[int], list[int]]:
    saved_coeff_axis, target_coeff_axis = saved.coefficient_axis, target.coefficient_axis
    same_coefficient_domain = (
        saved_coeff_axis.domain == target_coeff_axis.domain
        and saved_coeff_axis.geometry == target_coeff_axis.geometry
        and saved_coeff_axis.partition == target_coeff_axis.partition
        and saved_coeff_axis.extent == target_coeff_axis.extent
    )
    if (saved.input_axis != target.input_axis or saved.output_axis != target.output_axis
            or not same_coefficient_domain or saved.mechanisms != target.mechanisms):
        raise ValueError("checkpoint biological mechanism declaration does not match target")
    saved_ids = [i.words for i in saved.coefficient_ids]
    target_ids = [i.words for i in target.coefficient_ids]
    if set(saved_ids) != set(target_ids) or len(saved_ids) != len(target_ids):
        raise ValueError("checkpoint coefficient logical identities do not match target mechanism")
    if saved_ids != target_ids and saved_coeff_axis.order == target_coeff_axis.order:
        raise ValueError("a changed coefficient order requires a distinct biological order identity")
    return saved_ids, target_ids


def load_checkpoint(path: str | os.PathLike, model: nn.Module,
                    optimizer: torch.optim.Optimizer, *, map_location="cpu") -> None:
    """Restore ordinary model tensors, logical native owners, and named Adam state.

    The payload is fully checked before mutation. Any failure during mutation
    poisons all native owners so callers can retry from a whole checkpoint.
    """
    data = torch.load(path, map_location=map_location, weights_only=False)
    if data.get("format") != "celleratorch-training-v2":
        raise ValueError("unsupported CelleraTorch training checkpoint")
    owners, native_parameter_ids = _model_mechanisms(model)
    owner_by_name = {name: module for name, module in owners}
    if set(owner_by_name) != {record["module_name"] for record in data["native_owners"]}:
        raise ValueError("checkpoint native owner module paths do not match target model")
    permutations: dict[str, list[int]] = {}
    prepared_values: dict[str, Tensor] = {}
    for record in data["native_owners"]:
        module = owner_by_name[record["module_name"]]
        saved_spec = _spec_from_data(record["spec"])
        saved_ids, target_ids = _validate_spec_match(saved_spec, module.spec)
        if record["precision"] != module.precision:
            raise ValueError("checkpoint precision does not match prepared mechanism")
        permutations[record["module_name"]] = [saved_ids.index(i) for i in target_ids]
        values = record["coefficients"]
        if values.dtype != torch.float32 or values.ndim != 1 or values.numel() != len(saved_ids):
            raise ValueError("checkpoint native coefficient values have an invalid shape or dtype")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("checkpoint native coefficients must be finite")
        if module.precision == "mixed_f16" and not bool((values.abs() < 65520.0).all().item()):
            raise ValueError("checkpoint mixed coefficients exceed finite FP16 storage range")
        prepared_values[record["module_name"]] = values[permutations[record["module_name"]]].contiguous()

    current_state = model.state_dict()
    native_names = {name for name, parameter in _named_parameters_all(model).items()
                    if id(parameter) in native_parameter_ids}
    current_ordinary = {name: value for name, value in current_state.items() if name not in native_names}
    ordinary = data["ordinary_model"]
    if set(ordinary) != set(current_ordinary):
        raise ValueError("checkpoint ordinary model parameter/buffer names do not match target")
    restored_ordinary = {}
    for name, target in current_ordinary.items():
        value = ordinary[name]
        if not isinstance(value, Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"checkpoint ordinary tensor {name!r} has a mismatched shape or dtype")
        restored_ordinary[name] = value

    groups, names_by_parameter = _optimizer_named_groups(model, optimizer)
    saved_groups = data["optimizer_groups"]
    if len(groups) != len(saved_groups):
        raise ValueError("checkpoint optimizer group count does not match target")
    for (names, _), target_group, saved_group in zip(groups, optimizer.param_groups, saved_groups):
        if set(names) != set(saved_group["parameter_names"]) or len(names) != len(saved_group["parameter_names"]):
            raise ValueError("checkpoint optimizer parameter names do not match target groups")
        options = saved_group["options"]
        if set(options) != set(target_group) - {"params"}:
            raise ValueError("checkpoint optimizer option fields do not match target Adam")
        if (options.get("weight_decay", 0) != 0 or options.get("amsgrad", False)
                or options.get("maximize", False)
                or any(options.get(key, False) is not False
                       for key in ("foreach", "fused", "capturable", "differentiable"))):
            raise ValueError("checkpoint contains unsupported guarded Adam options")
    saved_optimizer_names = {name for group in saved_groups for name in group["parameter_names"]}
    if not set(data["optimizer_state"]).issubset(saved_optimizer_names):
        raise ValueError("checkpoint optimizer state references a parameter outside its groups")
    params_by_name = {name: parameter for name, parameter in _named_parameters_all(model).items()}
    # Prefer the same canonical name used during saving for a shared Parameter.
    params_by_name = {names_by_parameter[id(p)]: p for p in params_by_name.values()}
    coefficient_permutations: dict[str, list[int]] = {}
    for module_name, module in owners:
        parameter_name = names_by_parameter[id(module.coefficients)]
        permutation = permutations[module_name]
        previous = coefficient_permutations.setdefault(parameter_name, permutation)
        if previous != permutation:
            raise ValueError("one shared coefficient parameter has conflicting checkpoint permutations")
    for name, state in data["optimizer_state"].items():
        parameter = params_by_name.get(name)
        if parameter is None:
            raise ValueError(f"checkpoint optimizer state references unknown parameter {name!r}")
        for key, value in state.items():
            if key == "step":
                if isinstance(value, Tensor) and value.numel() != 1:
                    raise ValueError("checkpoint Adam step state must be scalar")
            elif key in ("exp_avg", "exp_avg_sq"):
                if not isinstance(value, Tensor) or value.shape != parameter.shape:
                    raise ValueError(f"checkpoint Adam state {key} does not match parameter {name!r}")
            else:
                raise ValueError(f"unsupported Adam checkpoint state field {key!r}")

    try:
        # A prior failed optimizer step may have queued partial writes on any
        # participating device. Drain those writes before restoring any part
        # of the model or native owner so checkpoint recovery cannot race them.
        cuda_devices = {
            tensor.device.index
            for tensor in list(model.parameters()) + list(model.buffers())
            + [p for group in optimizer.param_groups for p in group["params"]]
            if tensor.is_cuda and tensor.device.index is not None
        }
        cuda_devices.update(module._native.device() for _, module in owners)
        for device in sorted(cuda_devices):
            torch.cuda.synchronize(device)
        model.load_state_dict(restored_ordinary, strict=False)
        for name, module in owners:
            module._native.restore(prepared_values[name])
            module.coefficients._cellerator_expected_version = module.coefficients._version
        optimizer.state.clear()
        for name, state in data["optimizer_state"].items():
            parameter = params_by_name.get(name)
            if parameter is None:
                raise ValueError(f"checkpoint optimizer state references unknown parameter {name!r}")
            restored_state = {}
            for key, value in state.items():
                if isinstance(value, Tensor):
                    target_device = parameter.device if value.ndim > 0 else value.device
                    restored_value = value
                    permutation = coefficient_permutations.get(name)
                    if (permutation is not None and key in ("exp_avg", "exp_avg_sq")):
                        restored_value = restored_value[permutation]
                    restored_state[key] = restored_value.to(device=target_device).clone()
                else:
                    restored_state[key] = value
            optimizer.state[parameter] = restored_state
        for target_group, saved_group in zip(optimizer.param_groups, saved_groups):
            options = saved_group["options"]
            for key in tuple(target_group):
                if key != "params" and key in options:
                    target_group[key] = options[key]
        for module in model.modules():
            if isinstance(module, MechanismModule):
                module.coefficients._cellerator_expected_version = module.coefficients._version
    except Exception:
        for _, module in owners:
            try:
                module._native.poison()
            except Exception:
                pass
        raise
