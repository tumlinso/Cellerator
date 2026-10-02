"""Fresh CPU experiment restore; live tapes must drain before replacement.

This checkpoint contains the declared linear model, residual parameters, and
external Adam state. It does not serialize native compilation or GPU resources.
"""
from copy import deepcopy
import torch

from snapshot import SnapshotController
from regrowth import ResidualRegrowth, reset_adam_moments


def capture_training_checkpoint(controller, branch, optimizer):
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("checkpoint supports external torch.optim.Adam only")
    expected = list(branch.parameters())
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(expected) or any(a is not e for a, e in zip(actual, expected)):
        raise ValueError("optimizer must own residual parameters once in declared U,V order")
    if len(optimizer.param_groups) != 1:
        raise ValueError("checkpoint currently supports one Adam parameter group")
    return deepcopy({
        "schema_version": 1,
        "controller": controller.export_checkpoint(),
        "branch_config": {"input_dim": branch.U.shape[1],
                          "output_dim": branch.V.shape[0],
                          "capacity": branch.U.shape[0],
                          "dtype": str(branch.U.dtype)},
        "branch": branch.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_defaults": optimizer.defaults,
    })


def restore_training_checkpoint(payload, *, previous_controller=None):
    """Validate into fresh owners, then return a replacement session tuple.

    Caller installs the returned tuple only after this function succeeds. No
    existing model or optimizer is mutated by staging or a rejected payload.
    """
    if previous_controller is not None:
        previous_controller.assert_drained()
    required = {"schema_version", "controller", "branch_config", "branch",
                "optimizer", "optimizer_defaults"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("invalid training checkpoint schema")
    staged = deepcopy(payload)
    config = staged["branch_config"]
    if set(config) != {"input_dim", "output_dim", "capacity", "dtype"}:
        raise ValueError("invalid residual checkpoint configuration")
    dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}.get(config["dtype"])
    if dtype is None:
        raise ValueError("unsupported checkpoint dtype")
    controller = SnapshotController.from_checkpoint(staged["controller"])
    snapshot = controller.snapshot
    if dtype != snapshot.state.dtype:
        raise ValueError("residual and linear model dtype disagree")
    if config["input_dim"] != snapshot.state.numel() or config["output_dim"] != snapshot.readout.shape[0]:
        raise ValueError("residual and linear model dimensions disagree")
    branch = ResidualRegrowth(config["input_dim"], config["output_dim"],
                              config["capacity"], seed=0, dtype=dtype)
    expected_shapes = {"U": branch.U.shape, "V": branch.V.shape}
    for name, shape in expected_shapes.items():
        value = staged["branch"].get(name)
        if (not isinstance(value, torch.Tensor) or value.shape != shape or
                value.dtype != dtype or value.device.type != "cpu" or not torch.isfinite(value).all()):
            raise ValueError("invalid declared residual parameter storage")
    branch.load_state_dict(staged["branch"], strict=True)
    if not all(torch.isfinite(p).all() for p in branch.parameters()):
        raise ValueError("nonfinite residual parameters")
    optimizer = torch.optim.Adam(branch.parameters(), **staged["optimizer_defaults"])
    optimizer_payload = staged["optimizer"]
    if set(optimizer_payload) != {"state", "param_groups"} or len(optimizer_payload["param_groups"]) != 1:
        raise ValueError("invalid Adam checkpoint schema")
    group = optimizer_payload["param_groups"][0]
    parameter_ids = group.get("params", [])
    if len(parameter_ids) != 2 or len(set(parameter_ids)) != 2 or not set(optimizer_payload["state"]) <= set(parameter_ids):
        raise ValueError("invalid Adam parameter identity")
    if any(not torch.isfinite(torch.tensor(float(group.get(key, -1)))).item() or group.get(key, -1) < 0
           for key in ("lr", "eps", "weight_decay")):
        raise ValueError("invalid Adam hyperparameters")
    if len(group.get("betas", ())) != 2 or any(not 0 <= b < 1 for b in group["betas"]):
        raise ValueError("invalid Adam betas")
    for parameter_id, state in optimizer_payload["state"].items():
        if state and not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
            raise ValueError("incomplete Adam state")
        if state and group.get("amsgrad") and "max_exp_avg_sq" not in state:
            raise ValueError("missing AMSGrad state")
        parameter = list(branch.parameters())[parameter_ids.index(parameter_id)]
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in state and (not isinstance(state[name], torch.Tensor) or
                    state[name].dtype != parameter.dtype or state[name].device.type != "cpu"):
                raise ValueError("invalid declared Adam moment storage")
    optimizer.load_state_dict(staged["optimizer"])
    for parameter, state in optimizer.state.items():
        if state and not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
            raise ValueError("incomplete Adam state")
        if state and group.get("amsgrad") and "max_exp_avg_sq" not in state:
            raise ValueError("missing AMSGrad state")
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in state and (state[name].shape != parameter.shape or
                                  not torch.isfinite(state[name]).all()):
                raise ValueError("invalid Adam moments")
            if name in ("exp_avg_sq", "max_exp_avg_sq") and name in state and (state[name] < 0).any():
                raise ValueError("negative Adam second moments")
        if "step" in state and (not torch.isfinite(torch.as_tensor(state["step"])).all()
                                or torch.as_tensor(state["step"]).numel() != 1
                                or torch.as_tensor(state["step"]).item() < 0):
            raise ValueError("invalid Adam step")
    return controller, branch, optimizer


def refactor_training_transform(controller, branch, optimizer, transform,
                                new_coordinates, *, replica_maps=()):
    """Stage an equivalent full model, resetting incompatible incoming moments.

    Return replacement owners. The caller installs the tuple after success;
    output-side residual coefficients and their Adam moments are preserved.
    """
    controller.assert_drained()
    payload = capture_training_checkpoint(controller, branch, optimizer)
    staged_controller, staged_branch, staged_optimizer = restore_training_checkpoint(payload)
    old = controller.snapshot
    staged_controller.publish_transform(transform, new_coordinates,
        expected_epoch=old.structure_epoch, expected_coordinates=old.coordinates,
        replica_maps=replica_maps)
    transform = torch.as_tensor(transform, dtype=branch.U.dtype, device="cpu")
    with torch.no_grad():
        staged_branch.U.copy_(branch.U.detach() @ torch.linalg.inv(transform))
    reset_adam_moments(staged_optimizer, staged_branch.U)
    return staged_controller, staged_branch, staged_optimizer


def refactor_training_quotient(controller, branch, optimizer, projection,
                               embedding, new_coordinates, *, replica_maps=()):
    """Stage the full model on a caller-supplied invariant manifold."""
    controller.assert_drained()
    payload = capture_training_checkpoint(controller, branch, optimizer)
    staged_controller = SnapshotController.from_checkpoint(payload["controller"])
    old = controller.snapshot
    staged_controller.publish_quotient(projection, embedding, new_coordinates,
        expected_epoch=old.structure_epoch, expected_coordinates=old.coordinates,
        replica_maps=replica_maps)
    embedding = torch.as_tensor(embedding, dtype=branch.U.dtype, device="cpu")
    payload["controller"] = staged_controller.export_checkpoint()
    payload["branch_config"]["input_dim"] = embedding.shape[1]
    payload["branch"]["U"] = branch.U.detach() @ embedding
    parameters = list(branch.named_parameters())
    incoming_index = next(i for i, (name, _) in enumerate(parameters) if name == "U")
    incoming_id = payload["optimizer"]["param_groups"][0]["params"][incoming_index]
    incoming_state = payload["optimizer"]["state"].get(incoming_id, {})
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if name in incoming_state:
            incoming_state[name] = torch.zeros_like(payload["branch"]["U"])
    return restore_training_checkpoint(payload)
