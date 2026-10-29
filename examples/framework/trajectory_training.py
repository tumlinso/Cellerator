"""Two ragged cells learn one shared continuous regulatory surrogate on CPU.

Run: PYTHONPATH=python python examples/framework/trajectory_training.py
This synthetic example is a training-mechanics check, not biological validation.
"""
import torch
from torch import nn

from cellerator.adapters.torch import TorchModel
from cellerator.adapters.training import TrainingStep, TrainingTrajectory, TrajectoryWindowRunner


class RegulatoryDynamics(nn.Module):
    """Caller-chosen continuous production/decay surrogate with shared rates."""
    def __init__(self):
        super().__init__()
        self.log_production = nn.Parameter(torch.tensor(-0.4))
        self.log_decay = nn.Parameter(torch.tensor(-1.2))

    def forward(self, state, context):
        genes = state["genes"]
        production = self.log_production.exp() * torch.sigmoid(state["regulators"])
        genes = genes + context.dt * (production - self.log_decay.exp() * genes)
        return {"genes": genes, "regulators": state["regulators"], "gene_ids": state["gene_ids"]}


def initial_cells():
    # Different feature counts and identifiers; no dense population batch/padding.
    return (
        TrainingTrajectory("cell-a", {"genes": torch.tensor([.2, .4]),
                                     "regulators": torch.tensor([.8, -.2]),
                                     "gene_ids": ("gene-a", "gene-b")}),
        TrainingTrajectory("cell-b", {"genes": torch.tensor([.7]),
                                     "regulators": torch.tensor([.3]),
                                     "gene_ids": ("gene-c",)}),
    )


def observation_loss(prediction, target, context):
    # The scientific observation model and normalization are caller decisions.
    return (prediction - target).square().mean()


def main():
    model = TorchModel(RegulatoryDynamics(), lambda state, context: state["genes"])
    runner = TrajectoryWindowRunner(model, max_steps=2, reduction="mean_trajectories")
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    schedules = {
        "cell-a": [TrainingStep(.2), TrainingStep(.3, target=torch.tensor([.45, .50]))],
        "cell-b": [TrainingStep(.4, target=torch.tensor([.75]))],
    }
    cells = initial_cells()
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    result = runner.run_window(cells, schedules, observation_loss, optimizer=optimizer)
    assert result.transitions == 3 and result.observations == 2
    assert tuple(cell.id for cell in result.trajectories) == ("cell-a", "cell-b")
    assert [cell.time for cell in result.trajectories] == [.5, .4]
    assert [cell.state["genes"].numel() for cell in result.trajectories] == [2, 1]
    assert all(cell.state["genes"].grad_fn is None for cell in result.trajectories)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert torch.equal(cells[0].state["genes"], torch.tensor([.2, .4]))

    # Independent one-cell windows recover each contribution to the joint mean.
    contributions = []
    for cell in cells:
        reference = TorchModel(RegulatoryDynamics(), lambda state, context: state["genes"])
        one = TrajectoryWindowRunner(reference, max_steps=2, reduction="mean_trajectories")
        one.run_window([cell], {cell.id: schedules[cell.id]}, observation_loss, step_optimizer=False)
        contributions.append(tuple(p.grad.clone() for p in reference.parameters()))
    for index, parameter in enumerate(model.parameters()):
        expected = (contributions[0][index] + contributions[1][index]) / 2
        torch.testing.assert_close(parameter.grad, expected)

    # Continue persistent state, with cell-b inactive during the second window.
    continued = runner.run_window(result.trajectories, {
        "cell-a": [TrainingStep(.2, target=torch.tensor([.50, .52]))]},
        observation_loss, optimizer=optimizer)
    assert continued.trajectories[0].time == .7
    assert continued.trajectories[1].time == .4
    assert len(continued.trajectories[0].history) == 3
    assert len(continued.trajectories[1].history) == 1
    print(f"First window loss: {result.loss:.6f}; second: {continued.loss:.6f}")
    for cell in continued.trajectories:
        print(f"{cell.id}: time={cell.time:.1f}, genes={cell.state['genes'].tolist()}, "
              f"history={len(cell.history)} steps")
    print("Shared gradients equal the two private-cell contributions; explicit window boundaries verified.")


if __name__ == "__main__":
    main()
