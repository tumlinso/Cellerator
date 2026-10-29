import unittest

from cellerator.events import CellLineage, Event, EventHistory, Trajectory, TrajectoryPoint


class EventTests(unittest.TestCase):
    def test_history_preserves_order_repeats_provenance_and_ejects_explicitly(self):
        history = EventHistory()
        first = Event("activate", time=1, target="g1", source="observed")
        history.append(first)
        history.append(Event("activate", time=2, target="g1", source="intervention"))
        self.assertEqual(tuple(history), (first, history[1]))
        self.assertEqual(history.select(target="g1", source="intervention"), (history[1],))
        removed = history.eject_before(keep_last=1)
        self.assertEqual(removed, (first,))
        self.assertEqual(history.base_index, 1)
        with self.assertRaises(ValueError):
            history.since(0)

    def test_trajectory_clone_policy_and_explicit_branch(self):
        trajectory = Trajectory(clone_policy="deepcopy")
        state = {"x": [1]}
        trajectory.record(0, state)
        state["x"].append(2)
        self.assertEqual(trajectory.final.state, {"x": [1]})
        branch = trajectory.branch()
        branch.record(1, {"x": [3]})
        self.assertEqual(len(trajectory), 1)
        self.assertEqual(len(branch), 2)

    def test_trajectory_accepts_callable_clone_and_checks_initial_time_order(self):
        trajectory = Trajectory(clone_policy=lambda state: state.copy())
        trajectory.record(0, {"x": 1})
        self.assertEqual(trajectory.final.state, {"x": 1})
        with self.assertRaises(ValueError):
            Trajectory((
                TrajectoryPoint(2, "later"),
                TrajectoryPoint(1, "earlier"),
            ))

    def test_lineage_carries_shared_history_and_explicitly_inherits_state(self):
        parent = CellLineage("parent", state={"memory": ["a"]}, time=1)
        parent.events.append(Event("pulse", time=1, source="environment"))
        left, right = parent.divide(("left", "right"))
        left.trajectory.final.state["memory"].append("left")
        self.assertEqual(right.trajectory.final.state, {"memory": ["a"]})
        self.assertEqual(left.events[0], parent.events[0])
        self.assertIs(left.parent, parent)
        self.assertEqual(left.ancestors(), (parent,))
        with self.assertRaises(ValueError):
            right.divide(("left",))


if __name__ == "__main__":
    unittest.main()
