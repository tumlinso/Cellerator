"""Small behavioral examples for generic trajectory inference."""
import math
import random
import unittest

from cellerator.inference import (
    Observation,
    Particle,
    particle_filter,
    rank_event_hypotheses,
    systematic_resample,
)
from cellerator.modeling import Model, StepResult


class InferenceTests(unittest.TestCase):
    def test_particle_filter_updates_state_and_weights(self):
        model = Model(
            lambda state, ctx: StepResult(state + ctx.dt),
            observation=lambda state, _: state,
        )
        posterior = particle_filter(
            model,
            [Particle(0.0), Particle(2.0)],
            [Observation(1.0, 1.0)],
            lambda predicted, observed: -((predicted - observed) ** 2),
            resample=None,
        )

        self.assertEqual(tuple(p.state for p in posterior.particles), (1.0, 3.0))
        expected_first_weight = 1.0 / (1.0 + math.exp(-4.0))
        self.assertAlmostEqual(posterior.weights[0], expected_first_weight)
        self.assertAlmostEqual(posterior.weights[1], 1.0 - expected_first_weight)
        self.assertAlmostEqual(posterior.log_evidence, math.log((1.0 + math.exp(-4.0)) / 2.0))
        self.assertAlmostEqual(sum(posterior.weights), 1.0)
        self.assertEqual(posterior.particles[0].provenance, (("observation", 0),))

    def test_arbitrary_state_uses_explicit_copy_policy(self):
        source = {"gene-a": 2}
        model = Model(lambda state, _: {"gene-a": state["gene-a"] + 1})
        posterior = particle_filter(
            model,
            [Particle(source), Particle(source)],
            [Observation(1.0, None)],
            lambda _, __: 0.0,
            copy_state=lambda state: dict(state),
            resample=None,
        )

        self.assertEqual(source, {"gene-a": 2})
        self.assertEqual(posterior.particles[0].state, {"gene-a": 3})
        self.assertIsNot(posterior.particles[0].state, posterior.particles[1].state)

    def test_resampling_tracks_ancestry(self):
        model = Model(lambda state, _: state, observation=lambda state, _: state)
        posterior = particle_filter(
            model,
            [Particle("supported"), Particle("excluded")],
            [Observation(0.0, "supported")],
            lambda predicted, observed: 0.0 if predicted == observed else -1000.0,
            resample=systematic_resample,
            resample_below=0.9,
            rng=random.Random(7),
        )

        self.assertEqual(posterior.resampled_at, (0.0,))
        self.assertEqual(posterior.effective_sample_size, 2.0)
        self.assertEqual({p.state for p in posterior.particles}, {"supported"})
        self.assertTrue(all(p.lineage for p in posterior.particles))

    def test_early_model_step_is_not_mislabeled_with_observation_time(self):
        model = Model(lambda state, ctx: StepResult(state + 1, time=ctx.time + ctx.dt / 2))
        with self.assertRaisesRegex(ValueError, "advance explicitly"):
            particle_filter(
                model, [Particle(0)], [Observation(2.0, None)], lambda _, __: 0.0
            )

    def test_event_hypotheses_keep_provenance_and_normalize_scores(self):
        candidates = rank_event_hypotheses(
            [("a", "b"), ("a", "c")],
            lambda events: 0.0 if events[-1] == "b" else -math.log(3.0),
            provenance=("observed", "reconstructed"),
        )

        self.assertEqual(candidates[0].events, ("a", "b"))
        self.assertEqual(candidates[0].provenance, "observed")
        self.assertAlmostEqual(sum(candidate.probability for candidate in candidates), 1.0)
        self.assertGreater(candidates[0].probability, candidates[1].probability)

    def test_all_impossible_candidates_are_reported(self):
        with self.assertRaisesRegex(ValueError, "zero observation likelihood"):
            particle_filter(
                Model(lambda state, _: state), [0, 1],
                [Observation(1.0, None)], lambda _, __: -math.inf,
            )


if __name__ == "__main__":
    unittest.main()
