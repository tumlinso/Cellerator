# Independent review: optional sparse application policies

## Scope and identity

Reviewed the current worktree policy patch in
`python/cellerator/torch/sparse_support.py` and
`tests/sparse_support/test_optional_policies.py`, against baseline
`b63942d8c2edb024b9eb8c238dd8d7a27d244445`. The review covers only the new
P∪D edge-value policy and Karyoplast relation-input policy.

- Current `sparse_support.py` SHA-256:
  `0fbdcbdb9cea80d73cf628c90e9b45c747ff58fe9c8330aaf1baa46d90f80e85`.
- Current policy test SHA-256:
  `9fe511df8e661e3a629e0d13507a0b793dc7f0a79cb2c5da14662dc975d3bc23`.

## Verdict

No confirmed correctness blocker remains in the reviewed patch. The workset
union produces a sorted unique P∪D ordinal set, gathers those shared-parameter
values into the canonical E-vector, and leaves other slots zero. This retains
all derivative-only candidates needed by gate gradients. The original `all`
policy remains the default and continues to gather the full vector directly.

Karyoplast keeps the current event as the default relation input. Its
`eligible_history` option changes only the relation's source tensor: recurrent
state transitions still consume the current event, and gate scoring still uses
the eligible-history aggregate plus the current latent-state mean. Repeated
history occurrences remain additive. A same-policy window refresh is accepted;
changing relation-input policy after stream progress is rejected.

## Adversarial checkpoint finding and disposition

The first patch rejected an in-progress pre-policy `cellerator-training-v3`
checkpoint that lacked `relation_input_policy`. Such checkpoints were produced
by the earlier implementation, whose only relation input was `current_event`.
The new validation helper initially treated the absent key as a policy change.
I reproduced the rejection with
[`legacy_checkpoint_policy_probe.py`](legacy_checkpoint_policy_probe.py).

The owner then added a compatibility rule: if `event_position > 0` and the
stored key is absent, interpret the prior policy as `current_event`. The probe
now passes this legacy state. An in-progress checkpoint that explicitly stored
`eligible_history` must resume with
`relation_input="eligible_history"`; omitting the argument requests the public
`current_event` default and correctly rejects that semantic change. The
continuation rule is explicit in the implementation, though the user-facing
qualification note should state that an opted-in history stream must repeat
its option when resuming.

## Validation and limits

Using a temporary package overlay whose `sparse_support.py` hash matched the
reviewed source, I ran:

```text
python -m pytest -q tests/sparse_support/test_optional_policies.py
6 passed, 2 skipped
```

The skipped tests are the scheduler-gated CUDA integrations. The CPU probe
passed against the owner-updated source and checks legacy default inference,
policy-lock rejection for an opted-in checkpoint when `current_event` is
requested, and acceptance of a window refresh when the history policy remains
unchanged. `git diff --check` passed.

The existing CUDA integration test uses a mocked transition and candidate
scorer to inspect the selected relation input; the source path confirms the
actual transition and gate-context arguments remain current-event and
history-plus-state respectively. No GPU run or real-data evaluation was done
by this review. The root's separate v14 admitted regression remains the
integrated CUDA evidence gate; these checks make no real-data or performance
claim.

Root integration note: subsequent test-only additions cover the actual native
relation's input, shared-parameter, and closed-gate VJPs, and the history
assertion now follows the declared FP16 input rounding. The production source
reviewed above is unchanged. The final test hash and successful 83-test GPU
integration are recorded in `../../v14-optional-policies/REPORT.md` and its
snapshot identity.
