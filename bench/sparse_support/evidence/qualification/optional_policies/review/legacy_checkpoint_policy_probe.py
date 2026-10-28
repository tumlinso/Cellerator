#!/usr/bin/env python3
"""Reproduce relation-policy behavior for a pre-policy in-progress checkpoint."""

from cellerator.torch.sparse_support import _validate_relation_input_policy


def main():
    # Before relation_input_policy was introduced, Karyoplast's only relation
    # source was the current event. Its v3 checkpoint state has an event cursor
    # but cannot contain the new policy key.
    legacy_application_state = {
        "event_position": 7,
        "latent_state": "checkpointed",
        "last_event_ids": "checkpointed",
    }
    _validate_relation_input_policy(legacy_application_state, "current_event")

    # New opted-in checkpoints preserve their stored policy. Because the public
    # forward default is still current_event, resuming one without repeating the
    # option is treated as an attempted midstream policy change.
    history_application_state = {
        "event_position": 7,
        "relation_input_policy": "eligible_history",
        "windows": ((10, 20),),
    }
    try:
        _validate_relation_input_policy(history_application_state, "current_event")
    except ValueError as exc:
        message = str(exc)
        assert "cannot change relation_input" in message, message
    else:
        raise AssertionError("default current_event unexpectedly replaced history policy")

    # A window refresh is allowed as long as the input policy remains stable.
    history_application_state["windows"] = ((11, 21),)
    _validate_relation_input_policy(history_application_state, "eligible_history")
    print("legacy checkpoint defaults to current_event; opted-in checkpoint requires"
          " explicit policy on resume; same-policy window refresh is accepted")


if __name__ == "__main__":
    main()
