# Reconciled Project Control authority cursor

Todo: `CE-CCP1-A01-001`

This is a value-only, pointer-free observation record. It was captured from a
fresh Project Control process using `project_overview` followed by
`agent_status`; no field was inferred from an earlier report or from memory.

```json
{
  "schema_version": 1,
  "project_id": "cellerator",
  "project_uuid": "0ccaac37-dbbf-448e-a5f8-def197a70aba",
  "todo_revision": 3904,
  "todo_semantic_fingerprint": "fb53d368f34add25316a3aab81251f9dd19d43dcc5c4a672f1054cd271de3837",
  "workflow_revision": 3904,
  "workflow_semantic_fingerprint": "fb53d368f34add25316a3aab81251f9dd19d43dcc5c4a672f1054cd271de3837",
  "active_runs": [
    {
      "run_id": "CE-CCP1-RUN-V1",
      "status": "active"
    }
  ],
  "claims": [
    {
      "task_id": "CE-CCP1-A01-001",
      "observed_state": "active_claim",
      "classification": "claim_only_not_first_class_agent",
      "confidence": "authoritative",
      "heartbeat_lease_remaining_seconds": 6993
    }
  ],
  "observation_window": {
    "project_overview_observed_at": "2026-09-03T19:34:50Z",
    "agent_status_observed_at": "2026-09-03T19:34:54Z"
  },
  "provider_revision_skew": {
    "todo_export": 0,
    "todo_semantic_state": 0,
    "todo_status": 0,
    "todo_workflow": 0
  },
  "missing_provider_fields": [],
  "read_only_validation": {
    "todo_revision_before": 3904,
    "todo_revision_after": 3904,
    "workflow_revision_before": 3904,
    "workflow_revision_after": 3904,
    "semantic_fingerprint_unchanged": true,
    "workflow_fingerprint_unchanged": true
  }
}
```

## Provenance and reconciliation

- `project_overview`, observed `2026-09-03T19:34:50Z`, supplied the project
  UUID, Todo and workflow revisions, semantic fingerprints, active run, and
  provider skew.
- `agent_status`, observed `2026-09-03T19:34:54Z`, independently repeated the
  same identity, revisions, fingerprints, and active run and supplied the
  authoritative active-claim observation.
- Both reads reported revision 3904 and fingerprint
  `fb53d368f34add25316a3aab81251f9dd19d43dcc5c4a672f1054cd271de3837`.
  Their zero skew and unchanged before/after cursor show that the capture made
  no authority mutation.
- All requested provider fields were present in the fresh reconciled reads, so
  `missing_provider_fields` is explicitly the empty array. A separate
  long-lived observer read at `2026-09-03T19:34:33Z` reported the Todo semantic
  providers unavailable and therefore emitted null revisions and
  fingerprints. Those nulls were not substituted into this reconciled record;
  they are retained here as an explicit transient observer caveat.

The repository commit observed by both reads was
`31e491ed29de0fcde70259cbeab8c5c7ad353485`. This receipt records authority
state only and does not claim that repository commit as the eventual Todo
artifact commit.
