# Refresh activation during implementation

`controller_activation.py` captures native authority through read-only semantic
views. Actual claims come from task state; session identity comes from the
unique matching workflow dispatch. Root invokes it after
committing accepted source in all three repositories and refreshing native
controller heartbeats. Stale session/dispatch recovery still blocks capture. Supply reviewed exact
heads, the original 27-row inventory, and each currently claimed IS1 task with
its root controller session. Omit task/session pairs for repositories without a
current claim. Every other claim, agent, dispatch or active queue fails capture.
Allowed tasks must match the installed IS1 plan scope and actual run/dispatch.

From Cellerator, substitute the three reviewed commit identities:

```sh
python3 -B planning/integrated-substrate-v1/adoption/controller_activation.py \
  --reviewed-inventory planning/integrated-substrate-bootstrap-2026-10-02/reviewed-run-inventory.json \
  --expected-head cellerator=CE_SHA \
  --expected-head baseplane=BP_SHA \
  --expected-head glasshelix=GH_SHA \
  --allowed-current-task cellerator=CE-IS1-ADOPT \
  --controller-session cellerator=b8123753-136b-4c59-a196-30e83cfb03d1 \
  --allowed-current-task baseplane=BP-IS1-ADOPT \
  --controller-session baseplane=08882441-2288-4b0b-bc09-9238cdfa16d0 \
  --allowed-current-task glasshelix=GH-IS1-ADOPT \
  --controller-session glasshelix=15b33f5f-4970-43e0-9004-386b6029a511 \
  --reviewed-by 'root controller'
```

This is a dry run. Append `--approve-reviewed-inventory` for publication. The
shared bootstrap publisher first invalidates approval, writes sanitized actual
claims and all evidence/configuration, then rechecks heads and authority
fingerprints before atomically publishing approval. Failure keeps approval
invalid. Configured review freshness remains 600 seconds. Refresh immediately
before gates if source or authority changed. The helper never mutates Todo
state, closes historical work, or approves predecessor substitutes.

Run the required activation command from each native ADOPT task sheet after
publication. Focused helper checks:

```sh
python3 -B planning/integrated-substrate-v1/adoption/test_controller_activation.py
```
