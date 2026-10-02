# Fresh activation review

`prepare_activation.py` reads all three native authorities through the installed read-only `SemanticReader.state()` and `.workflow()` interfaces. It never claims work, applies plans, changes authority, runs workloads or dispatches agents.

The controller supplies the reviewed JSON list of all 27 original runs. Each row names `project`, `run_id`, `root_task_id`, `disposition`, and `reason`; deferred rows also require `preservation_owner` and `trigger`. Only the known Cellerator AMP permission lock and GEO deferral are accepted as original nonterminal runs. After import, each expected IS1 run is classified as preserved until explicit implementation dispatch. The original sealed package remains unchanged.

Pass one `--expected-head PROJECT=SHA` for each of `cellerator`, `baseplane`, and `glasshelix`, using the exact accepted commits after intentional bootstrap commits. The script rejects tracked dirty sources, unexpected/missing runs, active claims/dispatches/agents/children, recovery needs, blocking messages, unfinished predecessors, unaccepted pending commits and unexpected unresolved integration queues. Pending commits must have done task owners and be ancestors of their accepted HEAD. Historical NF1 queue records remain recorded and preserved.

Without `--approve-reviewed-inventory`, the script validates and prints a summary without writing any files. The root controller runs approval only after reviewing classifications and reconciliation:

```sh
python planning/integrated-substrate-bootstrap-2026-10-02/prepare_activation.py \
  --reviewed-inventory /absolute/path/to/controller-reviewed-runs.json \
  --expected-head cellerator=ACCEPTED_CE_SHA \
  --expected-head baseplane=ACCEPTED_BP_SHA \
  --expected-head glasshelix=ACCEPTED_GH_SHA \
  --reviewed-by 'root controller' \
  --approve-reviewed-inventory
```

Approval writes private ignored `planning/integrated-substrate-v1/results/authority-review.json` in each repository, central `results/activation-review.json` in Cellerator's package, and each package's ignored `local-config.json`. Evidence uses explicit field whitelists: no sessions, tokens, message text or opaque workspace results are exported. Each review binds current task states, dependencies/scopes, all runs and lanes/queues/workspaces, patch artifacts, pending commit checks, integration records, messages and rendezvous to the accepted heads and native authority fingerprints. The central review hashes those evidence files.

Run each original package's activation gate immediately after approval, using its generated local config. Reviews expire after 600 seconds and must be refreshed after imports or source commits. Importing the program and creating an activation review do not authorize dispatch by themselves; the controller retains task lifecycle and final acceptance.
