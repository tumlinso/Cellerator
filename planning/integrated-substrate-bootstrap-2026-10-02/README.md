# IS1 native bootstrap proposal

The sealed package stays in `../integrated-substrate-v1`. This proposal retains its IS1 tasks, prerequisites, run and lanes, and adds exact current definitions of the local completed predecessor roots to satisfy native standalone dependency validation. No existing interfaces, unrelated tasks or historical runs are included.

`native-validation.json` records supported Project Control validation and the bounded task diff. `composition-receipt.json` binds the original and derived plan hashes and describes the unavoidable metadata updates on exact predecessor replay. Native import increments those root task versions and updates their revision/time; the CE/GH closure gate revision also changes. Their definitions, states, results, completion commits and evidence remain intact.

Root-controlled import:

```sh
project-control plan apply --project cellerator --file /home/tumlinson/Cellerator/planning/integrated-substrate-bootstrap-2026-10-02/cellerator.todo-plan.json
python /home/tumlinson/Cellerator/planning/integrated-substrate-bootstrap-2026-10-02/verify_preservation.py --repo-root /home/tumlinson/Cellerator
```

The verification compares the native live read-only exports before and after import. Private raw observations stay ignored in `results/`; the durable receipt reports only counts, metadata allowances and errors.

The native run declares one ADOPT root lane with an exclusive workspace and bounded child lanes. Isolated children declare their integration destination. Existing historical lanes and managed workspaces remain separate. Bootstrap imports the program; dispatch requires the reviewed predecessor barrier and accepted source commits. `lane-and-gate-review.json` records the planned lanes and gate paths. Local config and source-bound activation reviews must be populated before dispatch, and activation must be rerun within its freshness limit.
