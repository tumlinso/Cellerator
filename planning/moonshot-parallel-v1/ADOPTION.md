# Cellerator adoption interface

The original Cellerator payload is copied byte-for-byte into `experiments/moonshot-parallel-v1/seeds`. All fourteen seed files are read-only and bound to the original archive manifest. Implementers adapt copies inside their own lane directories.

The agreement is `experiments/moonshot-parallel-v1/interfaces/contract.json`. It specifies nonowning canonical state identity, local slot incarnations, distinct structure/value/activity/parameter generations, one recorded read snapshot, explicit precision and derivative policies, output ownership, admission requirements and saved-primal rules. Prototype types remain experimental; final integration uses current native owners.

MMA writes only `mma`, STATE only `state`, DIFF only `diff`, REWRITE only `rewrite`, TRAJECTORY only `trajectory`. Shared seeds and interfaces are published by ADOPT. Intermediate merges publish accepted branch artifacts before downstream managed workspaces are created or reconciled.

The root reports companion tasks CE-MOON-020/030/040/050 active in disjoint Baseplane experiment families. Existing work and numerical evidence remain preserved. Broad native integration retains its required ML2-BIO ownership gate. No supersession, numerical execution or benchmark was performed by adoption.

Validation: `python3 -B planning/moonshot-parallel-v1/check_adoption.py --root .`. It verifies archive-matched immutable seeds and the agreed identity/precision/derivative/ownership fields. Temporary negative controls reject changed seeds and a missing parameter generation. Evidence is in `adoption-evidence.json`.

Proposed scoped commit: `Stage immutable Cellerator moonshot seeds and shared prototype contract`. Include only `experiments/moonshot-parallel-v1/seeds`, `experiments/moonshot-parallel-v1/interfaces`, `planning/moonshot-parallel-v1/check_adoption.py`, `ADOPTION.md`, and `adoption-evidence.json`. Root retains staging and commit ownership while other lanes write.
