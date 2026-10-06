# Local binding migration delivery

Migration source is committed on `codex/python-bindings` and integrated into the canonical checkout after the required adapter and consumer checks. This directory holds verified local backups, without committing compiled wheels or bundles to Git.

`checkpoint.bundle` contains the initial migration commit `c041eb63410b75211bfbc7ec4db05f93eb51b1ef`. `final.bundle` contains the completed migration branch with prerequisite `ebaa6d295423f8d5267027b5665287572f65ba6b`; verify with `git bundle verify planning/python-bindings-delivery/final.bundle`. The `host/` and `torch/` directories contain the tested Python wheels. Their source/build checks and SHA256 values are recorded in `planning/python-bindings-absorb/README.md` and its validation records.

The concurrently edited FP64 sources and generated Todo projections were preserved. No new numerical envelope, performance result, or biological validation is claimed.
