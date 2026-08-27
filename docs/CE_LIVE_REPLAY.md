# CE-LIVE CPE2 replay v1

CE-LIVE-34 proves the existing persistence boundary without assigning semantic
authority to CellShard. Cellerator constructs a pointer-free CPE2 image with a
real FMP1 projection. The unchanged CellShard CPEXEC01 compatibility path stores
and reloads those bytes opaquely, validates its transport identity, and uploads
the image once on the caller stream.

After reload, Cellerator alone validates the CPE2 structure, epoch, geometry,
catalog, and image identities. Typed projection activation aliases the uploaded
payload without allocation, conversion, checksum rescanning, device selection,
or synchronization. The existing executable program enumerates the built-in
catalog, selects the legal FMP1 candidate, consumes an external mutable value
generation, and reports explicit execution metadata. Its CUDA output is checked
against an independent coordinate calculation.

Run the focused correctness controller with:

```bash
python /home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py run \
  --spec bench/ce_live/replay/cuda_controller.json --json
```

Run Compute Sanitizer with:

```bash
python /home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py run \
  --spec bench/ce_live/replay/sanitizer_controller.json --json
```

This test uses only the existing CPEXEC01 compatibility transport. It neither
waits for nor implements the independent future CPEXEC02 CellShard program.

Accepted V100 foreground correctness evidence is
`846d64e9-2e82-4d1f-a22f-96e9d6746b6a`. Compute Sanitizer memcheck evidence
is `347e9723-9eb7-483b-9db0-9466efbbac3d`. Both completed without a foreign GPU
process or action-worthy finding.
