# Unified Preprocessing

The preprocessing layer is intentionally command-template driven. Different datasets may need different tools, but they should still run through one manifest-and-log contract.

Current entrypoint:

- `run_unified_preprocess.sh`: reads a TSV manifest, expands placeholders, and either prints or executes the requested commands

Supported engines are labels for bookkeeping rather than hard gates. `parabricks` is treated as the default GPU-oriented path, but the manifest may also describe custom commands when a dataset needs a different toolchain.

Supported placeholders in `command_template`:

- `{parabricks_bin}`
- `{data_root}`
- `{work_root}`
- `{input_dir}`
- `{output_dir}`
- `{reference_path}`
- `{run_id}`

Each run writes a timestamped log and a small context file next to the output root.

The current script interface still uses legacy `CELLERATOR_PAPER_*` and `PAPER_*` env names. Those names are left intact here to avoid changing runtime behavior.
