# Download Helpers

The download layer is for local FASTQ acquisition only. Keep manifests declarative and keep credentials or site-specific secrets outside the repo.

Current entrypoint:

- `fetch_fastq.sh`: reads a TSV manifest and either prints or executes the required fetch commands

Supported provider values in the example script:

- `sra`: `prefetch` followed by `fasterq-dump`
- `direct`, `ena`, `geo-http`: URL download via `curl` or `wget`

The script writes into `${CELLERATOR_PAPER_DATA_ROOT}/raw_fastq/<target_relpath>` and appends a timestamped log under `PAPER_LOG_ROOT`.

Those env names are legacy operational names from the earlier paper-oriented wrapper surface. The documentation now treats this as a pipeline helper, but the shell contract is unchanged.
