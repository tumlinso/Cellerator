#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pipeline_root="$(cd "${script_dir}/.." && pwd)"
default_config="${pipeline_root}/config/local.env"

usage() {
    cat <<'EOF'
Usage:
  fetch_fastq.sh [--config PATH] [--run] MANIFEST.tsv

Reads a tab-separated manifest with columns:
  dataset_id  provider  accession  target_relpath  layout  urls  notes

By default the script prints the command plan without executing it.
Use --run to perform the downloads.
EOF
}

config_path="${default_config}"
manifest_path=""
execute=0

while (($# > 0)); do
    case "$1" in
        --config)
            shift
            config_path="${1:-}"
            ;;
        --run)
            execute=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -n "${manifest_path}" ]]; then
                printf 'unexpected argument: %s\n' "$1" >&2
                usage >&2
                exit 1
            fi
            manifest_path="$1"
            ;;
    esac
    shift
done

if [[ -z "${manifest_path}" ]]; then
    usage >&2
    exit 1
fi

if [[ -f "${config_path}" ]]; then
    # shellcheck disable=SC1090
    source "${config_path}"
fi

: "${CELLERATOR_PAPER_DATA_ROOT:?set CELLERATOR_PAPER_DATA_ROOT in the env or config file}"

raw_root="${CELLERATOR_PAPER_DATA_ROOT%/}/raw_fastq"
log_root="${PAPER_LOG_ROOT:-${CELLERATOR_PAPER_WORK_ROOT:-${CELLERATOR_PAPER_DATA_ROOT%/}/work}/logs}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_log="${log_root%/}/download_${timestamp}.log"

mkdir -p "${raw_root}" "${log_root}"

PAPER_PREFETCH_BIN="${PAPER_PREFETCH_BIN:-prefetch}"
PAPER_FASTERQ_DUMP_BIN="${PAPER_FASTERQ_DUMP_BIN:-fasterq-dump}"
PAPER_CURL_BIN="${PAPER_CURL_BIN:-curl}"
PAPER_WGET_BIN="${PAPER_WGET_BIN:-wget}"
PAPER_FASTQ_THREADS="${PAPER_FASTQ_THREADS:-8}"

log() {
    printf '%s\n' "$*" | tee -a "${run_log}"
}

run_cmd() {
    local -a cmd=("$@")
    printf '+ ' | tee -a "${run_log}"
    printf '%q ' "${cmd[@]}" | tee -a "${run_log}"
    printf '\n' | tee -a "${run_log}"
    if ((execute)); then
        "${cmd[@]}"
    fi
}

download_url() {
    local url="$1"
    local target_dir="$2"
    local out_path="${target_dir}/$(basename "${url}")"
    if command -v "${PAPER_CURL_BIN}" >/dev/null 2>&1; then
        run_cmd "${PAPER_CURL_BIN}" -L --fail --retry 3 -o "${out_path}" "${url}"
    elif command -v "${PAPER_WGET_BIN}" >/dev/null 2>&1; then
        run_cmd "${PAPER_WGET_BIN}" -O "${out_path}" "${url}"
    else
        printf 'neither curl nor wget is available for URL download\n' >&2
        exit 1
    fi
}

log "manifest=${manifest_path}"
log "mode=$([[ ${execute} -eq 1 ]] && printf 'run' || printf 'dry-run')"
log "raw_root=${raw_root}"

while IFS=$'\037' read -r dataset_id provider accession target_relpath layout urls notes; do
    [[ -z "${dataset_id}" || "${dataset_id}" == "dataset_id" || "${dataset_id}" == \#* ]] && continue

    target_dir="${raw_root%/}/${target_relpath}"
    mkdir -p "${target_dir}"

    log ""
    log "[dataset:${dataset_id}] provider=${provider} accession=${accession} layout=${layout}"
    [[ -n "${notes}" ]] && log "notes=${notes}"

    case "${provider}" in
        sra)
            if [[ -z "${accession}" ]]; then
                printf 'missing accession for dataset %s\n' "${dataset_id}" >&2
                exit 1
            fi
            run_cmd "${PAPER_PREFETCH_BIN}" "${accession}" --output-directory "${target_dir}"
            run_cmd "${PAPER_FASTERQ_DUMP_BIN}" "${accession}" --split-files -e "${PAPER_FASTQ_THREADS}" -O "${target_dir}"
            ;;
        direct|ena|geo-http)
            if [[ -z "${urls}" ]]; then
                printf 'missing urls for dataset %s\n' "${dataset_id}" >&2
                exit 1
            fi
            IFS=',' read -r -a url_list <<< "${urls}"
            for url in "${url_list[@]}"; do
                [[ -z "${url}" ]] && continue
                download_url "${url}" "${target_dir}"
            done
            ;;
        *)
            printf 'unsupported provider %s for dataset %s\n' "${provider}" "${dataset_id}" >&2
            exit 1
            ;;
    esac
done < <(awk -F'\t' 'BEGIN { OFS = "\037" } { print $1, $2, $3, $4, $5, $6, $7 }' "${manifest_path}")

log ""
log "finished"
