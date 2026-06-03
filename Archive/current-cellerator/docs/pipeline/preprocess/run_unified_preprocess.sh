#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pipeline_root="$(cd "${script_dir}/.." && pwd)"
default_config="${pipeline_root}/config/local.env"

usage() {
    cat <<'EOF'
Usage:
  run_unified_preprocess.sh [--config PATH] [--run] MANIFEST.tsv

Reads a tab-separated manifest with columns:
  run_id  engine  input_relpath  output_relpath  reference_path  command_template  notes

By default the script prints the expanded command plan without executing it.
Use --run to execute the preprocessing commands.
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
CELLERATOR_PAPER_WORK_ROOT="${CELLERATOR_PAPER_WORK_ROOT:-${CELLERATOR_PAPER_DATA_ROOT%/}/work}"
PAPER_LOG_ROOT="${PAPER_LOG_ROOT:-${CELLERATOR_PAPER_WORK_ROOT%/}/logs}"
PAPER_PARABRICKS_BIN="${PAPER_PARABRICKS_BIN:-pbrun}"
PAPER_DEFAULT_REFERENCE="${PAPER_DEFAULT_REFERENCE:-}"

mkdir -p "${CELLERATOR_PAPER_WORK_ROOT}" "${PAPER_LOG_ROOT}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_log="${PAPER_LOG_ROOT%/}/preprocess_${timestamp}.log"

log() {
    printf '%s\n' "$*" | tee -a "${run_log}"
}

resolve_reference() {
    local reference_path="$1"
    if [[ -n "${reference_path}" ]]; then
        printf '%s' "${reference_path}"
    else
        printf '%s' "${PAPER_DEFAULT_REFERENCE}"
    fi
}

render_template() {
    local template="$1"
    local run_id="$2"
    local input_dir="$3"
    local output_dir="$4"
    local reference_path="$5"

    template="${template//\{parabricks_bin\}/${PAPER_PARABRICKS_BIN}}"
    template="${template//\{data_root\}/${CELLERATOR_PAPER_DATA_ROOT}}"
    template="${template//\{work_root\}/${CELLERATOR_PAPER_WORK_ROOT}}"
    template="${template//\{input_dir\}/${input_dir}}"
    template="${template//\{output_dir\}/${output_dir}}"
    template="${template//\{reference_path\}/${reference_path}}"
    template="${template//\{run_id\}/${run_id}}"
    printf '%s' "${template}"
}

write_context_file() {
    local context_path="$1"
    local run_id="$2"
    local engine="$3"
    local input_dir="$4"
    local output_dir="$5"
    local reference_path="$6"
    local command_template="$7"

    cat > "${context_path}" <<EOF
run_id=${run_id}
engine=${engine}
input_dir=${input_dir}
output_dir=${output_dir}
reference_path=${reference_path}
command_template=${command_template}
log_path=${run_log}
timestamp=${timestamp}
EOF
}

log "manifest=${manifest_path}"
log "mode=$([[ ${execute} -eq 1 ]] && printf 'run' || printf 'dry-run')"
log "data_root=${CELLERATOR_PAPER_DATA_ROOT}"
log "work_root=${CELLERATOR_PAPER_WORK_ROOT}"

while IFS=$'\037' read -r run_id engine input_relpath output_relpath reference_path command_template notes; do
    [[ -z "${run_id}" || "${run_id}" == "run_id" || "${run_id}" == \#* ]] && continue

    input_dir="${CELLERATOR_PAPER_DATA_ROOT%/}/${input_relpath}"
    output_dir="${CELLERATOR_PAPER_WORK_ROOT%/}/${output_relpath}"
    resolved_reference="$(resolve_reference "${reference_path}")"
    expanded_command="$(render_template "${command_template}" "${run_id}" "${input_dir}" "${output_dir}" "${resolved_reference}")"
    context_path="${output_dir}/run_context.env"

    mkdir -p "${output_dir}"

    log ""
    log "[run:${run_id}] engine=${engine}"
    log "input_dir=${input_dir}"
    log "output_dir=${output_dir}"
    [[ -n "${resolved_reference}" ]] && log "reference_path=${resolved_reference}"
    [[ -n "${notes}" ]] && log "notes=${notes}"
    log "+ ${expanded_command}"

    if ((execute)); then
        bash -lc "${expanded_command}" | tee -a "${run_log}"
        write_context_file "${context_path}" "${run_id}" "${engine}" "${input_dir}" "${output_dir}" "${resolved_reference}" "${command_template}"
    fi
done < <(awk -F'\t' 'BEGIN { OFS = "\037" } { print $1, $2, $3, $4, $5, $6, $7 }' "${manifest_path}")

log ""
log "finished"
