#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/cleanup_cuda_12_9_only.sh [--execute] [--no-apt]

Default mode is a dry run. Use --execute to mutate the system.

This keeps NVIDIA HPC SDK CUDA 12.9 as the active CUDA toolkit, backs up and
removes HPC SDK CUDA-versioned 13.1 payloads, purges package-managed CUDA 12.8
packages, and repoints generic SDK symlinks at the HPC SDK CUDA 12.9 payloads.

Options:
  --execute  Actually move files, change symlinks, and run apt purge/autoremove.
  --no-apt   Skip apt purge/autoremove; only handle HPC SDK 13.1 payloads and symlinks.
  --help     Show this help.
EOF
}

execute=false
run_apt=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      execute=true
      ;;
    --no-apt)
      run_apt=false
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

NVHPC=${NVHPC:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.1}
BACKUP_ROOT=${BACKUP_ROOT:-/mnt/block/backups}
BACKUP=${BACKUP:-$BACKUP_ROOT/nvhpc-26.1-cuda13.1-$(date +%Y-%m-%d)}

run() {
  if [[ "$execute" == true ]]; then
    "$@"
  else
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
  fi
}

sudo_run() {
  run sudo "$@"
}

backup_move_dir() {
  local src=$1
  local dest_name=$2

  if [[ -d "$src" ]]; then
    sudo_run mv "$src" "$BACKUP/$dest_name"
  else
    echo "No tree found at $src"
  fi
}

require_path() {
  local path=$1
  local desc=$2
  if [[ ! -e "$path" ]]; then
    echo "Missing $desc: $path" >&2
    exit 1
  fi
}

echo "Mode: $([[ "$execute" == true ]] && echo execute || echo dry-run)"
echo "NVHPC: $NVHPC"
echo "Backup: $BACKUP"

require_path "$NVHPC/cuda/12.9/bin/nvcc" "HPC SDK CUDA 12.9 nvcc"

if [[ "$execute" == true ]]; then
  sudo -v
fi

echo
echo "Current CUDA state:"
df -h / /mnt/block 2>/dev/null || true
readlink -f /usr/local/cuda 2>/dev/null || true
command -v nvcc 2>/dev/null || true
nvcc --version 2>/dev/null | sed -n '1,5p' || true

echo
echo "Backing up and removing HPC SDK 13.1 payloads, if present..."
sudo_run mkdir -p "$BACKUP"

backup_move_dir "$NVHPC/cuda/13.1" "cuda-13.1"
backup_move_dir "$NVHPC/math_libs/13.1" "math_libs-13.1"
backup_move_dir "$NVHPC/comm_libs/13.1" "comm_libs-13.1"
backup_move_dir "$NVHPC/profilers/13.1" "profilers-13.1"
backup_move_dir "$NVHPC/REDIST/cuda/13.1" "REDIST-cuda-13.1"
backup_move_dir "$NVHPC/REDIST/math_libs/13.1" "REDIST-math_libs-13.1"
backup_move_dir "$NVHPC/REDIST/comm_libs/13.1" "REDIST-comm_libs-13.1"

echo
echo "Repointing HPC SDK generic links to 12.9..."
sudo_run ln -sfn 12.9/bin "$NVHPC/cuda/bin"
sudo_run ln -sfn 12.9/include "$NVHPC/cuda/include"
sudo_run ln -sfn 12.9/lib64 "$NVHPC/cuda/lib64"
sudo_run ln -sfn 12.9/nvvm "$NVHPC/cuda/nvvm"
sudo_run ln -sfn 12.9/include "$NVHPC/math_libs/include"
sudo_run ln -sfn 12.9/lib64 "$NVHPC/math_libs/lib64"
sudo_run ln -sfn 12.9/nccl "$NVHPC/comm_libs/nccl"
sudo_run ln -sfn 12.9/nvshmem "$NVHPC/comm_libs/nvshmem"

if [[ -e "$NVHPC/profilers/12.9/Nsight_Systems_2025.3" || "$execute" == false ]]; then
  sudo_run ln -sfn Nsight_Systems_2025.3 "$NVHPC/profilers/12.9/Nsight_Systems"
fi

if [[ "$run_apt" == true ]]; then
  echo
  echo "Purging package-managed CUDA 12.8 stack..."
  sudo_run apt-get -y purge 'cuda-*12-8' 'libcu*12-8' 'libnv*12-8' \
    gds-tools-12-8 nsight-compute-2025.1.0 nsight-systems-2024.6.2 \
    cuda-repo-ubuntu2404-12-8-local

  echo
  echo "Removing now-unused packages..."
  sudo_run apt-get -y autoremove --purge
else
  echo
  echo "Skipping apt purge/autoremove because --no-apt was passed."
fi

echo
echo "Cleaning stale CUDA 12.8 alternatives and setting /usr/local/cuda..."
sudo_run update-alternatives --remove cuda /usr/local/cuda-12.8
sudo_run update-alternatives --remove cuda-12 /usr/local/cuda-12.8
sudo_run ln -sfn "$NVHPC/cuda/12.9" /usr/local/cuda

echo
echo "Validation commands:"
cat <<EOF
df -h / /mnt/block
command -v nvcc
nvcc --version
/usr/local/cuda/bin/nvcc --version
$NVHPC/cuda/bin/nvcc --version
readlink -f /usr/local/cuda
find $NVHPC/cuda -maxdepth 1 \\( -type l -o -type d \\) -printf '%y %p -> %l\\n' | sort
find $NVHPC -maxdepth 3 \\( -type d -name '13.1' -o -lname '*13.1*' \\) -printf '%y %p -> %l\\n' | sort
dpkg-query -W -f='\${Package}\\t\${Status}\\n' 'cuda-*12-8' 'libcu*12-8' 'libnv*12-8' 2>/dev/null | sort
EOF

if [[ "$execute" == false ]]; then
  echo
  echo "Dry run complete. Re-run with --execute to apply changes."
fi
