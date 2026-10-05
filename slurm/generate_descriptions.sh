#!/bin/bash
#SBATCH --job-name=geolingual-gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export VIGOR_ROOT="${VIGOR_ROOT:?set VIGOR_ROOT to your VIGOR dataset root}"
python scripts/generate_descriptions.py "Qwen/Qwen3-VL-30B-A3B-Instruct" "$1" "$2"
