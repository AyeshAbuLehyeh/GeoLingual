#!/bin/bash
#SBATCH --job-name=geolingual-rerank
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${VIGOR_ROOT:?set VIGOR_ROOT}"
: "${DRESS_ROOT:?set DRESS_ROOT to the AuxGeo/DReSS checkout}"
python -m geolingual.vision_rerank_eval --mode "$1" --subset hard --out_prefix cache/vision_rerank_full591
