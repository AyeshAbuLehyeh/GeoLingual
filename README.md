# What Words Keep of a Place: Zero-Shot Language Reasoning for Cross-View Geo-Localization

Code, prompts, data splits, and figures for the paper [*What Words Keep of a Place: Zero-Shot Language Reasoning for Cross-View Geo-Localization*](https://arxiv.org/abs/2610.07269), accepted to the NeurIPS 2026 Workshop on Physical World AI.

The pipeline uses a general vision-language model, with no training, to turn ground panoramas and satellite tiles into structured text. We then test that text for cross-view matching: as an embedding retrieval target, as an LLM judge over geo-nearest hard negatives, and as a reranker of a vision retriever's top-10.

## Requirements

- Python 3.12, one NVIDIA GPU with enough memory for the 30B bf16 model (we used NVIDIA H200), CUDA 12.x.
- The VIGOR dataset (Zhu et al., CVPR 2021), obtained from the official repository at https://github.com/Jeff-Zilence/VIGOR under its own license. Set `VIGOR_ROOT` to its root directory.
- For the vision-retrieval step: the AuxGeo checkpoint `vigor_same/convnext_base/1216225004/weights_e40_80.4258.pth` from the AuxGeo/DReSS code repository (https://github.com/summerpanking/dress). Set `DRESS_ROOT` to that root directory.
- The release archive `descriptions.tar.gz` (attached to the GitHub release). It contains the 19,536 VLM-generated descriptions, so you can skip the GPU generation step.

## Setup

```bash
git clone https://github.com/AyeshAbuLehyeh/GeoLingual.git
cd GeoLingual
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CUDA_HOME=/path/to/cuda        # CUDA toolkit root with nvcc, needed to build flash-attn
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
python -m nltk.downloader punkt_tab wordnet averaged_perceptron_tagger_eng omw-1.4
```

Set the paths, and unpack the descriptions into the repository root:

```bash
export VIGOR_ROOT=/path/to/VIGOR
tar -xzf /path/to/descriptions.tar.gz
```

Run every command from the repository root.

## Running the pipeline

**1. Build the valid-pair dataset** (CPU, about a minute):

```bash
python -m geolingual.dataset_utils
```

This keeps the 9,826 ground-satellite pairs whose descriptions validate.

**2. Lexical analysis and figures** (CPU):

```bash
python -m geolingual.lexical_analysis
python scripts/make_bow_figures.py
```

**3. Embedding retrieval and cosine analysis** (GPU recommended):

```bash
python -m geolingual.similarity_and_retrieval
python -m geolingual.refresh_kde_heatmap
```

**4. Structured verification, LLM judge** (GPU):

```bash
python -m geolingual.spatial_verification --mode run --n_per_city 150 --k_hard 9 --batch_size 16 --log_every 10 --out_prefix cache/spatial_verification_full
python -m geolingual.pool_embedding_baseline --n_per_city 150 --k_hard 9 --out cache/pool_embedding_baseline_full_metrics.json
python -m geolingual.spatial_verification --mode ablation --n_per_city 30 --batch_size 16 --out_prefix cache/spatial_verification
python -m geolingual.hallucination_check --n_per_city 20 --modality ground --out cache/hallucination_check_ground.json
```

**5. Vision retrieval and reranking** (GPU):

```bash
export DRESS_ROOT=/path/to/DReSS
python -m geolingual.vision_retrieval_eval --top_k 10 --out_prefix cache/vision_retrieval
sbatch slurm/vision_retrieval.sh text
sbatch slurm/vision_retrieval.sh image
sbatch slurm/vision_retrieval.sh both
```

Without Slurm, run `python -m geolingual.vision_rerank_eval --mode text --subset hard --out_prefix cache/vision_rerank_full591` and likewise for `image` and `both`.

**6. Remaining figures:**

```bash
python scripts/make_rerank_figures.py
python scripts/make_verifier_examples_figure.py
python scripts/make_qualitative_figure.py
python scripts/make_pipeline_diagram.py
```

Figures are written to `figures/`. The rerank figures need step 5, and the verifier-example and qualitative figures need `VIGOR_ROOT` set.

To regenerate the descriptions from scratch instead of using the archive, run `sbatch slurm/generate_descriptions.sh Chicago satellite` and the same for each city and modality.

## Repository layout

- `geolingual/`: library code. `config.py` holds the paths. `vlm.py` wraps the VLM, `xml_utils.py` repairs the generated XML, `dataset_utils.py` builds the pairs, and the remaining modules implement the analyses and experiments.
- `scripts/`: command-line entry points for description generation, dataset creation, and figures.
- `prompts/`: the exact system prompts used to generate the descriptions.
- `data/`: per-city split CSVs.
- `figures/`: the figures in the paper.
- `slurm/`: batch templates for the GPU steps.

## Citation

```bibtex
@article{geolingual,
  author  = {Lehyeh, Ayesh Abu and Hwasung Jung, Jay and Wshah, Safwan},
  title   = {What Words Keep of a Place: Zero-Shot Language Reasoning for Cross-View Geo-Localization},
  journal = {arXiv preprint arXiv:2610.07269},
  year    = {2026}
}
```

