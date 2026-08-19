<!--
# Research project
This is a template for the repository of a reseach project containing research code. It provides the folder structure and the pre-commit hooks, and it assumes you are using uv as your packaging manager.

1. After creating a new repo off this one, initialize a uv environment by running `uv init --python 3.12`. This creates the environment and adds all uv-related files to the repo.
2. Run `make install_hooks` to set up the pre-commit hooks.
3. Run `make install_jupyter` to get jupyter working.
4. Run `make install_python_basics` to install some python basic files.
5. For the installable package, a name has to be choosen, the `src/package_code` folder renamed, and the line `name = "package_code"` in `mypyproject.toml` file and notebook imports edited accordingly. Afterwards, in the uv environment, run `uv pip install -e .`.

To activate your uv environment, use `source .venv/bin/activate`.
-->

# PubMed retina

Mapping the retina research literature within the PubMed landscape: embedding abstracts with PubMedBERT, visualizing them with t-SNE, and labeling papers by species/taxon, country, and author gender to explore how animal models are used in vision research.

## Pipeline

1. `scripts/01-process-baseline.ipynb`: parses and cleans the PubMed baseline (filters empty/non-English abstracts, applies length thresholds, fixes malformed dates) and labels papers with country, US state, and predicted first/last author gender.
2. `scripts/obtain_BERT_embeddings.py`: computes PubMedBERT embeddings (CLS, SEP, mean-pooled) for all abstracts.
3. `scripts/obtain_tsne_embeddings.py` and its variants (`obtain_retina_tsne_embeddings.py`, `obtain_non_human_tsne_embeddings.py`, `obtain_non_human_non_mouse_tsne_embeddings.py`): run t-SNE on the embeddings, progressively restricted to retina papers, then non-human, then non-human-non-mouse subsets.
4. `scripts/02-generate-2025-tsne-plots.ipynb`: generates the full PubMed landscape t-SNE plots, colored by class, year, and country.
5. `scripts/03-exploration-retina.ipynb`: explores the retina subset: masks retina papers, labels animal models (mice, zebrafish, etc.) and taxa, and plots t-SNE colored by species/taxon.

Core code lives in `src/pubmed_retina/`:
- `process_pubmed_utils.py`: XML parsing, cleaning, and country/state labeling.
- `embeddings_pubmed_utils.py`: PubMedBERT embedding generation.
- `labeling.py`: animal/species/taxon text labeling and color mapping.
- `plotting.py`: t-SNE plotting utilities.
- `exploration.py`: helpers for searching abstracts by keyword.

## Data

Scripts expect the processed PubMed baseline and embeddings at `/gpfs01/berens/data/data/pubmed_processed` (not included in this repo). Intermediate and final results are written to `results/variables/` and figures to `results/figures/`.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python 3.12).

```bash
uv sync
make install_hooks  # installs pre-commit hooks
```
