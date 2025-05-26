import numpy as np
from pathlib import Path

from obtain_tsne_embeddings import run_tsne

# define paths
variables_path = Path("../results/variables")
berenslab_data_path = Path("/gpfs01/berens/data/data/pubmed_processed")

# import embeddings
saving_path = Path("embeddings/2025_baseline")
embeddings = np.load(
    berenslab_data_path / saving_path / "embedding_sep_all.npy",
)

mask_retina = np.load(variables_path / "mask_retina.npy")

run_tsne(
    embeddings[mask_retina],
    "_retina",
    saving_path=variables_path,
    save_interm = True
)
