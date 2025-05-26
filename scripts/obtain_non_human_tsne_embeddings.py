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
mask_retina_subject = np.load(variables_path / "mask_retina_subject.npy")
mask_retina_patient = np.load(variables_path / "mask_retina_patient.npy")
print(np.sum((~mask_retina_subject) & (~mask_retina_patient)))

run_tsne(
    embeddings[mask_retina][(~mask_retina_subject) & (~mask_retina_patient)],
    "_retina_non_human",
    saving_path=variables_path,
)
