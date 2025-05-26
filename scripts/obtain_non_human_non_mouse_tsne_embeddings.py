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
mask_non_human = (~mask_retina_subject) & (~mask_retina_patient)

animal_labels = np.load(variables_path / "retina_animal_labels.npy", allow_pickle=True)
anmls = animal_labels[mask_non_human]
print(embeddings[mask_retina][mask_non_human][anmls != "mouse"].shape)

run_tsne(
    embeddings[mask_retina][mask_non_human][anmls != "mouse"],
    "_retina_non_human_non_mouse",
    saving_path=variables_path,
)
