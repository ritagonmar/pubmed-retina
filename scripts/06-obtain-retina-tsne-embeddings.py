import numpy as np
from openTSNE import affinity, initialization, TSNEEmbedding
from pathlib import Path
import time

# import humanize
from datetime import timedelta

import scipy as sp

# define paths
variables_path = Path("../results/variables")
berenslab_data_path = Path("/gpfs01/berens/data/data/pubmed_processed")

# import embeddings
saving_path = Path("embeddings/2025_baseline")
embedding_sep_all = np.load(
    berenslab_data_path / saving_path / "embedding_sep_all.npy",
)
mask_retina = np.load(variables_path / "mask_retina.npy")
embedding_retina = embedding_sep_all[mask_retina]

### t-SNE
start = time.time()

## affinities
A = affinity.Uniform(
    embedding_retina,
    verbose=True,
    random_state=42,
    k_neighbors=10,
    n_jobs=-1,
)
# save
sp.sparse.save_npz(variables_path / "affinities_retina", A.P)
end = time.time()
print("Runtime Affinities: ", str(timedelta(seconds=end - start)))

## initialization
I = initialization.pca(embedding_retina, random_state=42)
np.save(variables_path / "initialization_retina", I)
end = time.time()
print("Runtime Initialization: ", str(timedelta(seconds=end - start)))

# ## load stuff
# # load affinities P
# affinities_P_bert_reparsed = sp.sparse.load_npz( variables_path / "affinities_retina.npz")
# A = affinity.Affinities()
# A.P = affinities_P_bert_reparsed
# # load init
# I=np.load( variables_path / "initialization_retina.npy")

## optimization
E = TSNEEmbedding(I, A, n_jobs=-1, random_state=42, verbose=True)
# early exaggeration
E = E.optimize(n_iter=125, exaggeration=12, momentum=0.5, n_jobs=-1, verbose=True)

# exaggeration annealing
exs = np.linspace(12, 1, 125)
for i in range(125):
    if (i + 1) % 50 == 0:
        E = E.optimize(
            n_iter=1,
            exaggeration=exs[i],
            momentum=0.8,
            n_jobs=-1,
            verbose=True,
        )

    else:
        E = E.optimize(
            n_iter=1, exaggeration=exs[i], momentum=0.8, n_jobs=-1, verbose=True
        )

# final optimization without exaggeration
E = E.optimize(
    n_iter=2000,
    exaggeration=1,
    momentum=0.8,
    n_jobs=-1,
    verbose=True,
)

tsne_sep = np.array(E)

# save
np.save(
    variables_path / "tsne_sep_retina",
    tsne_sep,
)

end = time.time()
runtime_total = end - start
print("Total runtime: ", str(timedelta(seconds=runtime_total)))
