import numpy as np
from openTSNE import affinity, initialization, TSNEEmbedding
from pathlib import Path
import time

# import humanize
from datetime import timedelta

import scipy as sp


def run_tsne(embeddings, saving_suffix, saving_path=None, save_interm=False):
    assert saving_path is not None, "Saving path is missing"

    ### t-SNE
    start = time.time()

    ## affinities
    A = affinity.Uniform(
        embeddings,
        verbose=True,
        random_state=42,
        k_neighbors=10,
        n_jobs=-1,
    )
    # save
    if save_interm:
        sp.sparse.save_npz(saving_path / f"affinities{saving_suffix}", A.P)
    end = time.time()
    print("Runtime Affinities: ", str(timedelta(seconds=end - start)))

    ## initialization
    I = initialization.pca(embeddings, random_state=42)
    if save_interm:
        np.save(saving_path / f"initialization{saving_suffix}", I)
    end = time.time()
    print("Runtime Initialization: ", str(timedelta(seconds=end - start)))

    # ## load stuff -- ONLY USED FOR 2025
    # # load affinities P
    # affinities_P_bert_reparsed = sp.sparse.load_npz( saving_path / "affinities_2025.npz")
    # A = affinity.Affinities()
    # A.P = affinities_P_bert_reparsed
    # # load init
    # I=np.load( variables_path / "initialization_2025.npy")

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

    tsne = np.array(E)

    # save
    np.save(
        saving_path / f"tsne{saving_suffix}",
        tsne,
    )

    end = time.time()
    runtime_total = end - start
    print("Total runtime: ", str(timedelta(seconds=runtime_total)))


if __name__ == "__main__":
    # define paths
    variables_path = Path("../results/variables")
    berenslab_data_path = Path("/gpfs01/berens/data/data/pubmed_processed")

    # import embeddings
    saving_path = Path("embeddings/2025_baseline")
    embeddings = np.load(
        berenslab_data_path / saving_path / "embedding_sep_all.npy",
    )

    run_tsne(
        embeddings,
        "_2025",
        saving_path=variables_path,
        save_interm=True,
    )
