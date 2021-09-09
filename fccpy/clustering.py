"""Functions to cluster similarity matrices."""

from collections import defaultdict

import numpy as np


def dbscan(idxs, sims, **kwargs):
    """Cluster elements in distance matrix using the DBSCAN algorithm.

    This implementation assumes the similarity metric used is 1.0 when
    two elements are equal, and 0 when they are completely dissimilar.
    This is the _opposite_ of usual implementations (sklearn), where
    metrics are e.g. euclidean distance, and sim(i,j) = 0 means i = j.

    Parameters
    ----------
    idxs : np.ndarray
        array of indices i and j, corresponding to each pair of elements
        for which similarities were computed. Indices don't need to be
        sequential (e.g. 1,3,4,7,13...).
    sims : np.ndarray
        array of similarities between pairs of elements i and i of shape (n, 2),
        where n is the number of pairwise combinations of all N elements of the
        set where s(i, j) or s(j, i) is non-zero.

    eps : float, optional
        minimum distance between two elements i and j so that i and j
        are considered neighbors. Only neighboring elements will be
        placed in the same cluster. Default is 0.6.
    minsize : int, optional
        minimum number of neighboring elements necessary to form a
        cluster. Default is 2.

    Returns
    -------
    a dictionary mapping each element i in the original idxs to a cluster,
    identified by a number k >= 0. If an element is missing from the dict
    then it was not found to be part of any cluster.
    """

    eps = kwargs.get("eps", 0.6)
    minsize = kwargs.get("minsize", 2)

    minsize = minsize - 1  # the core point counts towards the size of its own cluster.

    # Filter pairs based on eps, so we can operate on a smaller dataset.
    # The resulting array will contain only pairs of elements that are
    # neighbors.
    sele = np.all(np.where(sims >= eps, sims, 0) != 0, axis=1)
    idxs_ = idxs[sele]

    # Build neighbor dicts for fast lookup
    nbdict = defaultdict(set)
    for i, j in idxs_:
        nbdict[i].add(j)
        nbdict[j].add(i)

    num_neighbors = {k: len(v) for k, v in nbdict.items()}

    # Determine clusterable elements: keep indices of the original data as labels.
    uniq_ele = {k for tup in idxs_ for k in tup}
    clustdict = {k: 0 for k in uniq_ele}  # label everyone as noise to start with

    # Make clusters
    # Iterate on elements in decreasing order of neighbor lists.
    # Only operate on clusterable elements
    clust_idx = 0
    for ele in sorted(nbdict, key=num_neighbors.get, reverse=True):
        if clustdict[ele] or num_neighbors[ele] < minsize:
            continue  # previously processed or noise

        clust_idx += 1  # Update cluster label

        clustdict[ele] = clust_idx  # core point
        neighbors = nbdict[ele]

        # Walk over element neighbors and expand with their neighbors
        while neighbors:
            nb_ele = neighbors.pop()
            if not clustdict[nb_ele]:  # not previously assigned

                clustdict[nb_ele] = clust_idx  # mark as part of cluster

                # Check neighbor's neighbors. If nb_ele is core point,
                # add neighbors to neighbors set and check later.
                if num_neighbors[nb_ele] >= minsize:
                    neighbors.update(nbdict[nb_ele])

    return {k: v for k, v in clustdict.items() if v}
