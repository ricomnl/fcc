"""Functions to cluster similarity matrices."""

from collections import defaultdict
from itertools import combinations

import numpy as np

from fccpy.similarity import to_square_matrix


def dbscan(idxs, sims, **kwargs):
    """Cluster using the DBSCAN algorithm.

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
    then it was not found to be part of any cluster. Clusters are ordered
    from largest to smallest.
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


def disjoint_taylor_butina(idxs, sims, **kwargs):
    """Cluster using a disjoint asymmetric version of the Taylor-Butina algorithm.

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
    then it was not found to be part of any cluster. Clusters are ordered
    from largest to smallest.
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
    clustdict = {k: 0 for k in uniq_ele}  # label everyone as unclustered

    # Start clustering based on size of neighbors list: largest cluster first.
    clust_idx = 0
    for ele in sorted(nbdict, key=num_neighbors.get, reverse=True):
        if num_neighbors[ele] < minsize:
            continue  # previously processed or not enough neighbors

        clust_idx += 1  # Update cluster label

        # Add all neighbors to cluster and update neighbor lists
        # Once an element is clustered, it cannot belong to another cluster.
        cluster = {ele} | nbdict.pop(ele)
        for e in cluster:
            clustdict[e] = clust_idx

        for e in list(nbdict):  # bc assymetry
            nbdict[e] -= cluster
            num_neighbors[e] = len(nbdict[e])

    return {k: v for k, v in clustdict.items() if v}


def clique(idxs, sims, **kwargs):
    """Cluster using a maximal clique enumeration algorithm.

    This is _very_ computationally expensive and should not be expected to yield
    results for very large collections of structures. However, it has the unique
    advantage that all elements in each cluster are neighbors of each other, so
    clusters produced by this method are quite compact (if the value of eps is
    high enough of course).

    The algorithm finds all maximal cliques in the graph defined by the indices
    and similarity arrays. Then, cliques are sorted by the weighted sum of the
    values of their edges (ie. the average similarity between any two members
    of the clique) and filtered by picking the clique with the highest score
    of the set of cliques with common elements.

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
    then it was not found to be part of any cluster. Clusters are ordered
    from largest to smallest.
    """

    eps = kwargs.get("eps", 0.6)
    minsize = kwargs.get("minsize", 2)

    try:
        import networkx as nx
    except ImportError:
        raise ImportError("Package 'networkx' is required to use this method.")

    # Remove unwanted edges.
    sele = np.all(np.where(sims >= eps, sims, 0) != 0, axis=1)
    idxs_ = idxs[sele]
    sims_ = sims[sele]

    m = to_square_matrix(idxs_, sims_, symmetrize=True)
    g = nx.convert_matrix.from_numpy_array(m)

    # Remove isolated nodes
    isolates_list = list(nx.algorithms.isolates(g))
    g.remove_nodes_from(isolates_list)

    # Find all maximal cliques
    cliques = nx.find_cliques(g)

    # Prune cliques smaller than minsize
    cliques = [set(c) for c in cliques if len(c) > minsize]

    # Score cliques based on edge weights
    # NOTE: networkx is incredibly slow at fetching edge weights. We can get
    # the scores 4x faster like this, but only because the matrix is symmetrical.
    get_score = lambda c: m[tuple(zip(*combinations(c, 2)))].sum()  # noqa
    cliques.sort(key=get_score, reverse=True)

    # Make disjoint set of cliques
    disjoint_cliques = []
    for c in cliques:
        for dc in disjoint_cliques:
            if dc & c:
                break  # is subset of lower score
        else:
            disjoint_cliques.append(c)

    # Sort cliques by size so that the cluster label make some sense
    disjoint_cliques.sort(key=len, reverse=True)

    # Label elements by clique/cluster
    return {
        ele: idx for idx, clq in enumerate(disjoint_cliques, start=1) for ele in clq
    }
