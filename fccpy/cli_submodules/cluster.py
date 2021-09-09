#!/usr/bin/env python

"""Cluster a set of structures based on a similarity matrix."""

import argparse
import collections
import itertools
import os
from pathlib import Path
import sys
import time

from fccpy.clustering import dbscan
from fccpy.similarity import read_matrix
from .cli_utils import check_file, is_positive_float, list_of_paths, log

# Update here if necessary
CLUST_ALGOS = {"dbscan": dbscan}


def write_clusters(labels, filepath):
    """Write a summary file with cluster info.

    File contains one cluster per line, with clusters sorted by
    decreasing size (largest first). Each line lists all elements
    of that cluster, by the order (index) in which they appear in
    the original input file (indices of similarity matrix).
    """

    with filepath.open("wt") as handle:
        # Invert labels to clusters
        clusters = collections.defaultdict(list)
        for k, v in labels.items():
            clusters[v].append(k)

        # Sort clusters by size
        sorted_keys = sorted(clusters, key=lambda k: len(clusters.get(k)), reverse=True)

        for k in sorted_keys:
            elements = map(str, sorted(clusters[k]))
            print(",".join(elements), file=handle)


def write_clusters_as_pdb(labels, struct_list):
    """Write each cluster as a multi-model PDB file."""

    log("  writing cluster PDB files")

    clusters = collections.defaultdict(list)
    for k, v in labels.items():
        clusters[v].append(k)

    # Sort clusters by size
    sorted_keys = sorted(clusters, key=lambda k: len(clusters.get(k)), reverse=True)

    # Instead of parsing the files into Structure objects,
    # simply read ATOM/HETATM/TER lines and sandwich them
    # between MODEL/ENDMDL statements. Faster, less hassle.
    rectypes = ("ATOM  ", "HETATM", "TER  ")
    for clu in sorted_keys:
        struct_idx = sorted(clusters[clu])
        clu_structs = [struct_list[i] for i in struct_idx]

        clu_fp = f"cluster_{clu}.pdb"
        with open(clu_fp, "wt") as handle:
            lines = []
            for model_id, fn in enumerate(clu_structs):
                lines.append(f"MODEL {model_id:>4d}\n")
                with fn.open("rt") as pdbfile:
                    pdbdata = [ln for ln in pdbfile if ln.startswith(rectypes)]
                    lines.append(pdbdata)
                lines.append("ENDMDL\n")
            lines.append("END\n")

            print("".join(itertools.chain.from_iterable(lines)), file=handle)


def get_parser(cmd_args):
    """Create cmd arguments and options parser."""

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "matrix",
        type=check_file,
        help="Similarity matrix produced by mkcontacts, in h5 format.",
    )
    ap.add_argument(
        "-f",
        "--flist",
        type=list_of_paths,
        help=(
            "File listing structures to cluster, one per line. "
            "Order of structures in file must correspond to order "
            "or rows/columns in input matrix."
        ),
    )
    ap.add_argument(
        "-o",
        "--output_file",
        type=Path,
        default="clusters.out",
        help=(
            "Output file name to write clusters to. Clusters are written "
            "as a tsv file, with the centroid first. Indices represent the "
            "rows/columns of the similarity matrix. (default: %(default)s)"
        ),
    )
    ap.add_argument(
        "-wc",
        "--write-cluster-pdb",
        action="store_true",
        help=(
            "Write clusters as ensemble of models in PDB format. Clusters "
            "will be named 'cluster_i.pdb', where i is the cluster index."
        ),
    )
    ap.add_argument(
        "-m",
        "--method",
        choices=CLUST_ALGOS.keys(),
        type=CLUST_ALGOS.get,
        default="dbscan",
        help="Clustering method to use. (default: %(default)s)",
    )
    ap.add_argument(
        "--similarity-threshold",
        dest="eps",
        type=is_positive_float,
        default=0.6,
        help=(
            "Similarity threshold for two elements to belong in the "
            "same cluster. (default: %(default)s)"
        ),
    )
    ap.add_argument(
        "--min-cluster-size",
        dest="minsize",
        type=lambda x: int(is_positive_float(x)),
        default=4,
        help=(
            "Minimum number of elements that can form a cluster.  "
            "(default: %(default)s)"
        ),
    )

    # If we are calling from a submodule, the help might be wonky
    # because sys.argv[0] is the script name and cmd_args will be
    # sys.argv[2:]. So, check if sys.argv == cmd_args and if not
    # modify the parser .prog attribute
    if sys.argv != cmd_args:
        ap.prog = f"{os.path.basename(sys.argv[0])} {sys.argv[1]}"
    else:
        cmd_args = sys.argv[1:]  # remove script name

    args = ap.parse_args(cmd_args)

    # Do validation
    if args.write_cluster_pdb and not args.flist:
        raise ap.error("-wc/--write-clusters requires -flist")

    return args


def main(cmd_args):
    """Script entry point for setuptools."""

    args = get_parser(cmd_args)

    log("started")
    log(f"command: {' '.join(sys.argv)}")

    _start_time1 = time.time()
    idxs, sims = read_matrix(args.matrix)
    _end_time = time.time()
    read_time = _end_time - _start_time1
    log(f"  read similarity matrix in {read_time:4.2f} s")

    # Do clustering
    _start_time2 = time.time()
    labels = args.method(idxs, sims, eps=args.eps, minsize=args.minsize)
    _end_time = time.time()
    clust_time = _end_time - _start_time2

    n_clusters = len(set(labels.values()))
    n_clustered = len(labels)
    log(f"  clustering completed in {clust_time:6.3f} s")
    log(f"  clustered {n_clustered} elements in {n_clusters} clusters")

    # Write clusters.out
    write_clusters(labels, args.output_file)

    # Writing clusters as PDBs
    if args.write_cluster_pdb:
        write_clusters_as_pdb(labels, args.flist)

    _end_time = time.time()
    cumtime = _end_time - _start_time1

    # statistics
    log("finished!")
    log(f"  total processing time: {cumtime:4.2f} s")


if __name__ == "__main__":
    main(sys.argv)
