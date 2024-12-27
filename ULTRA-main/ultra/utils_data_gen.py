import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pact.spasmspace import SpasmSpace
from pact.graphwrapper import GraphWrapper
from pact.ui import default_progressbar
from pact.naive_exec import naive_pandas_plan_exec
from pact.naive_exec import _undir_df_degree_thres

import os
import tempfile
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
import multiprocess as mp
import pandas as pd
import dill
import numpy as np

import math
import json

plt.rcParams["figure.figsize"] = (2, 2)

_MIN_TW_INJECT = 1
THREADS = 48
INPUT_FILE = '/content/drive/MyDrive/Graph Representation Learning/Mini Project/Testing/zinc/ZINC12k.json'

BASE_BASIS_PATH = '/content/drive/MyDrive/Graph Representation Learning/Mini Project/Testing/bases'
wanted_features = [
    'cycles/cycle3_basis.json',
    'cycles/cycle4_basis.json',
    # 'cycles/cycle5_basis.json',

    'paths/path3_basis.json',
    'paths/path4_basis.json',
    # 'paths/path5_basis.json',
]

spsps = []
Pbases = []
for wanted_feature in wanted_features:
    with open(os.path.join(BASE_BASIS_PATH, wanted_feature), 'rb') as f:
        basis_info = dill.loads(f.read())
    spsps.append(basis_info['SpasmSpace'])
    Pbases.append(basis_info['basis'])


# bit lazy but this needs to be the same as in the preprocessing
_MARK_VTX = 0


# convert a networkx graph and into a pandas DataFrame as expected by the naive plan executer
def nxGtoDf(nxG):
    edges = [{'s': a, 't': b} for a, b in nxG.edges()] + [{'s': b, 't': a} for a, b in nxG.edges()]
    host_df = pd.DataFrame(edges).drop_duplicates()
    return host_df


# returns a dictionary that maps v -> homs(F, host)[_MARK_VTX -> v]
# if a vertex of the host graph has no entry in the dictionary it means that there are 0 homs
def homcounts_per_vertex(F, host_df):
    state, empty = naive_pandas_plan_exec(F.plan, host_df, sliced_eval={})
    if not empty:
        finalcount = state['node$0']
        key = _MARK_VTX
        return finalcount.groupby(key)['count'].sum().to_dict()
    else:
        return {}


# Takes a networkx graph and the basis and computes the vertex_wise counts
# for all patterns in the basis into the graph.
# Returns a dictionary that maps each vertex of the host (nxG) to a dictionary that itself maps basis graph ids to
# the count of that basis into G
def counts_per_v(nxG, basis, spsp):
    ret = {v: {} for v in nxG.nodes}
    host_df = nxGtoDf(nxG)
    i = 0
    for fid in basis.keys():
        i += 1
        f = spsp[fid]
        if f.td.ghw < _MIN_TW_INJECT:  # skip the acyclic ones
            continue

        if (not f.is_directed and hasattr(f, 'clique') and
                f.clique is not None and f.clique > 2):
            small_host = _undir_df_degree_thres(host_df, f.clique - 1)
            vcounts_f = homcounts_per_vertex(f, small_host)
        else:
            vcounts_f = homcounts_per_vertex(f, host_df)

        for v, vc in vcounts_f.items():
            ret[v][fid] = vc
    return ret


# reads the edge_list input format from zinc and returns an networkx graph
def COOtonxG(edge_index):
    a = edge_index[0]
    b = edge_index[1]
    edges = list()
    for i in range(len(a)):
        e = list(sorted([a[i], b[i]]))
        if e not in edges:
            edges.append(e)
    return nx.from_edgelist(list(edges))


ordered_bases = [[gid for gid in Pbasis if spsp[gid].td.ghw >= _MIN_TW_INJECT] for spsp, Pbasis in zip(spsps, Pbases)]


def compute_counts(edge_list):
    host = COOtonxG(edge_list)

    all_homcounts = []
    for i, (spsp, Pbasis, ordered_basis) in enumerate(zip(spsps, Pbases, ordered_bases)):
        tmpcount = counts_per_v(host, Pbasis, spsp)
        homcounts = {}
        for v, vhoms in tmpcount.items():
            homcounts[v] = float(sum([vhoms.get(bgid, 0) * Pbasis[bgid] for bgid in ordered_basis]))
        all_homcounts.append(homcounts)
        print(f"Basis {i + 1}/{len(spsps)} completed.")

    return all_homcounts
