import numpy as np
from rdkit import Chem
from util import get_nos_coords
from atom_features import to_onehot
from molecule_features import mol_to_nums_adj


def feat_edges(mol, MAX_ATOM_N=None, MAX_EDGE_N=None):
    """
    Create features for edges, edge connectivity
    matrix, and edge/vert matrix

    Note: We really really should parameterize this somehow.
    """

    atom_n = mol.GetNumAtoms()
    atomicno, vert_adj = mol_to_nums_adj(mol)

    double_edge_n = np.sum(vert_adj > 0)
    assert double_edge_n % 2 == 0
    edge_n = double_edge_n // 2

    edge_adj = np.zeros((edge_n, edge_n))
    edge_vert_adj = np.zeros((edge_n, atom_n))

    edge_list = []
    for i in range(atom_n):
        for j in range(i + 1, atom_n):
            if vert_adj[i, j] > 0:
                edge_list.append((i, j))
                e_idx = len(edge_list) - 1
                edge_vert_adj[e_idx, i] = 1
                edge_vert_adj[e_idx, j] = 1
    # now which edges are connected
    edge_adj = edge_vert_adj @ edge_vert_adj.T - 2 * np.eye(edge_n)
    assert edge_adj.shape == (edge_n, edge_n)

    # now create edge features
    edge_features = []

    for edge_idx, (i, j) in enumerate(edge_list):
        f = []
        f += to_onehot(vert_adj[i, j], [1.0, 1.5, 2.0, 3.0])

        edge_features.append(f)
        # maybe do more stuff here? I don't know

    edge_features = np.array(edge_features)

    return {
        "edge_edge": np.expand_dims(edge_adj, 0),
        "edge_feat": edge_features,
        "edge_vert": np.expand_dims(edge_vert_adj, 0),
    }
