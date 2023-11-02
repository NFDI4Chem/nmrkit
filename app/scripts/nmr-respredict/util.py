import contextlib
import os
import numpy as np
import tempfile

from sklearn.cluster import AffinityPropagation

from rdkit import Chem
from rdkit.Chem import AllChem
import pickle

# import pubchempy as pcp
import rdkit
import math
import sklearn.metrics.pairwise
import scipy.optimize
import pandas as pd
import re
import itertools
import time
import numba
import torch
import io
import zlib

import collections
import scipy.optimize
import scipy.special
import scipy.spatial.distance
import nets
from tqdm import tqdm

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


CHEMICAL_SUFFIXES = [
    "ane",
    "onl",
    "orm",
    "ene",
    "ide",
    "hyde",
    "ile",
    "nol",
    "one",
    "ate",
    "yne",
    "ran",
    "her",
    "ral",
    "ole",
    "ine",
]


@contextlib.contextmanager
def cd(path):
    old_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_path)


def conformers_best_rms(mol):
    num_conformers = mol.GetNumConformers()
    best_rms = np.zeros((num_conformers, num_conformers))
    for i in range(num_conformers):
        for j in range(num_conformers):
            best_rms[i, j] = AllChem.GetBestRMS(mol, mol, prbId=i, refId=j)
    return best_rms


def cluster_conformers(mol):
    """
    return the conformer IDs that represent cluster centers
    using affinity propagation

    return conformer positions from largest to smallest cluster

    """
    best_rms = conformers_best_rms(mol)

    af = AffinityPropagation(affinity="precomputed").fit(best_rms)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    cluster_sizes = np.zeros(n_clusters_)
    for i in range(n_clusters_):
        cluster_sizes[i] = np.sum(labels == i)
    sorted_indices = cluster_centers_indices[np.argsort(cluster_sizes)[::-1]]

    return sorted_indices, labels, best_rms


def GetCalcShiftsLabels(
    numDS, BShieldings, labels, omits, TMS_SC_C13=191.69255, TMS_SC_H1=31.7518583
):
    """
    originally from pydp4
    """

    Clabels = []
    Hlabels = []
    Cvalues = []
    Hvalues = []

    for DS in range(numDS):
        Cvalues.append([])
        Hvalues.append([])

        # loops through particular output and collects shielding constants
        # and calculates shifts relative to TMS
        for atom in range(len(BShieldings[DS])):
            shift = 0
            atom_label = labels[atom]
            atom_symbol = re.match("(\D+)\d+", atom_label).groups()[0]

            if atom_symbol == "C" and not labels[atom] in omits:
                # only read labels once, i.e. the first diastereomer
                if DS == 0:
                    Clabels.append(labels[atom])
                shift = (TMS_SC_C13 - BShieldings[DS][atom]) / (
                    1 - (TMS_SC_C13 / 10**6)
                )
                Cvalues[DS].append(shift)

            if atom_symbol == "H" and not labels[atom] in omits:
                # only read labels once, i.e. the first diastereomer
                if DS == 0:
                    Hlabels.append(labels[atom])
                shift = (TMS_SC_H1 - BShieldings[DS][atom]) / (
                    1 - (TMS_SC_H1 / 10**6)
                )
                Hvalues[DS].append(shift)

    return Cvalues, Hvalues, Clabels, Hlabels


def mol_to_sdfstr(mol):
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as fid:
        writer = Chem.SDWriter(fid)
        writer.write(mol)
        writer.close()
        fid.flush()
        fid.seek(0)
        return fid.read()


def download_cas_to_mol(molecule_cas, sanitize=True):
    """
    Download molecule via cas, add hydrogens, clean up
    """
    sdf_str = cirpy.resolve(molecule_cas, "sdf3000", get_3d=True)
    mol = sdbs_util.sdfstr_to_mol(sdf_str)
    mol = Chem.AddHs(mol)

    # this is not a good place to do this
    # # FOR INSANE REASONS I DONT UNDERSTAND we get
    # #  INITROT  --  Rotation about     1     4 occurs more than once in Z-matrix
    # # and supposeldy reordering helps

    # np.random.seed(0)
    # mol = Chem.RenumberAtoms(mol, np.random.permutation(mol.GetNumAtoms()).astype(int).tolist())

    # mol.SetProp("_Name", molecule_cas)
    # rough geometry
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    return mol


def check_prop_failure(is_success, infile, outfile):
    if not is_success:
        pickle.dump(
            {"success": False, "previous_success": False, "infile": infile},
            open(outfile, "wb"),
        )
    return not is_success


def pubchem_cid_to_sdf(cid, cleanup_3d=True):
    """
    Go from pubmed CID to
    """
    with tempfile.TemporaryDirectory() as tempdir:
        fname = f"{tempdir}/test.sdf"
        pcp.download("SDF", fname, cid, "cid", overwrite=True)
        suppl = Chem.SDMolSupplier(fname, sanitize=True)
        mol = suppl[0]
        mol = Chem.AddHs(mol)
        if cleanup_3d:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        return mol


def render_2d(mol):
    mol = Chem.Mol(mol)

    AllChem.Compute2DCoords(mol)
    return mol


def array_to_conf(mat):
    """
    Take in a (N, 3) matrix of 3d positions and create
    a conformer for those positions.

    ASSUMES atom_i = row i so make sure the
    atoms in the molecule are the right order!

    """
    N = mat.shape[0]
    conf = Chem.Conformer(N)

    for ri in range(N):
        p = rdkit.Geometry.rdGeometry.Point3D(*mat[ri])
        conf.SetAtomPosition(ri, p)
    return conf


def add_empty_conf(mol):
    N = mol.GetNumAtoms()
    pos = np.zeros((N, 3))

    conf = array_to_conf(pos)
    mol.AddConformer(conf)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    From https://stackoverflow.com/a/6802723/1073963
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotate_mat(theta, phi):
    """
    generate a rotation matrix with theta around x-axis
    and phi around y
    """
    return np.dot(
        rotation_matrix([1, 0, 0], theta),
        rotation_matrix([0, 1, 0], phi),
    )


def mismatch_dist_mat(num_a, num_b, mismatch_val=100):
    """
    Distance handicap matrix. Basically when matching elements
    from num_a to num_b, if they disagree (and thus shoudln't be
    matched) add mismatch_value
    """

    m = np.zeros((len(num_a), len(num_b)))
    for i, a_val in enumerate(num_a):
        for j, b_val in enumerate(num_b):
            if a_val != b_val:
                m[i, j] = mismatch_val
    return m


def create_rot_mats(ANGLE_GRID_N=48):
    """
    Create a set of rotation matrices through angle gridding
    """

    theta_points = np.linspace(0, np.pi * 2, ANGLE_GRID_N, endpoint=False)
    rotate_points = np.array(
        [a.flatten() for a in np.meshgrid(theta_points, theta_points)]
    ).T
    rot_mats = np.array([rotate_mat(*a) for a in rotate_points])

    return rot_mats


def weight_heavyatom_mat(num_a, num_b, heavy_weight=10.0):
    """ """

    m = np.zeros((len(num_a), len(num_b)))
    for i, a_val in enumerate(num_a):
        for j, b_val in enumerate(num_b):
            if a_val > 1 and b_val > 1:
                m[i, j] = heavy_weight
    return m


def compute_rots_and_assignments(
    points_1, points_2, dist_mat_mod=None, ANGLE_GRID_N=48
):
    """
    Compute the distance between points for all possible
    gridded rotations.

    """

    rot_mats = create_rot_mats(ANGLE_GRID_N)

    all_test_points = np.dot(rot_mats, points_2.T)

    total_dists = []
    assignments = []
    for test_points in all_test_points:
        dist_mat = sklearn.metrics.pairwise.euclidean_distances(points_1, test_points.T)
        if dist_mat_mod is not None:
            dist_mat += dist_mat_mod
        cost_assignment = scipy.optimize.linear_sum_assignment(dist_mat)
        assignments.append(cost_assignment)

        match_distances = dist_mat[np.array(list(zip(*cost_assignment)))]
        total_dist = np.sum(match_distances)

        total_dists.append(total_dist)
    assert assignments[0][0].shape[0] == points_1.shape[0]
    return total_dists, assignments


def find_best_ordering(
    sdf_positions, sdf_nums, table_positions, table_nums, mismatch_val=100
):
    """
    Find the ordering of table_positions that minimizes
    the distance between it and sdf_positions at some rotation
    """
    mod_dist_mat = mismatch_dist_mat(sdf_nums, table_nums, mismatch_val=mismatch_val)
    mod_dist_mat += weight_heavyatom_mat(sdf_nums, table_nums, 10.0)
    # print(mod_dist_mat)
    total_dists, assignments = compute_rots_and_assignments(
        sdf_positions, table_positions, dist_mat_mod=mod_dist_mat, ANGLE_GRID_N=48
    )

    best_assign_i = np.argmin(total_dists)
    # pylab.axvline(best_assign_i, c='r')
    best_assignment = assignments[best_assign_i]
    return best_assignment[1], total_dists[best_assign_i]


def explode_df(df, lst_cols, fill_value=""):
    """
    Take a data frame with a column that's a list of entries and return
    one with a row for each element in the list

    From https://stackoverflow.com/a/40449726/1073963

    """
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return (
            pd.DataFrame(
                {
                    col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
                    for col in idx_cols
                }
            )
            .assign(**{col: np.concatenate(df[col].values) for col in lst_cols})
            .loc[:, df.columns]
        )
    else:
        # at least one list in cells is empty
        return (
            pd.DataFrame(
                {
                    col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
                    for col in idx_cols
                }
            )
            .assign(**{col: np.concatenate(df[col].values) for col in lst_cols})
            .append(df.loc[lens == 0, idx_cols])
            .fillna(fill_value)
            .loc[:, df.columns]
        )


def generate_canonical_fold_sets(BLOCK_N, HOLDOUT_N):
    """
    This generates a canonical ordering of N choose K where:
    1. the returned subset elements are always sorted in ascending order
    2. the union of the first few is the full set

    This is useful for creating canonical cross-validation/holdout sets
    where you want to compare across different experimental setups
    but you want to make sure you see all the data in the first N
    """

    if BLOCK_N % HOLDOUT_N == 0:
        COMPLETE_FOLD_N = BLOCK_N // HOLDOUT_N
        # evenly divides, we can do sane thing
        init_sets = []
        for i in range(HOLDOUT_N):
            s = np.array(np.split((np.arange(BLOCK_N) + i) % BLOCK_N, COMPLETE_FOLD_N))
            init_sets.append([sorted(i) for i in s])
        init_folds = np.concatenate(init_sets)

        all_folds = set(
            [
                tuple(sorted(a))
                for a in itertools.combinations(np.arange(BLOCK_N), HOLDOUT_N)
            ]
        )

        # construct set of init
        init_folds_set = set([tuple(a) for a in init_folds])
        assert len(init_folds_set) == len(init_folds)
        assert init_folds_set.issubset(all_folds)
        non_init_folds = all_folds - init_folds_set

        all_folds_array = np.zeros((len(all_folds), HOLDOUT_N), dtype=np.int)
        all_folds_array[: len(init_folds)] = init_folds
        all_folds_array[len(init_folds) :] = list(non_init_folds)

        return all_folds_array
    else:
        raise NotImplementedError()


def dict_product(d):
    dicts = {}
    for k, v in d.items():
        if not isinstance(v, (list, tuple, np.ndarray)):
            v = [v]
        dicts[k] = v

    return list((dict(zip(dicts, x)) for x in itertools.product(*dicts.values())))


class SKLearnAdaptor(object):
    def __init__(
        self, model_class, feature_col, pred_col, model_args, save_debug=False
    ):
        """
        feature_col is either :
        1. a single string for a feature column which will be flattened and float32'd
        2. a list of [(df_field_name, out_field_name, dtype)]
        """

        self.model_class = model_class
        self.model_args = model_args

        self.m = self.create_model(model_class, model_args)
        self.feature_col = feature_col
        self.pred_col = pred_col
        self.save_debug = save_debug

    def create_model(self, model_class, model_args):
        return model_class(**model_args)

    def get_X(self, df):
        if isinstance(self.feature_col, str):
            # do the default thing
            return np.vstack(
                df[self.feature_col].apply(lambda x: x.flatten()).values
            ).astype(np.float32)
        else:
            # X is a dict of arrays
            return {
                out_field: np.stack(df[in_field].values).astype(dtype)
                for in_field, out_field, dtype in self.feature_col
            }

    def fit(self, df, partial=False):
        X = self.get_X(df)
        y = np.array(df[self.pred_col]).astype(np.float32).reshape(-1, 1)
        if isinstance(X, dict):
            for k, v in X.items():
                assert len(v) == len(y)
        else:
            assert len(X) == len(y)
        if self.save_debug:
            pickle.dump(
                {"X": X, "y": y},
                open("/tmp/SKLearnAdaptor.fit.{}.pickle".format(t), "wb"),
                -1,
            )
        if partial:
            self.m.partial_fit(X, y)
        else:
            self.m.fit(X, y)

    def predict(self, df):
        X_test = self.get_X(df)

        pred_vect = pd.DataFrame(
            {"est": self.m.predict(X_test).flatten()}, index=df.index
        )
        if self.save_debug:
            pickle.dump(
                {"X_test": X_test, "pred_vect": pred_vect},
                open("/tmp/SKLearnAdaptor.predict.{}.pickle".format(t), "wb"),
                -1,
            )

        return pred_vect


@numba.jit(nopython=True)
def create_masks(BATCH_N, row_types, out_types):
    MAT_N = row_types.shape[1]
    OUT_N = len(out_types)

    M = np.zeros((BATCH_N, MAT_N, MAT_N, OUT_N), dtype=np.float32)
    for bi in range(BATCH_N):
        for i in range(MAT_N):
            for j in range(MAT_N):
                for oi in range(OUT_N):
                    if out_types[oi] == row_types[bi, j]:
                        M[bi, i, j, oi] = 1
    return M


def numpy(x):
    """
    pytorch convenience method just to get a damn
    numpy array back from a tensor or variable
    wherever the hell it lives
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.array(x)

    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()
    raise NotImplementedError(str(type(x)))


def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)


def split_df(dfm, chunk_size):
    """
    For splitting a df in to chunks of approximate size chunk_size
    """
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.array_split(dfm, indices)


def create_col_constraint(max_col_sum):
    """
    N = len(max_col_sum)
    for a NxN matrix x create a matrix A (N x NN*2)
    such that A(x.flatten)=b constrains the columns of x to equal max_col_sum

    return A, b
    """
    N = len(max_col_sum)
    A = np.zeros((N, N**2))
    b = max_col_sum
    Aidx = np.arange(N * N).reshape(N, N)
    for row_i, max_i in enumerate(max_col_sum):
        sub_i = Aidx[:, row_i]
        A[row_i, sub_i] = 1
    return A, b


def create_row_constraint(max_row_sum):
    """
    N = len(max_row_sum)
    for a NxN matrix x create a matrix A (N x NN*2)
    such that A(x.flatten)=b constrains the row of x to equal max_row_sum

    return A, b
    """
    N = len(max_row_sum)
    A = np.zeros((N, N**2))
    b = max_row_sum
    Aidx = np.arange(N * N).reshape(N, N)
    for row_i, max_i in enumerate(max_row_sum):
        sub_i = Aidx[row_i, :]
        A[row_i, sub_i] = 1
    return A, b


def row_col_sums(max_vals):
    Ac, bc = create_row_constraint(max_vals)

    Ar, br = create_col_constraint(max_vals)
    Aall = np.vstack([Ac, Ar])
    ball = np.concatenate([bc, br])
    return Aall, ball


def adj_to_mol(adj_mat, atom_types):
    assert adj_mat.shape == (len(atom_types), len(atom_types))

    mol = Chem.RWMol()
    for atom_i, a in enumerate(atom_types):
        if a > 0:
            atom = Chem.Atom(int(a))
            idx = mol.AddAtom(atom)
    for a_i in range(len(atom_types)):
        for a_j in range(a_i + 1, len(atom_types)):
            bond_order = adj_mat[a_i, a_j]
            bond_order_int = np.round(bond_order)
            if bond_order_int == 0:
                pass
            elif bond_order_int == 1:
                bond = Chem.rdchem.BondType.SINGLE
            elif bond_order_int == 2:
                bond = Chem.rdchem.BondType.DOUBLE
            else:
                raise ValueError()

            if bond_order_int > 0:
                mol.AddBond(a_i, a_j, order=bond)
    return mol


def get_bond_order(m, i, j):
    """
    return numerical bond order
    """
    b = m.GetBondBetweenAtoms(int(i), int(j))
    if b is None:
        return 0
    c = b.GetBondTypeAsDouble()
    return c


def get_bond_order_mat(m):
    """
    for a given molecule get the adj matrix with the right bond order
    """

    ATOM_N = m.GetNumAtoms()
    A = np.zeros((ATOM_N, ATOM_N))
    for i in range(ATOM_N):
        for j in range(i + 1, ATOM_N):
            b = get_bond_order(m, i, j)
            A[i, j] = b
            A[j, i] = b
    return A


def get_bond_list(m):
    """
    return a multiplicty-respecting list of bonds
    """
    ATOM_N = m.GetNumAtoms()
    bond_list = []
    for i in range(ATOM_N):
        for j in range(i + 1, ATOM_N):
            b = get_bond_order(m, i, j)
            for bi in range(int(b)):
                bond_list.append((i, j))
    return bond_list


def clear_bonds(mrw):
    """
    in-place clear bonds
    """
    ATOM_N = mrw.GetNumAtoms()
    for i in range(ATOM_N):
        for j in range(ATOM_N):
            if mrw.GetBondBetweenAtoms(i, j) is not None:
                mrw.RemoveBond(i, j)
    return mrw


def set_bonds_from_list(m, bond_list):
    """
    for molecule M, set its bonds from the list
    """
    mrw = Chem.RWMol(m)
    clear_bonds(mrw)
    for i, j in bond_list:
        b_order = get_bond_order(mrw, i, j)
        set_bond_order(mrw, i, j, b_order + 1)
    return Chem.Mol(mrw)


def edge_array(G):
    return np.array(list(G.edges()))


def canonicalize_edge_array(X):
    """
    Sort an edge array first by making sure each
    edge is (a, b) with a <= b
    and then lexographically
    """
    Y = np.sort(X)
    return Y[np.lexsort(np.rot90(Y))]


def set_bond_order(m, i, j, order):
    i = int(i)
    j = int(j)
    # remove existing bond
    if m.GetBondBetweenAtoms(i, j) is not None:
        m.RemoveBond(i, j)

    order = int(np.floor(order))
    if order == 0:
        return
    if order == 1:
        rd_order = rdkit.Chem.BondType.SINGLE
    elif order == 2:
        rd_order = rdkit.Chem.BondType.DOUBLE
    elif order == 3:
        rd_order = rdkit.Chem.BondType.TRIPLE
    else:
        raise ValueError(f"unkown order {order}")

    m.AddBond(i, j, order=rd_order)


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def conf_not_null(mol, conf_i):
    _, coords = get_nos_coords(mol, conf_i)

    if np.sum(coords**2) < 0.01:
        return False
    return True


def get_nos_coords(mol, conf_i):
    conformer = mol.GetConformer(conf_i.item())
    coord_objs = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    coords = np.array([(c.x, c.y, c.z) for c in coord_objs])
    atomic_nos = np.array([a.GetAtomicNum() for a in mol.GetAtoms()]).astype(int)
    return atomic_nos, coords


def get_nos(mol):
    return np.array([a.GetAtomicNum() for a in mol.GetAtoms()]).astype(int)


def move(tensor, cuda=False):
    from torch import nn

    if cuda:
        if isinstance(tensor, nn.Module):
            return tensor.cuda()
        else:
            return tensor.cuda(non_blocking=True)
    else:
        return tensor.cpu()


def mol_df_to_neighbor_atoms(mol_df):
    """
    Take in a molecule df and return a dataframe mapping
    (mol_id, atom_idx)
    """

    neighbors = []
    for mol_id, row in tqdm(mol_df.iterrows(), total=len(mol_df)):
        m = row.rdmol
        for atom_idx in range(m.GetNumAtoms()):
            a = m.GetAtomWithIdx(atom_idx)
            nas = a.GetNeighbors()
            r = {"mol_id": mol_id, "atom_idx": atom_idx}
            for na in nas:
                s = na.GetSymbol()
                if s in r:
                    r[s] += 1
                else:
                    r[s] = 1
            r["num_atoms"] = m.GetNumAtoms()
            neighbors.append(r)
    neighbors_df = pd.DataFrame(neighbors).fillna(0).set_index(["mol_id", "atom_idx"])
    return neighbors_df


def np_to_bytes(arr):
    fid = io.BytesIO()
    np.save(fid, arr)
    return fid.getvalue()


def recursive_update(d, u):
    ### Dict recursive update
    ### https://stackoverflow.com/a/3233356/1073963
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def morgan4_crc32(m):
    mf = Chem.rdMolDescriptors.GetHashedMorganFingerprint(m, 4)
    crc = zlib.crc32(mf.ToBinary())
    return crc


def get_atom_counts(rdmol):
    counts = {}
    for a in rdmol.GetAtoms():
        s = a.GetSymbol()
        if s not in counts:
            counts[s] = 0
        counts[s] += 1
    return counts


def get_ring_size_counts(rdmol):
    counts = {}
    ssr = Chem.rdmolops.GetSymmSSSR(rdmol)
    for ring_members in ssr:
        rs = len(ring_members)
        rs_str = rs

        if rs_str not in counts:
            counts[rs_str] = 0
        counts[rs_str] += 1
    return counts


def filter_mols(mol_dicts, filter_params, other_attributes=[]):
    """
    Filter molecules per criteria
    """

    skip_reason = []
    ## now run the query
    output_mols = []
    for row in tqdm(mol_dicts):
        mol_id = row["id"]
        mol = Chem.Mol(row["mol"])
        atom_counts = get_atom_counts(mol)
        if not set(atom_counts.keys()).issubset(filter_params["elements"]):
            skip_reason.append({"mol_id": mol_id, "reason": "elements"})
            continue

        if mol.GetNumAtoms() > filter_params["max_atom_n"]:
            skip_reason.append({"mol_id": mol_id, "reason": "max_atom_n"})
            continue

        if mol.GetNumHeavyAtoms() > filter_params["max_heavy_atom_n"]:
            skip_reason.append({"mol_id": mol_id, "reason": "max_heavy_atom_n"})
            continue

        ring_size_counts = get_ring_size_counts(mol)
        if len(ring_size_counts) > 0:
            if np.max(list(ring_size_counts.keys())) > filter_params["max_ring_size"]:
                skip_reason.append({"mol_id": mol_id, "reason": "max_ring_size"})
                continue
            if np.min(list(ring_size_counts.keys())) < filter_params["min_ring_size"]:
                skip_reason.append({"mol_id": mol_id, "reason": "min_ring_size"})
                continue
        skip_mol = False
        for a in mol.GetAtoms():
            if (
                a.GetFormalCharge() != 0
                and not filter_params["allow_atom_formal_charge"]
            ):
                skip_mol = True
                skip_reason.append({"mol_id": mol_id, "reason": "atom_formal_charge"})

                break

            if (
                a.GetHybridization() == 0
                and not filter_params["allow_unknown_hybridization"]
            ):
                skip_mol = True
                skip_reason.append(
                    {"mol_id": mol_id, "reason": "unknown_hybridization"}
                )

                break
            if a.GetNumRadicalElectrons() > 0 and not filter_params["allow_radicals"]:
                skip_mol = True
                skip_reason.append({"mol_id": mol_id, "reason": "radical_electrons"})

                break
        if skip_mol:
            continue

        if (
            Chem.rdmolops.GetFormalCharge(mol) != 0
            and not filter_params["allow_mol_formal_charge"]
        ):
            skip_reason.append({"mol_id": mol_id, "reason": "mol_formal_charge"})

            continue

        skip_reason.append({"mol_id": mol_id, "reason": None})

        out_row = {
            "molecule_id": mol_id,
            # 'mol': row['mol'],
            # 'source' : row['source'],  # to ease downstream debugging
            # 'source_id' : row['source_id'],
            "simple_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False),
        }
        for f in other_attributes:
            out_row[f] = row[f]

        output_mols.append(out_row)
    output_mol_df = pd.DataFrame(output_mols)
    skip_reason_df = pd.DataFrame(skip_reason)
    return output_mol_df, skip_reason_df


PERM_MISSING_VALUE = 1000


def vect_pred_min_assign(pred, y, mask, Y_MISSING_VAL=PERM_MISSING_VALUE):
    new_y = np.zeros_like(y)
    new_mask = np.zeros_like(mask)

    true_vals = y  # [mask>0]
    true_vals = true_vals[true_vals < Y_MISSING_VAL]

    dist = scipy.spatial.distance.cdist(pred.reshape(-1, 1), true_vals.reshape(-1, 1))
    dist[mask == 0] = 1e5
    ls_assign = scipy.optimize.linear_sum_assignment(dist)
    mask_out = np.zeros_like(mask)
    y_out = np.zeros_like(y)

    for i, o in zip(*ls_assign):
        mask_out[i] = 1
        y_out[i] = true_vals[o]

    return y_out, mask_out


def min_assign(pred, y, mask, Y_MISSING_VAL=PERM_MISSING_VALUE):
    """
    Find the minimum assignment of y to pred

    pred, y, and mask are (BATCH, N, 1) but Y is unordered and
    has missing entries set to Y_MISSING_VAL

    returns a new y and pred which can be used
    """
    BATCH_N, _ = pred.shape
    if pred.ndim > 2:
        pred = pred.squeeze(-1)
        y = y.squeeze(-1)
        mask = mask.squeeze(-1)

    y_np = y.cpu().detach().numpy()
    mask_np = mask.numpy()
    # print("total mask=", np.sum(mask_np))
    pred_np = pred.numpy()

    out_y_np = np.zeros_like(y_np)
    out_mask_np = np.zeros_like(pred_np)
    for i in range(BATCH_N):
        # print("batch_i=", i, pred_np[i],
        #       y_np[i],
        #       mask_np[i])
        out_y_np[i], out_mask_np[i] = vect_pred_min_assign(
            pred_np[i], y_np[i], mask_np[i], Y_MISSING_VAL
        )

    out_y = torch.Tensor(out_y_np)
    out_mask = torch.Tensor(out_mask_np)
    if torch.sum(mask) > 0:
        assert torch.sum(out_mask) > 0
    return out_y, out_mask


def mol_with_atom_index(mol, make2d=True):
    mol = Chem.Mol(mol)
    if make2d:
        Chem.AllChem.Compute2DCoords(mol)
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp(
            "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
        )
    return mol


def kcal_to_p(energies, T=298):
    k_kcal_mol = 0.001985875  # kcal/(mol⋅K)

    es_kcal_mol = np.array(energies)
    log_pstar = -es_kcal_mol / (k_kcal_mol * T)
    pstar = log_pstar - scipy.special.logsumexp(log_pstar)
    p = np.exp(pstar)
    p = p / np.sum(p)
    return p


def obabel_conf_gen(
    mol,
    ff_name="mmff94",
    rmsd_cutoff=0.5,
    conf_cutoff=16000000,
    energy_cutoff=10.0,
    confab_verbose=True,
    prob_cutoff=0.01,
):
    """
    Generate conformers using obabel.

    returns probs,
    """
    from openbabel import pybel
    from openbabel import openbabel as ob

    import tempfile

    tf = tempfile.NamedTemporaryFile(mode="w+")

    sdw = Chem.SDWriter(tf.name)
    sdw.write(mol)
    sdw.close()

    pybel_mol = next(pybel.readfile("sdf", tf.name))
    ob_mol = pybel_mol.OBMol

    ff = ob.OBForceField.FindForceField(ff_name)
    ff.Setup(ob_mol)
    ff.DiverseConfGen(rmsd_cutoff, conf_cutoff, energy_cutoff, confab_verbose)
    ff.GetConformers(ob_mol)
    energies = ob_mol.GetEnergies()

    probs = kcal_to_p(energies)

    output_format = "sdf"

    obconversion = ob.OBConversion()
    obconversion.SetOutFormat(output_format)
    rdkit_mols = []
    rdkit_weights = []

    for conf_num in range(len(probs)):
        ob_mol.SetConformer(conf_num)
        if probs[conf_num] >= prob_cutoff:
            rdkit_mol = Chem.MolFromMolBlock(
                obconversion.WriteString(ob_mol), removeHs=False
            )
            rdkit_mols.append(rdkit_mol)
            rdkit_weights.append(probs[conf_num])

    rdkit_weights = np.array(rdkit_weights)
    rdkit_weights = rdkit_weights / np.sum(rdkit_weights)

    if len(rdkit_mols) > 1:
        out_mol = rdkit_mols[0]
        for m in rdkit_mols[1:]:
            c = m.GetConformers()[0]
            out_mol.AddConformer(c)
    else:
        out_mol = rdkit_mols[0]

    return rdkit_weights, out_mol


def get_methyl_hydrogens(m):
    """
    returns list of (carbon index, list of methyl Hs)


    Originally in nmrabinitio
    """

    for c in m.GetSubstructMatches(Chem.MolFromSmarts("[CH3]")):
        yield c[0], [
            a.GetIdx()
            for a in m.GetAtomWithIdx(c[0]).GetNeighbors()
            if a.GetSymbol() == "H"
        ]


def create_methyl_atom_eq_classes(mol):
    """
    Take in a mol and return an equivalence-class assignment vector
    of a list of frozensets

    Originally in nmrabinitio
    """
    mh = get_methyl_hydrogens(mol)
    N = mol.GetNumAtoms()
    eq_classes = []
    for c, e in mh:
        eq_classes.append(frozenset(e))
    assert len(frozenset().intersection(*eq_classes)) == 0
    existing = frozenset().union(*eq_classes)
    for i in range(N):
        if i not in existing:
            eq_classes.append(frozenset([i]))
    return eq_classes


class EquivalenceClasses:
    """
    Equivalence classes of atoms and the kinds of questions
    we might want to ask. For example, treating all hydrogens
    in a methyl the same, or treating all equivalent atoms
    (from RDKit's perspective) the same.

    Originally in nmrabinitio
    """

    def __init__(self, eq):
        """
        eq is a list of disjoint frozen sets of the partitioned
        equivalence classes. Note that every element must be
        in at least one set and there can be no gaps.

        """

        all_elts = frozenset().union(*eq)

        N = np.max(list(frozenset().union(*eq))) + 1
        # assert all elements in set
        assert frozenset(list(range(N))) == all_elts

        self.eq = eq
        self.N = N

    def get_vect(self):
        assign_vect = np.zeros(self.N, dtype=int)
        for si, s in enumerate(sorted(self.eq, key=len)):
            for elt in s:
                assign_vect[elt] = si
        return assign_vect

    def get_pairwise(self):
        """
        From list of frozensets to all-possible pairwise assignment
        equivalence classes

        """
        eq = self.eq
        N = self.N

        assign_mat = np.ones((N, N), dtype=int) * -1
        eq_i = 0
        for s1_i, s1 in enumerate(sorted(eq, key=len)):
            for s2_i, s2 in enumerate(sorted(eq, key=len)):
                for i in s1:
                    for j in s2:
                        assign_mat[i, j] = eq_i
                eq_i += 1
        assert (assign_mat != -1).all()
        return assign_mat

    def get_series(self, index_name="atom_idx"):
        v = self.get_vect()
        res = [{index_name: i, "eq": a} for i, a in enumerate(v)]
        return pd.DataFrame(res).set_index(index_name)["eq"]

    def get_pairwise_series(
        self, index_names=["atomidx_1", "atomidx_2"], only_upper_tri=True
    ):
        m = self.get_pairwise()
        res = []
        for i in range(self.N):
            for j in range(self.N):
                if only_upper_tri:
                    if i > j:
                        continue
                res.append({index_names[0]: i, index_names[1]: j, "eq": m[i, j]})
        df = pd.DataFrame(res)
        df = df.set_index(index_names)
        return df["eq"]
