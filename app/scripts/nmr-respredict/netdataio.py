import numpy as np
import torch
import pandas as pd
import atom_features
import molecule_features
import edge_features
import torch.utils.data
import util
from atom_features import to_onehot
import pickle


class MoleculeDatasetMulti(torch.utils.data.Dataset):
    def __init__(
        self,
        records,
        MAX_N,
        feat_vert_args={},
        feat_edge_args={},
        adj_args={},
        mol_args={},
        dist_mat_args={},
        coupling_args={},
        pred_config={},
        passthrough_config={},
        combine_mat_vect=None,
        combine_mat_feat_adj=False,
        combine_mol_vect=False,
        max_conf_sample=1,
        extra_npy_filenames=[],
        frac_per_epoch=1.0,
        shuffle_observations=False,
        spect_assign=True,
        extra_features=None,
        allow_cache=True,
        recon_feat_edge_args={},
        methyl_eq_vert=False,
        methyl_eq_edge=False,
    ):
        self.records = records
        self.MAX_N = MAX_N
        if allow_cache:
            self.cache = {}
        else:
            self.cache = None
            # print("WARNING: running without cache")
        self.feat_vert_args = feat_vert_args
        self.feat_edge_args = feat_edge_args
        self.adj_args = adj_args
        self.mol_args = mol_args
        self.dist_mat_args = dist_mat_args
        self.coupling_args = coupling_args
        self.passthrough_config = passthrough_config
        # self.single_value = single_value
        self.combine_mat_vect = combine_mat_vect
        self.combine_mat_feat_adj = combine_mat_feat_adj
        self.combine_mol_vect = combine_mol_vect
        self.recon_feat_edge_args = recon_feat_edge_args
        # self.mask_zeroout_prob = mask_zeroout_prob

        self.extra_npy_filenames = extra_npy_filenames
        self.frac_per_epoch = frac_per_epoch
        self.shuffle_observations = shuffle_observations
        if shuffle_observations:
            print("WARNING: Shuffling observations")
        self.spect_assign = spect_assign
        self.extra_features = extra_features
        self.max_conf_sample = max_conf_sample

        self.rtp = RecordToPredict(**pred_config)

        self.use_conf_weights = True

        self.methyl_eq_vert = methyl_eq_vert
        self.methyl_eq_edge = methyl_eq_edge

    def __len__(self):
        return int(len(self.records) * self.frac_per_epoch)

    def cache_key(self, idx, conf_indices):
        return (idx, conf_indices)

    def _conf_avg(self, val, conf_weights):
        in_ndim = val.ndim
        for i in range(in_ndim - 1):
            conf_weights = conf_weights.unsqueeze(-1)

        p_sum = torch.sum(conf_weights)
        if torch.abs(1 - p_sum) > 1e-5:
            raise ValueError(f"Error, probs sum to {p_sum}")

        assert np.isfinite(val.numpy()).all()
        v = torch.sum(val * conf_weights, dim=0)
        # pickle.dump({'val' : val, 'v' : v,
        #              'conf_weights' : conf_weights},
        #             open("/tmp/test.pickle", 'wb'))
        assert np.isfinite(v.numpy()).all()
        # assert v.ndim == (in_ndim-1)
        return v

    def __getitem__(self, idx):
        if self.frac_per_epoch < 1.0:
            # randomly get an index each time
            idx = np.random.randint(len(self.records))
        record = self.records[idx]

        mol = record["rdmol"]

        max_conf_sample = self.max_conf_sample
        CONF_N = mol.GetNumConformers()
        if max_conf_sample == -1:
            # combine all conformer features
            conf_indices = tuple(range(CONF_N))
        else:
            conf_indices = tuple(
                np.sort(np.random.permutation(CONF_N)[:max_conf_sample])
            )

        if "weights" in record and self.use_conf_weights:
            conf_weights = torch.Tensor(record["weights"])
            assert len(conf_weights) == CONF_N
        else:
            conf_weights = torch.ones(CONF_N) / CONF_N

        #     print("conf_idx=", conf_idx)
        if self.cache is not None and self.cache_key(idx, conf_indices) in self.cache:
            return self.cache[self.cache_key(idx, conf_indices)]

        # mol/experiment features such as solvent
        f_mol = molecule_features.whole_molecule_features(record, **self.mol_args)

        f_vect_per_conf = torch.stack(
            [
                atom_features.feat_tensor_atom(
                    mol, conf_idx=conf_idx, **self.feat_vert_args
                )
                for conf_idx in conf_indices
            ]
        )
        # f_vect = torch.sum(f_vect_per_conf * conf_weights.unsqueeze(-1).unsqueeze(-1), dim=0)
        f_vect = self._conf_avg(f_vect_per_conf, conf_weights)

        if self.combine_mol_vect:
            f_vect = torch.cat(
                [f_vect, f_mol.reshape(1, -1).expand(f_vect.shape[0], -1)], -1
            )

        DATA_N = f_vect.shape[0]

        vect_feat = np.zeros((self.MAX_N, f_vect.shape[1]), dtype=np.float32)
        vect_feat[:DATA_N] = f_vect

        f_mat_per_conf = torch.stack(
            [
                molecule_features.feat_tensor_mol(
                    mol, conf_idx=conf_idx, **self.feat_edge_args
                )
                for conf_idx in conf_indices
            ]
        )
        # f_mat = torch.sum(f_mat_per_conf * conf_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), dim=0)
        f_mat = self._conf_avg(f_mat_per_conf, conf_weights)

        if self.combine_mat_vect:
            MAT_CHAN = f_mat.shape[2] + vect_feat.shape[1]
        else:
            MAT_CHAN = f_mat.shape[2]
        if MAT_CHAN == 0:  # Dataloader can't handle tensors with empty dimensions
            MAT_CHAN = 1
        mat_feat = np.zeros((self.MAX_N, self.MAX_N, MAT_CHAN), dtype=np.float32)
        # do the padding
        mat_feat[:DATA_N, :DATA_N, : f_mat.shape[2]] = f_mat

        if self.combine_mat_vect == "row":
            # row-major
            for i in range(DATA_N):
                mat_feat[i, :DATA_N, f_mat.shape[2] :] = f_vect
        elif self.combine_mat_vect == "col":
            # col-major
            for i in range(DATA_N):
                mat_feat[:DATA_N, i, f_mat.shape[2] :] = f_vect

        if self.methyl_eq_edge or self.methyl_eq_vert:
            methyl_atom_eq_classes = util.create_methyl_atom_eq_classes(mol)
            # if len(methyl_atom_eq_classes) < mol.GetNumAtoms():
            #     print(methyl_atom_eq_classes)

            eqc = util.EquivalenceClasses(methyl_atom_eq_classes)

            if self.methyl_eq_vert:
                vect_eq = eqc.get_vect()
                for eq_i in np.unique(vect_eq):
                    eq_mask = np.zeros(vect_feat.shape[0], dtype=np.bool)
                    eq_mask[: len(vect_eq)] = vect_eq == eq_i

                    vect_feat[eq_mask] == np.mean(vect_feat[eq_mask], axis=0)
            if self.methyl_eq_edge:
                mat_eq = eqc.get_pairwise()
                for eq_i in np.unique(mat_eq):
                    eq_mask = np.zeros(
                        (mat_feat.shape[0], mat_feat.shape[1]), dtype=np.bool
                    )
                    eq_mask[: len(mat_eq), : len(mat_eq)] = mat_eq == eq_i

                    mat_feat[eq_mask] == np.mean(mat_feat[eq_mask], axis=0)

        adj_nopad = molecule_features.feat_mol_adj(mol, **self.adj_args)
        adj = torch.zeros((adj_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj[:, : adj_nopad.shape[1], : adj_nopad.shape[2]] = adj_nopad

        if self.combine_mat_feat_adj:
            adj = torch.cat([adj, torch.Tensor(mat_feat).permute(2, 0, 1)], 0)

        ### Simple one-hot encoding for reconstruction
        adj_oh_nopad = molecule_features.feat_mol_adj(
            mol,
            split_weights=[1.0, 1.5, 2.0, 3.0],
            edge_weighted=False,
            norm_adj=False,
            add_identity=False,
        )

        adj_oh = torch.zeros((adj_oh_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj_oh[:, : adj_oh_nopad.shape[1], : adj_oh_nopad.shape[2]] = adj_oh_nopad

        ## per-edge features
        feat_edge_dict = edge_features.feat_edges(
            mol,
        )

        # pad each of these
        edge_edge_nopad = feat_edge_dict["edge_edge"]
        edge_edge = torch.zeros((edge_edge_nopad.shape[0], self.MAX_N, self.MAX_N))

        edge_feat_nopad = feat_edge_dict["edge_feat"]
        ### NOT IMPLEMENTED RIGHT NOW
        edge_feat = torch.zeros((self.MAX_N, 1))  # edge_feat_nopad.shape[1]))

        edge_vert_nopad = feat_edge_dict["edge_vert"]
        edge_vert = torch.zeros((edge_vert_nopad.shape[0], self.MAX_N, self.MAX_N))

        ### FIXME FIXME do conf averaging
        atomicnos, coords = util.get_nos_coords(
            mol, conf_indices[0]
        )  ### DEBUG THIS FIXME)
        coords_t = torch.zeros((self.MAX_N, 3))
        coords_t[: len(coords), :] = torch.Tensor(coords)

        # recon features
        dist_mat_per_conf = torch.stack(
            [
                torch.Tensor(
                    molecule_features.dist_mat(
                        mol, conf_idx=conf_idx, **self.dist_mat_args
                    )
                )
                for conf_idx in conf_indices
            ]
        )
        dist_mat = self._conf_avg(dist_mat_per_conf, conf_weights)
        dist_mat_t = torch.zeros((self.MAX_N, self.MAX_N, dist_mat.shape[-1]))
        dist_mat_t[: len(coords), : len(coords), :] = dist_mat

        ##################################################
        ### Create output values to predict
        ##################################################
        pred_out = self.rtp(record, self.MAX_N)
        vert_pred = pred_out["vert"].data
        vert_pred_mask = ~pred_out["vert"].mask

        edge_pred = pred_out["edge"].data
        edge_pred_mask = ~pred_out["edge"].mask

        # input mask
        input_mask = torch.zeros(self.MAX_N)
        input_mask[:DATA_N] = 1.0

        v = {
            "adj": adj,
            "vect_feat": vect_feat,
            "mat_feat": mat_feat,
            "mol_feat": f_mol,
            "dist_mat": dist_mat_t,
            #'vals' : vals,
            "vert_pred": vert_pred,
            #'pred_mask' : pred_mask,
            "vert_pred_mask": vert_pred_mask,
            "edge_pred": edge_pred,
            "edge_pred_mask": edge_pred_mask,
            "adj_oh": adj_oh,
            "coords": coords_t,
            "input_mask": input_mask,
            "input_idx": idx,
            "edge_edge": edge_edge,
            "edge_vert": edge_vert,
            "edge_feat": edge_feat,
        }

        #################################################
        ### extra field-to-arg mapping
        #################################################
        # coupling_types = encode_coupling_types(record)
        for p_k, p_v in self.passthrough_config.items():
            if p_v["func"] == "coupling_types":
                kv = "passthrough_" + p_k
                v[kv] = coupling_types(record, self.MAX_N, **p_v)

        # semisupervised features for edges
        gp = molecule_features.recon_features_edge(mol, **self.recon_feat_edge_args)
        v["recon_features_edge"] = molecule_features.pad(gp, self.MAX_N).astype(
            np.float32
        )

        for k, kv in v.items():
            if not np.isfinite(kv).all():
                debug_filename = "/tmp/pred_debug.pickle"
                pickle.dump({"v": v, "nofinite_key": k}, open(debug_filename, "wb"))
                raise Exception(
                    f"{k} has non-finite vals, debug written to {debug_filename}"
                )
        if self.cache is not None:
            self.cache[self.cache_key(idx, conf_indices)] = v

        return v


class RecordToPredict(object):
    """
    Convert a whole record into predictor features


    """

    def __init__(self, vert={}, edge={}):
        self.vert_configs = vert
        self.edge_configs = edge

    def __call__(self, record, MAX_N):
        """
        Returns masked arrays, where mask = True ==> data is not observed
        """

        vert_out_num = len(self.vert_configs)
        vert_out = np.ma.zeros((MAX_N, vert_out_num), dtype=np.float32)
        vert_out.mask = True

        edge_out_num = len(self.edge_configs)
        edge_out = np.ma.zeros((MAX_N, MAX_N, edge_out_num), dtype=np.float32)
        edge_out.mask = True

        for vert_out_i, vert_config in enumerate(self.vert_configs):
            if "data_field" in vert_config:
                d = record.get(vert_config["data_field"])
                if "index" in vert_config:
                    d = d[vert_config["index"]]
                for i, v in d.items():
                    vert_out[i, vert_out_i] = v

        for edge_out_i, edge_config in enumerate(self.edge_configs):
            if "data_field" in edge_config:
                d = record.get(edge_config["data_field"], {})
                symmetrize = edge_config.get("symmetrize", True)
                for (i, j), v in d.items():
                    edge_out[i, j, edge_out_i] = v
                    if symmetrize:
                        edge_out[j, i, edge_out_i] = v
        return {"edge": edge_out, "vert": vert_out}


def coupling_types(
    record, MAX_N, coupling_types_lut=[("CH", 1), ("HH", 2), ("HH", 3)], **kwargs
):
    coupling_types = record["coupling_types"]

    coupling_types_encoded = (
        np.ones((MAX_N, MAX_N), dtype=np.int32) * -2
    )  # the not-observed val

    for (coupling_idx1, coupling_idx2), ct in coupling_types.items():
        ct_lut_val = -1
        for v_i, c_v in enumerate(coupling_types_lut):
            if ct == tuple(c_v):
                ct_lut_val = v_i

        coupling_types_encoded[coupling_idx1, coupling_idx2] = ct_lut_val
        coupling_types_encoded[coupling_idx2, coupling_idx1] = ct_lut_val

    return coupling_types_encoded
