"""
Code to wrap the model such that we can easily use
it as a predictor. The goal is to move as much 
model-specific code out of the main codepath. 

"""

import pickle
import torch
import netdataio
import netutil
import pandas as pd


class PredModel(object):
    """
    Predictor can predict two types of values,
    per-vert and per-edge.

    """

    def __init__(
        self,
        meta_filename,
        checkpoint_filename,
        USE_CUDA=False,
        override_pred_config=None,
    ):
        meta = pickle.load(open(meta_filename, "rb"))

        self.meta = meta

        self.USE_CUDA = USE_CUDA

        if self.USE_CUDA:
            net = torch.load(checkpoint_filename)
        else:
            net = torch.load(
                checkpoint_filename, map_location=lambda storage, loc: storage
            )

        self.net = net
        self.net.eval()
        self.override_pred_config = override_pred_config

    def pred(
        self,
        records,
        BATCH_SIZE=32,
        debug=False,
        prog_bar=False,
        pred_fields=None,
        return_res=False,
        num_workers=0,
    ):
        dataset_hparams = self.meta["dataset_hparams"]
        MAX_N = self.meta.get("max_n", 32)

        USE_CUDA = self.USE_CUDA

        feat_vect_args = dataset_hparams["feat_vect_args"]
        feat_edge_args = dataset_hparams.get("feat_edge_args", {})
        adj_args = dataset_hparams["adj_args"]
        mol_args = dataset_hparams.get("mol_args", {})
        dist_mat_args = dataset_hparams.get("dist_mat_args", {})
        coupling_args = dataset_hparams.get("coupling_args", {})
        extra_data_args = dataset_hparams.get("extra_data", [])
        other_args = dataset_hparams.get("other_args", {})

        # pred_config = self.meta.get('pred_config', {})
        # passthrough_config = self.meta.get('passthrough_config', {})

        ### pred-config controls the extraction of true values for supervised
        ### training and is generally not used at pure-prediction time
        if self.override_pred_config is not None:
            pred_config = self.override_pred_config
        else:
            pred_config = self.meta["pred_config"]
        passthrough_config = self.meta["passthrough_config"]

        # we force set this here
        if "allow_cache" in other_args:
            del other_args["allow_cache"]
        ds = netdataio.MoleculeDatasetMulti(
            records,
            MAX_N,
            feat_vect_args,
            feat_edge_args,
            adj_args,
            mol_args,
            dist_mat_args=dist_mat_args,
            coupling_args=coupling_args,
            pred_config=pred_config,
            passthrough_config=passthrough_config,
            # combine_mat_vect=COMBINE_MAT_VECT,
            allow_cache=False,
            **other_args,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
        )

        allres = []
        alltrue = []
        results_df = []
        m_pos = 0

        res = netutil.run_epoch(
            self.net,
            None,
            None,
            dl,
            pred_only=True,
            USE_CUDA=self.USE_CUDA,
            return_pred=True,
            print_shapes=debug,
            desc="predict",
            progress_bar=prog_bar,
        )

        if return_res:
            return res  # debug
        # by default we predict everything the net throws as tus
        if pred_fields is None:
            pred_fields = [f for f in list(res.keys()) if f.startswith("pred_")]

        for f in pred_fields:
            if f not in res:
                raise Exception(f"{f} not in res, {list(res.keys())}")

        per_vert_fields = []
        per_edge_fields = []
        for field in pred_fields:
            if len(res[field].shape) == 3:
                per_vert_fields.append(field)
            else:
                per_edge_fields.append(field)

        ### create the per-vertex fields
        per_vert_out = []
        for rec_i, rec in enumerate(records):
            rdmol = rec["rdmol"]
            atom_n = rdmol.GetNumAtoms()

            for atom_idx in range(atom_n):
                vert_rec = {"rec_idx": rec_i, "atom_idx": atom_idx}
                for field in per_vert_fields:
                    for ji, v in enumerate(res[field][rec_i, atom_idx]):
                        vr = vert_rec.copy()
                        vr["val"] = v
                        vr["field"] = field
                        vr["pred_chan"] = ji
                        per_vert_out.append(vr)

        vert_results_df = pd.DataFrame(per_vert_out)

        ### create the per-edge fields
        if len(per_edge_fields) == 0:
            edge_results_df = None
        else:
            per_edge_out = []
            for rec_i, rec in enumerate(records):
                rdmol = rec["rdmol"]
                atom_n = rdmol.GetNumAtoms()

                for atomidx_1 in range(atom_n):
                    for atomidx_2 in range(atomidx_1 + 1, atom_n):
                        edge_rec = {
                            "rec_idx": rec_i,
                            "atomidx_1": atomidx_1,
                            "atomidx_2": atomidx_2,
                        }

                        for field in per_edge_fields:
                            for ji, v in enumerate(
                                res[field][rec_i, atomidx_1, atomidx_2]
                            ):
                                er = edge_rec.copy()
                                er["val"] = v
                                er["field"] = field
                                er["pred_chan"] = ji
                                per_edge_out.append(er)
            edge_results_df = pd.DataFrame(per_edge_out)
            # edge_results_df['atomidx_1'] = edge_results_df['atomidx_1'].astype(int)
            # edge_results_df['atomidx_2'] = edge_results_df['atomidx_2'].astype(int)

        return vert_results_df, edge_results_df


if __name__ == "__main__":
    pass
