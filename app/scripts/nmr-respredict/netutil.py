import torch

import netdataio
import pickle
import copy

import torch
import torch.autograd
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import os
import util
import nets

default_atomicno = [1, 6, 7, 8, 9, 15, 16, 17]

### Create datasets and data loaders

default_feat_vect_args = dict(
    feat_atomicno_onehot=default_atomicno,
    feat_pos=False,
    feat_atomicno=True,
    feat_valence=True,
    aromatic=True,
    hybridization=True,
    partial_charge=False,
    formal_charge=True,  # WE SHOULD REALLY USE THIS
    r_covalent=False,
    total_valence_onehot=True,
    mmff_atom_types_onehot=False,
    r_vanderwals=False,
    default_valence=True,
    rings=True,
)

default_feat_edge_args = dict(feat_distances=False, feat_r_pow=None)

default_split_weights = [1, 1.5, 2, 3]

default_adj_args = dict(
    edge_weighted=False,
    norm_adj=True,
    add_identity=True,
    split_weights=default_split_weights,
)


default_mol_args = dict()  # possible_solvents= ['CDCl3', 'DMSO-d6', 'D2O', 'CCl4'])

default_dist_mat_args = dict()

DEFAULT_DATA_HPARAMS = {
    "feat_vect_args": default_feat_vect_args,
    "feat_edge_args": default_feat_edge_args,
    "adj_args": default_adj_args,
    "mol_args": default_mol_args,
    "dist_mat_args": default_dist_mat_args,
    "coupling_args": {"compute_coupling": False},
}


def dict_combine(d1, d2):
    d1 = copy.deepcopy(d1)
    d1.update(d2)
    return d1


class CVSplit:
    def __init__(self, how, **args):
        self.how = how
        self.args = args

    def get_phase(self, mol, fp):
        if self.how == "morgan_fingerprint_mod":
            mod = self.args["mod"]
            test = self.args["test"]

            if (fp % mod) in test:
                return "test"
            else:
                return "train"

        else:
            raise ValueError(f"unknown method {self.how}")


def make_dataset(
    dataset_config,
    hparams,
    pred_config,
    MAX_N,
    cv_splitter,
    train_sample=0,
    passthrough_config={},
):
    """ """

    filename = dataset_config["filename"]
    phase = dataset_config.get("phase", "train")
    dataset_spect_assign = dataset_config.get("spect_assign", True)
    frac_per_epoch = dataset_config.get("frac_per_epoch", 1.0)
    force_tgt_nucs = dataset_config.get("force_tgt_nucs", None)
    d = pickle.load(open(filename, "rb"))
    if dataset_config.get("subsample_to", 0) > 0:
        if len(d) > dataset_config["subsample_to"]:
            d = d.sample(
                dataset_config["subsample_to"],
                random_state=dataset_config.get("subsample_seed", 0),
            )

    filter_max_n = dataset_config.get("filter_max_n", 0)
    spect_dict_field = dataset_config.get("spect_dict_field", "spect_dict")
    print("THE SPECT DICT IS", spect_dict_field)
    filter_bond_max_n = dataset_config.get("filter_bond_max_n", 0)

    if filter_max_n > 0:
        d["atom_n"] = d.rdmol.apply(lambda m: m.GetNumAtoms())

        print("filtering for atom max_n <=", filter_max_n, " from", len(d))
        d = d[d.atom_n <= filter_max_n]
        print("after filter length=", len(d))

    if filter_bond_max_n > 0:
        d["bond_n"] = d.rdmol.apply(lambda m: m.GetNumBonds())

        print("filtering for bond max_n <=", filter_bond_max_n, " from", len(d))
        d = d[d.bond_n <= filter_bond_max_n]
        print("after filter length=", len(d))

    d_phase = d.apply(
        lambda row: cv_splitter.get_phase(row.rdmol, row.morgan4_crc32), axis=1
    )

    df = d[d_phase == phase]
    if force_tgt_nucs is None:
        if spect_dict_field in df:
            num_tgt_nucs = len(df.iloc[0][spect_dict_field])
        else:
            num_tgt_nucs = len(df.iloc[0].spect_list)

    else:
        num_tgt_nucs = force_tgt_nucs

    datasets = {}

    # dataset_extra_data = []
    # for extra_data_rec in extra_data:
    #     extra_data_rec = extra_data_rec.copy()
    #     extra_data_rec['filenames'] = df[extra_data_rec['name'] + "_filename"].tolist()
    #     dataset_extra_data.append(extra_data_rec)
    other_args = hparams.get("other_args", {})

    if dataset_spect_assign:
        spect_data = df[spect_dict_field].tolist()
    else:
        if "spect_list" in df:
            spect_data = df.spect_list.tolist()
        else:

            def to_unassign(list_of_spect_dict):
                return [(list(n.keys()), list(n.values())) for n in list_of_spect_dict]

            print("WARNING: Manually discarding assignment information")
            spect_data = df[spect_dict_field].apply(to_unassign).tolist()

    ds = netdataio.MoleculeDatasetMulti(  # df.rdmol.tolist(),
        # spect_data,
        df.to_dict("records"),
        MAX_N,  # num_tgt_nucs,
        hparams["feat_vect_args"],
        hparams["feat_edge_args"],
        hparams["adj_args"],
        hparams["mol_args"],
        hparams["dist_mat_args"],
        hparams["coupling_args"],
        pred_config=pred_config,
        passthrough_config=passthrough_config,
        # extra_npy_filenames = dataset_extra_data,
        frac_per_epoch=frac_per_epoch,
        spect_assign=dataset_spect_assign,
        **other_args,
    )

    print(f"{phase} has {len(df)} records")

    phase_data = {"mol": df.rdmol, "spect": spect_data, "df": df}
    return ds, phase_data


def create_checkpoint_func(every_n, filename_str):
    def checkpoint(epoch_i, net, optimizer):
        if epoch_i % every_n > 0:
            return {}
        checkpoint_filename = filename_str.format(epoch_i=epoch_i)
        t1 = time.time()
        torch.save(net.state_dict(), checkpoint_filename + ".state")
        torch.save(net, checkpoint_filename + ".model")
        t2 = time.time()
        return {"savetime": (t2 - t1)}

    return checkpoint


def run_epoch(
    net,
    optimizer,
    criterion,
    dl,
    pred_only=False,
    USE_CUDA=True,
    return_pred=False,
    desc="train",
    print_shapes=False,
    progress_bar=True,
    writer=None,
    epoch_i=None,
    res_skip_keys=[],
    clip_grad_value=None,
    scheduler=None,
):
    t1_total = time.time()

    ### DEBUGGING we should clean this up
    MAX_N = 64

    if not pred_only:
        net.train()
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        if optimizer is not None:
            optimizer.zero_grad()
        torch.set_grad_enabled(False)

    accum_pred = []
    extra_loss_fields = {}

    running_loss = 0.0
    total_points = 0
    total_compute_time = 0.0
    if progress_bar:
        iterator = tqdm(enumerate(dl), total=len(dl), desc=desc, leave=False)
    else:
        iterator = enumerate(dl)

    input_row_count = 0
    for i_batch, batch in iterator:
        t1 = time.time()
        if print_shapes:
            for k, v in batch.items():
                print("{}.shape={}".format(k, v.shape))
        if not pred_only:
            optimizer.zero_grad()

        if isinstance(batch, dict):
            batch_t = {k: move(v, USE_CUDA) for k, v in batch.items()}
            use_geom = False
        else:
            batch_t = batch.to("cuda")
            use_geom = True
        # with torch.autograd.detect_anomaly():
        # for k, v in batch_t.items():
        #     assert not torch.isnan(v).any()

        if use_geom:
            res = net(batch_t)
            pred_mask_batch_t = batch_t.pred_mask.reshape(-1, MAX_N, 1)
            y_batch_t = batch_t.y.reshape(-1, MAX_N, 1)
            input_mask_t = batch_t.input_mask.reshape(-1, MAX_N, 1)
            input_idx_t = batch_t.input_idx.reshape(-1, 1)

        else:
            res = net(**batch_t)
            vert_pred_batch_t = batch_t["vert_pred"]
            vert_pred_mask_batch_t = batch_t["vert_pred_mask"]
            edge_pred_batch_t = batch_t["edge_pred"]
            edge_pred_mask_batch_t = batch_t["edge_pred_mask"]

            input_mask_t = batch_t["input_mask"]
            input_idx_t = batch_t["input_idx"]
        if return_pred:
            accum_pred_val = {}
            if isinstance(res, dict):
                for k, v in res.items():
                    if k not in res_skip_keys:
                        if isinstance(res[k], torch.Tensor):
                            accum_pred_val[k] = res[k].cpu().detach().numpy()
            else:
                accum_pred_val["res"] = res.cpu().detach().numpy()
            accum_pred_val["vert_pred_mask"] = (
                vert_pred_mask_batch_t.cpu().detach().numpy()
            )
            accum_pred_val["vert_pred"] = vert_pred_batch_t.cpu().detach().numpy()
            accum_pred_val["edge_pred_mask"] = (
                edge_pred_mask_batch_t.cpu().detach().numpy()
            )
            accum_pred_val["edge_pred"] = edge_pred_batch_t.cpu().detach().numpy()
            accum_pred_val["input_idx"] = (
                input_idx_t.cpu().detach().numpy().reshape(-1, 1)
            )
            accum_pred_val["input_mask"] = input_mask_t.cpu().detach().numpy()

            # extra fields
            for k, v in batch.items():
                if k.startswith("passthrough_"):
                    accum_pred_val[k] = v.cpu().detach().numpy()

            accum_pred.append(accum_pred_val)
        loss_dict = {}
        if criterion is None:
            loss = 0.0
        else:
            loss = criterion(
                res,
                vert_pred_batch_t,
                vert_pred_mask_batch_t,
                edge_pred_batch_t,
                edge_pred_mask_batch_t,
                ## EDGE HERE
                input_mask_t,
            )
            if isinstance(loss, dict):
                loss_dict = loss
                loss = loss_dict["loss"]

        if not pred_only:
            loss.backward()
            # for n, p in net.named_parameters():
            #     if 'weight' in n:
            #         writer.add_scalar(f"grads/{n}", torch.max(torch.abs(p.grad)), epoch_i)

            if clip_grad_value is not None:
                nn.utils.clip_grad_value_(net.parameters(), clip_grad_value)

            optimizer.step()

        train_points = batch["input_mask"].shape[0]
        if criterion is not None:
            running_loss += loss.item() * train_points
            for k, v in loss_dict.items():
                if k not in extra_loss_fields:
                    extra_loss_fields[k] = v.item() * train_points
                else:
                    extra_loss_fields[k] += v.item() * train_points

        total_points += train_points

        t2 = time.time()
        total_compute_time += t2 - t1

        input_row_count += batch["adj"].shape[0]

        if scheduler is not None:
            scheduler.step()
    t2_total = time.time()

    # print('running_loss=', running_loss)
    total_points = max(total_points, 1)
    res = {
        "timing": 0.0,
        "running_loss": running_loss,
        "total_points": total_points,
        "mean_loss": running_loss / total_points,
        "runtime": t2_total - t1_total,
        "compute_time": total_compute_time,
        "run_efficiency": total_compute_time / (t2_total - t1_total),
        "pts_per_sec": input_row_count / (t2_total - t1_total),
    }

    for elf, v in extra_loss_fields.items():
        # print(f"extra loss fields {elf} = {v}")
        res[f"loss_total_{elf}"] = v
        res[f"loss_mean_{elf}"] = v / total_points

    if return_pred:
        keys = accum_pred[0].keys()
        for k in keys:
            accum_pred_v = np.vstack([a[k] for a in accum_pred])
            res[f"pred_{k}"] = accum_pred_v

    return res


VALIDATE_EVERY = 1


def generic_runner(
    net,
    optimizer,
    scheduler,
    criterion,
    dl_train,
    dl_test,
    MAX_EPOCHS=1000,
    USE_CUDA=True,
    use_std=False,
    writer=None,
    validate_funcs=None,
    checkpoint_func=None,
    prog_bar=True,
    clip_grad_value=None,
):
    # loss_scale = torch.Tensor(loss_scale)
    # std_scale = torch.Tensor(std_scale)

    res_skip_keys = ["g_in", "g_decode"]

    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        per_batch_scheduler = scheduler
    else:
        per_batch_scheduler = None

    for epoch_i in tqdm(range(MAX_EPOCHS)):
        running_loss = 0.0
        total_compute_time = 0.0
        t1_total = time.time()

        net.train()
        train_res = run_epoch(
            net,
            optimizer,
            criterion,
            dl_train,
            pred_only=False,
            USE_CUDA=USE_CUDA,
            return_pred=True,
            progress_bar=prog_bar,
            desc="train",
            writer=writer,
            epoch_i=epoch_i,
            res_skip_keys=res_skip_keys,
            clip_grad_value=clip_grad_value,
            scheduler=per_batch_scheduler,
        )

        [v(train_res, "train_", epoch_i) for v in validate_funcs]

        if epoch_i % VALIDATE_EVERY == 0:
            net.eval()
            test_res = run_epoch(
                net,
                optimizer,
                criterion,
                dl_test,
                pred_only=True,
                USE_CUDA=USE_CUDA,
                progress_bar=prog_bar,
                return_pred=True,
                desc="validate",
                res_skip_keys=res_skip_keys,
            )
            [v(test_res, "validate_", epoch_i) for v in validate_funcs]

        if checkpoint_func is not None:
            checkpoint_func(epoch_i=epoch_i, net=net, optimizer=optimizer)

        if scheduler is not None and (per_batch_scheduler is None):
            scheduler.step()


def move(tensor, cuda=False):
    if cuda:
        if isinstance(tensor, nn.Module):
            return tensor.cuda()
        else:
            return tensor.cuda(non_blocking=True)
    else:
        return tensor.cpu()


class PredModel(object):
    def __init__(self, meta_filename, checkpoint_filename, USE_CUDA=False):
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

    def pred(
        self, rdmols, values, whole_records, BATCH_SIZE=32, debug=False, prog_bar=False
    ):
        dataset_hparams = self.meta["dataset_hparams"]
        MAX_N = self.meta.get("max_n", 32)

        USE_CUDA = self.USE_CUDA

        COMBINE_MAT_VECT = "row"

        feat_vect_args = dataset_hparams["feat_vect_args"]
        feat_edge_args = dataset_hparams.get("feat_edge_args", {})
        adj_args = dataset_hparams["adj_args"]
        mol_args = dataset_hparams.get("mol_args", {})
        extra_data_args = dataset_hparams.get("extra_data", [])
        other_args = dataset_hparams.get("other_args", {})

        ds = netdataio.MoleculeDatasetMulti(
            rdmols,
            values,
            whole_records,
            MAX_N,
            len(self.meta["tgt_nucs"]),
            feat_vect_args,
            feat_edge_args,
            adj_args,
            mol_args,
            combine_mat_vect=COMBINE_MAT_VECT,
            extra_npy_filenames=extra_data_args,
            allow_cache=False,
            **other_args,
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        allres = []
        alltrue = []
        results_df = []
        m_pos = 0

        res = run_epoch(
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

        for rd_mol_i, (rdmol, true_val) in enumerate(zip(rdmols, values)):
            for nuc_i, nuc in enumerate(self.meta["tgt_nucs"]):
                true_nuc_spect = true_val[nuc_i]
                for atom_idx, true_shift in true_nuc_spect.items():
                    atom_res = {}
                    for pred_key in ["pred_mu", "pred_std"]:
                        atom_res[pred_key] = res[pred_key][rd_mol_i, atom_idx, nuc_i]
                    atom_res["nuc_i"] = nuc_i
                    atom_res["nuc"] = nuc
                    atom_res["atom_idx"] = atom_idx
                    atom_res["m_pos"] = rd_mol_i
                    atom_res["value"] = true_shift

                    results_df.append(atom_res)

        results_df = pd.DataFrame(results_df)
        return results_df


def rand_dict(d):
    p = {}
    for k, v in d.items():
        if isinstance(v, list):
            p[k] = v[np.random.randint(len(v))]
        else:
            p[k] = v
    return p


def create_validate_func(tgt_nucs):
    def val_func(res):  # val, mask, truth):
        val = res["pred_res"]
        mask = res["pred_mask"]
        truth = res["pred_truth"]
        res = {}
        for ni, n in enumerate(tgt_nucs):
            delta = (val[:, :, ni] - truth[:, :, ni])[mask[:, :, ni] > 0].flatten()
            if len(delta) == 0:
                continue
            res[f"{n}/test_std_err"] = np.std(delta)
            res[f"{n}/test_max_error"] = np.max(np.abs(delta))
            res[f"{n}/test_mean_abs_err"] = np.mean(np.abs(delta))
            res[f"{n}/test_abs_err_90"] = np.percentile(np.abs(delta), 90)
        return res

    return val_func


def create_shift_uncertain_validate_func(config, writer):
    def val_func(input_res, prefix, epoch_i):  # val, mask, truth):
        mu = input_res["pred_shift_mu"]
        std = input_res["pred_shift_std"]
        pred_mask = input_res["pred_vert_pred_mask"]
        truth = input_res["pred_vert_pred"]
        mean_loss = input_res["mean_loss"]
        # print("validate_func mu.shape=", mu.shape, "Truth.shape=", truth.shape)
        res = {
            "mean_loss": mean_loss,
            "run_epoch_time": input_res["runtime"],
            "run_efficinecy": input_res["run_efficiency"],
            "run_pts_per_sec": input_res["pts_per_sec"],
        }

        # extra losses
        for k, v in input_res.items():
            if "loss_total_" in k:
                res[k] = v
            if "loss_mean_" in k:
                res[k] = v

        for ni, n in enumerate(config["fields"]):
            delta = (mu[:, :, ni] - truth[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
            if len(delta) == 0:
                continue
            masked_std = (std[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
            res[f"{n}/delta_std"] = np.std(delta)
            res[f"{n}/delta_max"] = np.max(np.abs(delta))
            res[f"{n}/delta_mean_abs"] = np.mean(np.abs(delta))
            res[f"{n}/delta_abs_90"] = np.percentile(np.abs(delta), 90)
            res[f"{n}/std/mean"] = np.mean(masked_std)
            res[f"{n}/std/min"] = np.min(masked_std)
            res[f"{n}/std/max"] = np.max(masked_std)
            delta = np.nan_to_num(delta)
            masked_std = np.nan_to_num(masked_std)

            writer.add_histogram(f"{prefix}{n}_delta_abs", np.abs(delta), epoch_i)
            writer.add_histogram(
                f"{prefix}{n}_delta_abs_dB", np.log10(np.abs(delta) + 1e-6), epoch_i
            )

            writer.add_histogram(f"{n}_std", masked_std, epoch_i)
            sorted_delta_abs = np.abs(delta)[np.argsort(masked_std)]

            for frac in [10, 50, 90]:
                res[f"{n}/sorted_delta_abs_{frac}"] = np.mean(
                    sorted_delta_abs[: int(frac / 100.0 * len(sorted_delta_abs))]
                )
                res[f"{n}/sorted_delta_abs_{frac}_max"] = np.max(
                    sorted_delta_abs[: int(frac / 100.0 * len(sorted_delta_abs))]
                )

        exception = False

        for metric_name, metric_val in res.items():
            # print(f"{metric_name} is {metric_val}")

            if not np.isfinite(metric_val):
                exception = True
                # print(f"{metric_name} is {metric_val}")
            writer.add_scalar("{}{}".format(prefix, metric_name), metric_val, epoch_i)
        if exception:
            raise ValueError(f"{prefix}{metric_name} found some nans")

    return val_func


def create_permutation_validate_func(tgt_nucs, writer):
    def val_func(input_res, prefix, epoch_i):  # val, mask, truth):
        mu = input_res["pred_mu"]
        std = input_res["pred_std"]
        pred_mask = input_res["pred_vert_pred_mask"]
        truth = input_res["pred_vert_pred"]
        mean_loss = input_res["mean_loss"]
        # print("validate_func mu.shape=", mu.shape, "Truth.shape=", truth.shape)

        res = {}
        for ni, n in enumerate(tgt_nucs):
            out_y, out_mask = util.min_assign(
                torch.Tensor(mu[:, :, ni]),
                torch.Tensor(truth[:, :, ni]),
                torch.Tensor(pred_mask[:, :, ni]),
            )
            out_y = out_y.numpy()
            out_mask = out_mask.numpy()
            delta = (mu[:, :, ni] - out_y)[out_mask > 0].flatten()
            delta = np.nan_to_num(delta)
            if len(delta) > 0:
                res[f"{n}/perm_delta_max"] = np.max(np.abs(delta))
                res[f"{n}/perm_delta_mean_abs"] = np.mean(np.abs(delta))

        exception = False

        for metric_name, metric_val in res.items():
            if not np.isfinite(metric_val):
                exception = True
                print(f"{metric_name} is {metric_val}")
            writer.add_scalar("{}{}".format(prefix, metric_name), metric_val, epoch_i)
        if exception:
            raise ValueError("found some nans")

    return val_func


def create_save_val_func(checkpoint_base_dir):
    def val_func(input_res, prefix, epoch_i):  # val, mask, truth):
        # print("input_res.keys()=", list(input_res.keys()))

        if epoch_i % 10 != 0:
            return

        mu = input_res["pred_mu"]
        std = input_res["pred_std"]
        pred_mask = input_res["pred_mask"]
        truth = input_res["pred_truth"]
        mean_loss = input_res["mean_loss"]
        pred_input_idx = input_res["pred_input_idx"]
        outfile = checkpoint_base_dir + f".{prefix}.{epoch_i:08d}.output"

        out = {
            "mu": mu,
            "std": std,
            "pred_mask": pred_mask,
            "pred_truth": truth,
            "pred_input_idx": pred_input_idx,
            "mean_loss": mean_loss,
        }
        for k, v in input_res.items():
            out[f"res_{k}"] = v
        print("saving", outfile)
        pickle.dump(out, open(outfile, "wb"))

    return val_func


def create_coupling_validate_func(config, writer):
    coupling_type_list = config.get("coupling_lut", {})
    coupling_type_lut = {-1: "other"}
    for si, s in enumerate(coupling_type_list):
        coupling_type_lut[si] = f"{s[1]}J{s[0]}"
    coupling_index = config["coupling_index"]

    def val_func(input_res, prefix, epoch_i):
        # print("input_res.keys() =", input_res.keys())
        # print( input_res['pred_edge_pred'].shape)

        coupling_pred = input_res["pred_coupling_pred"][:, :, :, coupling_index]
        coupling_truth = input_res["pred_edge_pred"][:, :, :, coupling_index]
        coupling_mask = input_res["pred_edge_pred_mask"][:, :, :, coupling_index]

        # coupling_truth =

        BATCH_N, MAX_N, _ = coupling_pred.shape

        coupling_types = input_res["pred_passthrough_coupling_types_encoded"]

        delta = coupling_pred - coupling_truth

        delta_present = delta[coupling_mask > 0]
        delta_types = coupling_types[coupling_mask > 0]

        metrics = {
            "coupling_delta_abs": np.mean(np.abs(delta_present)),
            "coupling_delta_sq": np.mean(delta_present**2),
            "coupling_n": np.sum(coupling_mask),
        }

        # break errors into luts
        different_coupling_types = {k: list() for k in coupling_type_lut.keys()}
        for ct, v in zip(delta_types, delta_present):
            different_coupling_types[ct].append(np.abs(v))

        for k, v in coupling_type_lut.items():
            # print('adding metric', f"coupling_{v}_delta_abs")
            if len(different_coupling_types[k]) > 0:
                metrics[f"coupling_{v}_delta_abs"] = np.mean(
                    different_coupling_types[k]
                )
            else:
                pass

            # print("Warning, only",
            # len(different_coupling_types[k]),
            #          "entries for", k, v)
            # print("done")

        exception = False
        for metric_name, metric_val in metrics.items():
            if not np.isfinite(metric_val):
                exception = True
                print(f"{metric_name} is {metric_val}")

            writer.add_scalar("{}{}".format(prefix, metric_name), metric_val, epoch_i)
        if exception:
            raise ValueError(f"{prefix}{metric_name} found some nans")

        # mu = input_res['pred_mu']
        # std = input_res['pred_std']
        # pred_mask = input_res['pred_mask']
        # truth = input_res['pred_truth']
        # mean_loss = input_res['mean_loss']
        # #print("validate_func mu.shape=", mu.shape, "Truth.shape=", truth.shape)
        # res = {'mean_loss' : mean_loss,
        #        'run_epoch_time' : input_res['runtime'],
        #        'run_efficinecy' : input_res['run_efficiency'],
        #        'run_pts_per_sec' : input_res['pts_per_sec']}

        # # extra losses
        # for k, v in input_res.items():
        #     if 'loss_total_' in k:
        #         res[k] = v
        #     if 'loss_mean_' in k:
        #         res[k] = v

        # for ni, n in enumerate(tgt_nucs):
        #     delta = (mu[:, :, ni] - truth[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
        #     if len(delta) == 0:
        #         continue
        #     masked_std = (std[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
        #     res[f"{n}/delta_std"] = np.std(delta)
        #     res[f"{n}/delta_max"] = np.max(np.abs(delta))
        #     res[f"{n}/delta_mean_abs"] = np.mean(np.abs(delta))
        #     res[f"{n}/delta_abs_90"] = np.percentile(np.abs(delta), 90)
        #     res[f"{n}/std/mean"] = np.mean(masked_std)
        #     res[f"{n}/std/min"] = np.min(masked_std)
        #     res[f"{n}/std/max"] = np.max(masked_std)
        #     delta = np.nan_to_num(delta)
        #     masked_std = np.nan_to_num(masked_std)

        #     writer.add_histogram(f"{prefix}{n}_delta_abs",
        #                          np.abs(delta), epoch_i)
        #     writer.add_histogram(f"{prefix}{n}_delta_abs_dB",
        #                          np.log10(np.abs(delta)+1e-6), epoch_i)

        #     writer.add_histogram(f"{n}_std",
        #                          masked_std, epoch_i)
        #     sorted_delta_abs = np.abs(delta)[np.argsort(masked_std)]

        #     for frac in [10, 50, 90]:
        #         res[f"{n}/sorted_delta_abs_{frac}"] = np.mean(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])
        #         res[f"{n}/sorted_delta_abs_{frac}_max"] = np.max(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])

        # exception = False

        # for metric_name, metric_val in res.items():
        #     #print(f"{metric_name} is {metric_val}")

        #     if not np.isfinite(metric_val):
        #         exception = True
        #         #print(f"{metric_name} is {metric_val}")
        #     writer.add_scalar("{}{}".format(prefix, metric_name),
        #                       metric_val, epoch_i)
        # if exception:
        #     raise ValueError(f"{prefix}{metric_name} found some nans")

    return val_func


def create_coupling_uncertain_validate_func(config, writer):
    coupling_type_list = config.get("coupling_lut", {})
    coupling_type_lut = {-1: "other"}
    for si, s in enumerate(coupling_type_list):
        coupling_type_lut[si] = f"{s[1]}J{s[0]}"
    coupling_index = config["coupling_index"]

    def val_func(input_res, prefix, epoch_i):
        # print("input_res.keys() =", input_res.keys())
        # print( input_res['pred_edge_pred'].shape)

        coupling_mu = input_res["pred_coupling_mu"][:, :, :, coupling_index]
        coupling_std = input_res["pred_coupling_std"][:, :, :, coupling_index]

        coupling_truth = input_res["pred_edge_pred"][:, :, :, coupling_index]
        coupling_mask = input_res["pred_edge_pred_mask"][:, :, :, coupling_index]

        BATCH_N, MAX_N, _ = coupling_mu.shape

        coupling_types = input_res["pred_passthrough_coupling_types_encoded"]

        delta = coupling_mu - coupling_truth

        delta_present = delta[coupling_mask > 0]
        std_present = coupling_std[coupling_mask > 0]

        delta_types = coupling_types[coupling_mask > 0]

        metrics = {
            "coupling_delta_abs": np.mean(np.abs(delta_present)),
            "coupling_delta_sq": np.mean(delta_present**2),
            "coupling_n": np.sum(coupling_mask),
        }

        # break errors into luts
        different_coupling_types_delta = {k: list() for k in coupling_type_lut.keys()}
        different_coupling_types_std = {k: list() for k in coupling_type_lut.keys()}
        for ct, delta, std in zip(delta_types, delta_present, std_present):
            different_coupling_types_delta[ct].append(np.abs(delta))
            different_coupling_types_std[ct].append(std)

        for k, v in coupling_type_lut.items():
            # print('adding metric', f"coupling_{v}_delta_abs")
            if len(different_coupling_types_delta[k]) > 0:
                deltas = np.array(different_coupling_types_delta[k])
                stds = np.array(different_coupling_types_std[k])

                base_metric = f"coupling_{v}"
                metrics[f"{base_metric}/delta_abs"] = np.mean(deltas)

                sorted_delta_abs = np.abs(deltas)[np.argsort(stds)]

                for frac in [10, 50, 90]:
                    metrics[f"{base_metric}/sorted_delta_abs_{frac}"] = np.mean(
                        sorted_delta_abs[: int(frac / 100.0 * len(sorted_delta_abs))]
                    )
                    metrics[f"{base_metric}/sorted_delta_abs_{frac}_max"] = np.max(
                        sorted_delta_abs[: int(frac / 100.0 * len(sorted_delta_abs))]
                    )

            else:
                pass

            # print("Warning, only",
            # len(different_coupling_types[k]),
            #          "entries for", k, v)
            # print("done")

        exception = False
        for metric_name, metric_val in metrics.items():
            if not np.isfinite(metric_val):
                exception = True
                print(f"{metric_name} is {metric_val}")

            writer.add_scalar("{}{}".format(prefix, metric_name), metric_val, epoch_i)
        if exception:
            raise ValueError(f"{prefix}{metric_name} found some nans")

        # mu = input_res['pred_mu']
        # std = input_res['pred_std']
        # pred_mask = input_res['pred_mask']
        # truth = input_res['pred_truth']
        # mean_loss = input_res['mean_loss']
        # #print("validate_func mu.shape=", mu.shape, "Truth.shape=", truth.shape)
        # res = {'mean_loss' : mean_loss,
        #        'run_epoch_time' : input_res['runtime'],
        #        'run_efficinecy' : input_res['run_efficiency'],
        #        'run_pts_per_sec' : input_res['pts_per_sec']}

        # # extra losses
        # for k, v in input_res.items():
        #     if 'loss_total_' in k:
        #         res[k] = v
        #     if 'loss_mean_' in k:
        #         res[k] = v

        # for ni, n in enumerate(tgt_nucs):
        #     delta = (mu[:, :, ni] - truth[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
        #     if len(delta) == 0:
        #         continue
        #     masked_std = (std[:, :, ni])[pred_mask[:, :, ni] > 0].flatten()
        #     res[f"{n}/delta_std"] = np.std(delta)
        #     res[f"{n}/delta_max"] = np.max(np.abs(delta))
        #     res[f"{n}/delta_mean_abs"] = np.mean(np.abs(delta))
        #     res[f"{n}/delta_abs_90"] = np.percentile(np.abs(delta), 90)
        #     res[f"{n}/std/mean"] = np.mean(masked_std)
        #     res[f"{n}/std/min"] = np.min(masked_std)
        #     res[f"{n}/std/max"] = np.max(masked_std)
        #     delta = np.nan_to_num(delta)
        #     masked_std = np.nan_to_num(masked_std)

        #     writer.add_histogram(f"{prefix}{n}_delta_abs",
        #                          np.abs(delta), epoch_i)
        #     writer.add_histogram(f"{prefix}{n}_delta_abs_dB",
        #                          np.log10(np.abs(delta)+1e-6), epoch_i)

        #     writer.add_histogram(f"{n}_std",
        #                          masked_std, epoch_i)
        #     sorted_delta_abs = np.abs(delta)[np.argsort(masked_std)]

        #     for frac in [10, 50, 90]:
        #         res[f"{n}/sorted_delta_abs_{frac}"] = np.mean(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])
        #         res[f"{n}/sorted_delta_abs_{frac}_max"] = np.max(sorted_delta_abs[:int(frac/100.0 * len(sorted_delta_abs))])

        # exception = False

        # for metric_name, metric_val in res.items():
        #     #print(f"{metric_name} is {metric_val}")

        #     if not np.isfinite(metric_val):
        #         exception = True
        #         #print(f"{metric_name} is {metric_val}")
        #     writer.add_scalar("{}{}".format(prefix, metric_name),
        #                       metric_val, epoch_i)
        # if exception:
        #     raise ValueError(f"{prefix}{metric_name} found some nans")

    return val_func


def create_optimizer(opt_params, net_params):
    opt_direct_params = {}
    optimizer_name = opt_params.get("optimizer", "adam")
    if optimizer_name == "adam":
        for p in ["lr", "amsgrad", "eps", "weight_decay", "momentum"]:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adam(net_params, **opt_direct_params)
    elif optimizer_name == "adamax":
        for p in ["lr", "eps", "weight_decay", "momentum"]:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adamax(net_params, **opt_direct_params)

    elif optimizer_name == "adagrad":
        for p in ["lr", "eps", "weight_decay", "momentum"]:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adagrad(net_params, **opt_direct_params)

    elif optimizer_name == "rmsprop":
        for p in ["lr", "eps", "weight_decay", "momentum"]:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.RMSprop(net_params, **opt_direct_params)

    elif optimizer_name == "sgd":
        for p in ["lr", "momentum"]:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.SGD(net_params, **opt_direct_params)

    return optimizer


def create_loss(loss_params, USE_CUDA):
    loss_name = loss_params["loss_name"]

    std_regularize = loss_params.get("std_regularize", 0.01)
    mu_scale = move(torch.Tensor(loss_params.get("mu_scale", [1.0])), USE_CUDA)
    std_scale = move(torch.Tensor(loss_params.get("std_scale", [1.0])), USE_CUDA)

    if loss_name == "NormUncertainLoss":
        criterion = nets.NormUncertainLoss(
            mu_scale, std_scale, std_regularize=std_regularize
        )
    elif loss_name == "UncertainLoss":
        criterion = nets.UncertainLoss(
            mu_scale,
            std_scale,
            norm=loss_params["norm"],
            std_regularize=std_regularize,
            std_pow=loss_params["std_pow"],
            use_reg_log=loss_params["use_reg_log"],
            std_weight=loss_params["std_weight"],
        )

    elif loss_name == "NoUncertainLoss":
        criterion = nets.NoUncertainLoss(**loss_params)
    elif loss_name == "SimpleLoss":
        criterion = nets.SimpleLoss(**loss_params)
    elif "EnsembleLoss" in loss_name:
        subloss = loss_name.split("-")[1]

        criterion = nets.EnsembleLoss(subloss, **loss_params)
    elif loss_name == "PermMinLoss":
        criterion = nets.PermMinLoss(**loss_params)
    elif loss_name == "ReconLoss":
        criterion = seminets.ReconLoss(**loss_params)
    elif loss_name == "CouplingLoss":
        criterion = coupling.CouplingLoss(**loss_params)
    elif loss_name == "DistReconLoss":
        criterion = seminets.DistReconLoss(**loss_params)
    else:
        raise ValueError(loss_name)

    return criterion
