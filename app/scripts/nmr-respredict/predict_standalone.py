from rdkit import Chem
import numpy as np
from datetime import datetime
import click
import pickle
import pandas as pd
import time
import json
import sys
import util
import netutil
import warnings
import torch
from urllib.parse import urlparse
import io
import gzip
import predwrap
import os

warnings.filterwarnings("ignore")

nuc_to_atomicno = {"13C": 6, "1H": 1}


def predict_mols(
    raw_mols,
    predictor,
    MAX_N,
    to_pred=None,
    add_h=True,
    sanitize=True,
    add_default_conf=True,
    num_workers=0,
):
    t1 = time.time()
    if add_h:
        mols = [Chem.AddHs(m) for m in raw_mols]
    else:
        mols = [Chem.Mol(m) for m in raw_mols]  # copy

    if sanitize:
        [Chem.SanitizeMol(m) for m in mols]

    # sanity check
    for m in mols:
        if m.GetNumAtoms() > MAX_N:
            raise ValueError("molecule has too many atoms")

        if len(m.GetConformers()) == 0 and add_default_conf:
            print("adding default conf")
            util.add_empty_conf(m)

    if to_pred in ["13C", "1H"]:
        pred_fields = ["pred_shift_mu", "pred_shift_std"]
    else:
        raise ValueError(f"Don't know how to predict {to_pred}")

    pred_t1 = time.time()
    vert_result_df, edge_results_df = predictor.pred(
        [{"rdmol": m} for m in mols],
        pred_fields=pred_fields,
        BATCH_SIZE=256,
        num_workers=num_workers,
    )

    pred_t2 = time.time()
    # print("The prediction took {:3.2f} ms".format((pred_t2-pred_t1)*1000)),

    t2 = time.time()

    all_out_dict = []

    ### pred cleanup
    if to_pred in ["13C", "1H"]:
        shifts_df = pd.pivot_table(
            vert_result_df,
            index=["rec_idx", "atom_idx"],
            columns=["field"],
            values=["val"],
        ).reset_index()

        for rec_idx, mol_vert_result in shifts_df.groupby("rec_idx"):
            m = mols[rec_idx]
            out_dict = {"smiles": Chem.MolToSmiles(m)}

            # tgt_idx = [int(a.GetIdx()) for a in m.GetAtoms() if a.GetAtomicNum() == nuc_to_atomicno[to_pred]]

            # a = mol_vert_result.to_dict('records')
            out_shifts = []
            # for row_i, row in mol_vert_result.iterrows():
            for row in mol_vert_result.to_dict(
                "records"
            ):  # mol_vert_result.iterrows():
                atom_idx = int(row[("atom_idx", "")])
                if (
                    m.GetAtomWithIdx(atom_idx).GetAtomicNum()
                    == nuc_to_atomicno[to_pred]
                ):
                    out_shifts.append(
                        {
                            "atom_idx": atom_idx,
                            "pred_mu": row[("val", "pred_shift_mu")],
                            "pred_std": row[("val", "pred_shift_std")],
                        }
                    )

            out_dict[f"shifts_{to_pred}"] = out_shifts

            out_dict["success"] = True
            all_out_dict.append(out_dict)

    return all_out_dict


DEFAULT_FILES = {
    "13C": {
        "meta": "models/default_13C.meta",
        "checkpoint": "models/default_13C.checkpoint",
    },
    "1H": {
        "meta": "models/default_1H.meta",
        "checkpoint": "models/default_1H.checkpoint",
    },
}


def s3_split(url):
    o = urlparse(url)
    bucket = o.netloc
    key = o.path.lstrip("/")
    return bucket, key


@click.command()
@click.option(
    "--filename", help="filename of file to read, or stdin if unspecified", default=None
)
@click.option(
    "--format",
    help="file format (sdf, rdkit)",
    default="sdf",
    type=click.Choice(["rdkit", "sdf"], case_sensitive=False),
)
@click.option(
    "--pred",
    help="Nucleus (1H or 13C) or coupling (coupling)",
    default="13C",
    type=click.Choice(["1H", "13C", "coupling"], case_sensitive=True),
)
@click.option("--model_meta_filename")
@click.option("--model_checkpoint_filename")
@click.option(
    "--print_data",
    default=None,
    help="print the smiles/fingerprint of the data used for train or test",
)
@click.option("--output", default=None)
@click.option("--num_data_workers", default=0, type=click.INT)
@click.option("--cuda/--no-cuda", default=True)
@click.option("--version", default=False, is_flag=True)
@click.option(
    "--sanitize/--no-sanitize", help="sanitize the input molecules", default=True
)
@click.option("--addhs", help="Add Hs to the input molecules", default=False)
@click.option(
    "--skip-molecule-errors/--no-skip-molecule-errors",
    help="skip any errors",
    default=True,
)
def predict(
    filename,
    format,
    pred,
    model_meta_filename,
    model_checkpoint_filename,
    cuda=False,
    output=None,
    sanitize=True,
    addhs=True,
    print_data=None,
    version=False,
    skip_molecule_errors=True,
    num_data_workers=0,
):
    ts_start = time.time()
    if version:
        print(os.environ.get("GIT_COMMIT", ""))
        sys.exit(0)

    if model_meta_filename is None:
        # defaults
        model_meta_filename = DEFAULT_FILES[pred]["meta"]
        model_checkpoint_filename = DEFAULT_FILES[pred]["checkpoint"]

    if print_data is not None:
        data_info_filename = model_meta_filename.replace(
            ".meta", "." + print_data + ".json"
        )
        print(open(data_info_filename, "r").read())
        sys.exit(0)

    meta = pickle.load(open(model_meta_filename, "rb"))

    MAX_N = meta["max_n"]

    cuda_attempted = cuda
    if cuda and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available, running with CPU")
        cuda = False
    predictor = predwrap.PredModel(
        model_meta_filename,
        model_checkpoint_filename,
        cuda,
        override_pred_config={},
    )

    input_fileobj = None

    if filename is not None and filename.startswith("s3://"):
        import boto3

        bucket, key = s3_split(filename)
        s3 = boto3.client("s3")
        input_fileobj = io.BytesIO()
        s3.download_fileobj(bucket, key, input_fileobj)
        input_fileobj.seek(0)

    if format == "sdf":
        if filename is None:
            mol_supplier = Chem.ForwardSDMolSupplier(sys.stdin.buffer)
        elif input_fileobj is not None:
            mol_supplier = Chem.ForwardSDMolSupplier(input_fileobj)
        else:
            mol_supplier = Chem.SDMolSupplier(filename)
    elif format == "rdkit":
        if filename is None:
            bin_data = sys.stdin.buffer.read()
            mol_supplier = [Chem.Mol(m) for m in pickle.loads(bin_data)]
        elif input_fileobj is not None:
            mol_supplier = [Chem.Mol(m) for m in pickle.load(input_fileobj)]
        else:
            mol_supplier = [Chem.Mol(m) for m in pickle.load(open(filename, "rb"))]

    mols = list(mol_supplier)
    if len(mols) > 0:
        all_results = predict_mols(
            mols,
            predictor,
            MAX_N,
            pred,
            add_h=addhs,
            sanitize=sanitize,
            num_workers=num_data_workers,
        )
    else:
        all_results = []
    ts_end = time.time()
    output_dict = {
        "predictions": all_results,
        "meta": {
            "max_n": MAX_N,
            "to_pred": pred,
            "model_checkpoint_filename": model_checkpoint_filename,
            "model_meta_filename": model_meta_filename,
            "ts_start": datetime.fromtimestamp(ts_start).isoformat(),
            "ts_end": datetime.fromtimestamp(ts_end).isoformat(),
            "runtime_sec": ts_end - ts_start,
            "git_commit": os.environ.get("GIT_COMMIT", ""),
            "rate_mol_sec": len(all_results) / (ts_end - ts_start),
            "num_mol": len(all_results),
            "cuda_attempted": cuda_attempted,
            "use_cuda": cuda,
        },
    }
    json_str = json.dumps(output_dict, sort_keys=False, indent=4)
    if output is None:
        print(json_str)
    else:
        if output.startswith("s3://"):
            bucket, key = s3_split(output)
            s3 = boto3.client("s3")

            json_bytes = json_str.encode("utf-8")
            if key.endswith(".gz"):
                json_bytes = gzip.compress(json_bytes)

            output_fileobj = io.BytesIO(json_bytes)
            s3.upload_fileobj(output_fileobj, bucket, key)

        else:
            with open(output, "w") as fid:
                fid.write(json_str)


if __name__ == "__main__":
    predict()
