from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Shifts13CItem(BaseModel):
    atom_idx: int
    pred_mu: float
    pred_std: float


class Prediction(BaseModel):
    smiles: str
    shifts_13C: List[Shifts13CItem]
    success: bool


class Meta(BaseModel):
    max_n: int
    to_pred: str
    model_checkpoint_filename: str
    model_meta_filename: str
    ts_start: str
    ts_end: str
    runtime_sec: float
    git_commit: str
    rate_mol_sec: float
    num_mol: int
    cuda_attempted: bool
    use_cuda: bool


class ResPredictModel(BaseModel):
    predictions: List[Prediction]
    meta: Meta
