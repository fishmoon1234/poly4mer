"""
train_poly4mer_v3_property_reg.py: joint training of pSMILES reconstruction + fire-property prediction
=====================================================================================================

Lives in training/; models.py, utils1.py, smi_ted_light/ and poly4mer_v3.ckpt are in the
repository root one level up. Run it from the repository root:
    python training/train_poly4mer_v3_property_reg.py ...

This is the joint-training script for the poly4mer v3 model (star encoder,
encoder, decoder, token predictor, and property heads).

By default it INITIALIZES every network, including the property heads
and their target standardization, from poly4mer_v3.ckpt (in this folder).
Pass --fresh_heads to start the property heads from random weights instead.
See README.md.

Model (one shared latent z per polymer)
---------------------------------------
smi-ted-Light (IBM) is used as a FROZEN tokenizer + token embedder.

    pSMILES string
      -> smi-ted (frozen)          idx  [B, 202]        token ids (vocab 2393, pad to 202)
                                   emb  [B, 202, 768]   token embeddings
      -> star_encoder              replaces the embedding of every '*' token (id 439)
                                   with a learned vector: Linear(1,256)-ReLU-Linear(256,768)
      -> autoencoder.encoder       flatten(202*768) -> z  [B, 768]          <- the latent
      -> decoder                   z -> emb_rec [B, 202, 768]
      -> predictor (token classifier, `prediction_Model`)
                                   emb_rec -> logits [B*202, 2393]
      -> property heads (only used for the property loss):
            tig, pkhrr : MLP([z, thickness_norm, flux_norm])   in_dim 770
            sea, co    : MLP(z)                                in_dim 768

Loss (per optimizer step, "joint" mode)
---------------------------------------
Each step draws one batch of SMILES-only data (v1) and one batch of labeled
data (v2 FDS-simulated + experimental). With CE = token cross-entropy and
MSE_emb = MSE between input and reconstructed token embeddings:

    loss =  (CE_unlabeled + CE_labeled) / 2                      # reconstruction, cross-entropy
          + lamb          * (MSE_emb_unlabeled + MSE_emb_labeled) / 2   # reconstruction, embedding MSE
          + property_lamb * sum_p MSE(pred_p, y_p)               # property prediction

Property targets are standardized with the training-set mean/std. Missing
labels (NaN) are masked per property. The labeled loader is cycled because it
is much smaller than the SMILES-only loader.

Trainable: star_encoder, autoencoder.encoder, decoder, predictor, 4 property heads.
Frozen:    smi-ted.

Data formats
------------
--train_smiles / --val_data : .smi (first whitespace token per line) or .csv
                              (column smiles_canonicalized / smiles).
--train_labeled / --val_labeled :
    * .smi without header, 10 columns:
          index name smiles thickness flux tig pkhrr sea co igt   (v2 FDS format)
    * .smi with header line, or .csv, containing the columns
          smiles_canonicalized thickness flux tig pkhrr sea co
Units: thickness mm, flux kW/m^2, tig s, pkhrr kW/m^2, co kg/kg; sea as in the source data.

Dependencies
------------
torch, numpy, pandas, tqdm, transformers, rdkit, regex, plus (in this folder):
  models.py  : Decoder2, prediction_Model, star_encoder, AutoEncoderLayer3
  utils1.py  : load_smi_ted_explicit / find_smi_ted_dir
  smi_ted_light/ : smi-ted-Light_40.pt, bert_vocab_curated.txt, load.py
Tested with torch 2.8.0+cu128 on an RTX 5090.

Checkpoints
-----------
After each epoch the script evaluates reconstruction on --val_data and, if the
sequence accuracy is >= the best so far, saves
    <checkpoint_dir>/model_params<N>_lamb<lamb>_best.ckpt
with keys: star_encoder, autoencoder_encoder, decoder, predictor,
property_regressors, property_stats, optimizer, epoch, best_val_acc.
That file is ~12 GB because it includes the Adam state. poly4mer_v3.ckpt has
the same network keys plus `meta`, but no optimizer state, so it can initialize
a new run (--init_recon_ckpt, the default) but not --resume one.

Loading poly4mer_v3.ckpt
------------------------
    ck = torch.load("poly4mer_v3.ckpt",
                    map_location="cpu", weights_only=False)
    A   = star_encoder(dim=1);                                   A.load_state_dict(ck["star_encoder"])
    ae  = AutoEncoderLayer3(202*768, 768*4, 768*2, 768);         ae.encoder.load_state_dict(ck["autoencoder_encoder"])
    dec = Decoder2(202*768, 768*4, 768*2, 768);                  dec.load_state_dict(ck["decoder"])
    tok = prediction_Model(n_embd=768, mid_size=768*4, n_vocab=2393); tok.load_state_dict(ck["predictor"])
    # z = ae.encoder(token embeddings with '*' replaced by A(...)), see forward_pass() below.
    heads = {n: build_regressor(770 if PROP_USES_THK_FLUX[n] else 768) for n in PROPERTY_NAMES}
    for n in PROPERTY_NAMES: heads[n].load_state_dict(ck["property_regressors"][n])
    st = ck["property_stats"]   # thk/flux mean+std, y_mean/y_std in order tig, pkhrr, sea, co
    # tig/pkhrr input: [z, (thk-thk_mean)/thk_std, (flux-flux_mean)/flux_std]; sea/co input: z
    # raw property = head(x) * y_std[k] + y_mean[k]
"""

import argparse
import os
import sys
from timeit import default_timer

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../training
REPO_DIR = os.path.dirname(SCRIPT_DIR)                           # repository root: models.py, utils1.py, smi_ted_light/
for _p in (REPO_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import prediction_Model, Decoder2, star_encoder, AutoEncoderLayer3
from utils1 import load_smi_ted_explicit, find_smi_ted_dir


# =============================================================================
# Constants
# =============================================================================

MAX_LEN = 202          # smi-ted sequence length (tokens, padded)
EMB_DIM = 768          # smi-ted token-embedding size = latent size
VOCAB_SIZE = 2393      # smi-ted vocabulary size
STAR_TOKEN_ID = 439    # id of '*' (polymer attachment point) in the smi-ted vocab

PROPERTY_NAMES = ["tig", "pkhrr", "sea", "co"]
# tig and pkhrr depend on sample thickness and heat flux; sea and co do not.
PROP_USES_THK_FLUX = {"tig": True, "pkhrr": True, "sea": False, "co": False}

# Column layout of v2_110k_FDS.smi-style files (10 columns, no header).
V2_COLUMNS = ["index", "name", "smiles_canonicalized", "thickness", "flux",
              "tig", "pkhrr", "sea", "co", "igt"]

# File names of optional pretrained property-head files (read only if
# --init_heads_dir points to a directory holding them).
PROP_PRETRAINED_FNAMES = {
    "tig":   "tig_model_params821761_mean.ckpt",
    "pkhrr": "pkhrr_model_params821761_mean.ckpt",
    "sea":   "Ysmk_model_params821761_mean.ckpt",
    "co":    "Yco_model_params820737_mean.ckpt",
}


# =============================================================================
# Small utilities
# =============================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def lr_at_epoch(base_lr, epoch, step, gamma):
    """Step decay: base_lr * gamma ** (epoch // step). A huge `step` gives a constant lr."""
    return base_lr * np.power(gamma, epoch // step)


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


def parse_paths(arg):
    """'a.smi,b.smi' -> ['a.smi', 'b.smi']; None -> []."""
    if arg is None:
        return []
    return [p.strip() for p in arg.split(",") if p.strip()]


# =============================================================================
# Data
# =============================================================================

def read_smiles_file(path):
    """SMILES list from a .csv (smiles column) or a .smi (first token of each line)."""
    if os.path.splitext(path)[1].lower() == ".csv":
        df = pd.read_csv(path)
        for col in ("smiles_canonicalized", "smiles", "SMILES"):
            if col in df.columns:
                return df[col].astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()
    smiles = []
    with open(path) as f:
        for line in f:
            tokens = line.split()
            if tokens:
                smiles.append(tokens[0])
    return smiles


def read_property_data(path):
    """Labeled table with columns smiles_canonicalized, thickness, flux, tig, pkhrr, sea, co.

    Rows without SMILES/thickness/flux are dropped, as are rows where all four
    properties are missing. Individual missing properties stay NaN and are
    masked in the loss.
    """
    if os.path.splitext(path)[1].lower() == ".csv":
        df = pd.read_csv(path)
    else:
        with open(path) as f:
            first = next((ln.split() for ln in f if ln.strip()), [])
        if "smiles_canonicalized" in first:      # .smi with a header line
            df = pd.read_csv(path, sep=r"\s+", engine="python")
        else:                                    # v2 FDS format, no header
            df = pd.read_csv(path, sep=r"\s+", header=None, engine="python", names=V2_COLUMNS)

    required = ["smiles_canonicalized", "thickness", "flux"] + PROPERTY_NAMES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Required: {required}")
    df = df[required].copy()
    for c in ["thickness", "flux"] + PROPERTY_NAMES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["smiles_canonicalized", "thickness", "flux"])
    df = df[df[PROPERTY_NAMES].notna().any(axis=1)].reset_index(drop=True)
    return df


def load_smiles(paths):
    out = []
    for p in paths:
        s = read_smiles_file(p)
        print(f"   + {p}: {len(s)} SMILES")
        out.extend(s)
    return out


def load_labeled(paths):
    dfs = []
    for p in paths:
        d = read_property_data(p)
        print(f"   + {p}: {len(d)} labeled rows")
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


class SmilesDataset(Dataset):
    """Yields raw SMILES strings (tokenization happens inside smi-ted)."""

    def __init__(self, smiles):
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles[i]


class PropertyDataset(Dataset):
    """Yields (smiles, thickness, flux, tig, pkhrr, sea, co). NaN labels are kept."""

    def __init__(self, df):
        self.smiles = df["smiles_canonicalized"].astype(str).tolist()
        self.thickness = df["thickness"].astype(np.float32).values
        self.flux = df["flux"].astype(np.float32).values
        self.y = df[PROPERTY_NAMES].astype(np.float32).values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return (self.smiles[i], float(self.thickness[i]), float(self.flux[i]),
                float(self.y[i, 0]), float(self.y[i, 1]), float(self.y[i, 2]), float(self.y[i, 3]))


def labeled_batch_to_tensors(batch, device):
    """Unpack a PropertyDataset batch -> (smiles list, thickness [B], flux [B], y [B, 4])."""
    smi, thk, flux, tig, pkhrr, sea, co = batch
    y = torch.stack([t.to(device, dtype=torch.float32) for t in (tig, pkhrr, sea, co)], dim=-1)
    return (list(smi), thk.to(device, dtype=torch.float32),
            flux.to(device, dtype=torch.float32), y)


def compute_property_stats(df, device):
    """Mean/std of thickness, flux and the four targets (NaN-aware) for standardization."""
    y_mean = df[PROPERTY_NAMES].mean().values.astype(np.float32)
    y_std = np.maximum(df[PROPERTY_NAMES].std().values.astype(np.float32), 1e-6)
    return {
        "thk_mean":  torch.tensor(float(df["thickness"].mean()), device=device),
        "thk_std":   torch.tensor(max(float(df["thickness"].std()), 1e-6), device=device),
        "flux_mean": torch.tensor(float(df["flux"].mean()), device=device),
        "flux_std":  torch.tensor(max(float(df["flux"].std()), 1e-6), device=device),
        "y_mean":    torch.tensor(y_mean, device=device),
        "y_std":     torch.tensor(y_std, device=device),
    }


# =============================================================================
# Model pieces
# =============================================================================

def build_regressor(input_dim):
    """Property head: 5-layer MLP with GELU, scalar output (standardized target)."""
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.GELU(),
        nn.Linear(512, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, 128), nn.GELU(),
        nn.Linear(128, 1),
    )


def forward_pass(smiles_batch, smi_ted, A_encoder, autoencoder, decoder, predictor, device):
    """SMILES -> (token ids, input embeddings, reconstructed embeddings, logits, z)."""
    # 1) Frozen smi-ted: token ids and per-token embeddings.
    with torch.no_grad():
        idx, emb, _ = smi_ted.extract_embeddings(smiles_batch)
    idx = idx.to(device)
    emb = emb.to(device).detach().clone()

    # 2) Replace the embedding of every '*' token with the learned star embedding,
    #    so gradients reach star_encoder through these positions.
    pos = torch.nonzero(idx == STAR_TOKEN_ID, as_tuple=False)
    if pos.numel() > 0:
        star_in = idx[pos[:, 0], pos[:, 1]].view(-1, 1).float()
        emb[pos[:, 0], pos[:, 1], :] = A_encoder(star_in)

    # 3) Encode to the 768-d latent, decode back to token embeddings, classify tokens.
    z = autoencoder.encoder(emb.view(-1, MAX_LEN * EMB_DIM))            # [B, 768]
    emb_rec = decoder(z).view(-1, MAX_LEN, EMB_DIM)                       # [B, 202, 768]
    logits = predictor(emb_rec.view(-1, EMB_DIM))                         # [B*202, 2393]
    return idx, emb, emb_rec, logits, z


def recon_losses(idx, emb, emb_rec, logits):
    """Token cross-entropy and embedding MSE for one batch."""
    ce = F.nll_loss(F.log_softmax(logits, dim=1), idx.view(-1).long())
    mse = F.mse_loss(emb_rec, emb)
    return ce, mse


def property_loss(z, thickness, flux, y, regressors, stats):
    """Sum over properties of MSE in standardized space, masking NaN labels per property."""
    thk_n = (thickness - stats["thk_mean"]) / stats["thk_std"]
    flux_n = (flux - stats["flux_mean"]) / stats["flux_std"]
    x770 = torch.cat([z, thk_n.view(-1, 1), flux_n.view(-1, 1)], dim=-1)

    total = z.new_zeros(())
    for k, name in enumerate(PROPERTY_NAMES):
        x = x770 if PROP_USES_THK_FLUX[name] else z
        pred = regressors[name](x).view(-1)
        target = (y[:, k] - stats["y_mean"][k]) / stats["y_std"][k]
        mask = ~torch.isnan(target)
        if mask.any():
            total = total + F.mse_loss(pred[mask], target[mask])
    return total


# =============================================================================
# Initialization and resuming
# =============================================================================

def init_from_pretrained(recon_ckpt, heads_dir, decoder, predictor, autoencoder, A_encoder,
                         regressors, device, fresh_heads=False):
    """Load network weights from a checkpoint such as poly4mer_v3.ckpt.

    `recon_ckpt` must have the keys autoencoder_encoder, decoder, predictor and
    optionally star_encoder, property_regressors and property_stats.

    Property heads, in order of preference:
      1. `property_regressors` from `recon_ckpt` (unless fresh_heads=True). Their
         `property_stats` are returned, because the heads only make sense with the
         target standardization they were trained with.
      2. Pretrained head files in `heads_dir`. A head
         with a wider input than the current one (sea: 770 -> 768) has its extra
         first-layer input columns dropped.
      3. Random weights.
    Returns the checkpoint's property_stats when its heads were loaded, else None.
    """
    ck = None
    if recon_ckpt and os.path.isfile(recon_ckpt):
        print(f">> Init encoder/decoder/predictor from {recon_ckpt}")
        ck = torch.load(recon_ckpt, map_location=device, weights_only=False)
        decoder.load_state_dict(ck["decoder"])
        predictor.load_state_dict(ck["predictor"])
        autoencoder.encoder.load_state_dict(ck["autoencoder_encoder"])
        if "star_encoder" in ck:
            A_encoder.load_state_dict(ck["star_encoder"])
    else:
        print(f">> No reconstruction init found at {recon_ckpt!r}; random init.")

    if ck is not None and "property_regressors" in ck and not fresh_heads:
        for name in PROPERTY_NAMES:
            regressors[name].load_state_dict(ck["property_regressors"][name])
        print(f">> Init property heads (tig, pkhrr, sea, co) from {recon_ckpt}")
        return {k: torch.as_tensor(v, device=device) for k, v in ck["property_stats"].items()}

    if not heads_dir or not os.path.isdir(heads_dir):
        print(">> Property heads: random init.")
        return None
    for name, fname in PROP_PRETRAINED_FNAMES.items():
        path = os.path.join(heads_dir, fname)
        if not os.path.isfile(path):
            print(f">> [{name}] {path} missing; random init.")
            continue
        sd = torch.load(path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        cur_in = regressors[name].state_dict()["0.weight"].shape[1]
        if sd["0.weight"].shape[1] > cur_in:
            sd = dict(sd)
            sd["0.weight"] = sd["0.weight"][:, :cur_in].clone()
            print(f">> [{name}] truncated first-layer input to {cur_in}")
        regressors[name].load_state_dict(sd)
        print(f">> Init {name} head from {path}")
    return None


def find_latest_best(ckpt_dir):
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") and "best" in f]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)), reverse=True)
    return os.path.join(ckpt_dir, files[0]) if files else None


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(loader, n_total, smi_ted, A_encoder, autoencoder, decoder, predictor, device):
    """Mean CE, mean embedding MSE, exact-sequence accuracy and token accuracy."""
    for m in (A_encoder, autoencoder.encoder, decoder, predictor):
        m.eval()
    ce_sum = mse_sum = 0.0
    seq_ok = tok_ok = tok_n = 0
    for smiles in loader:
        idx, emb, emb_rec, logits, _ = forward_pass(smiles, smi_ted, A_encoder, autoencoder,
                                                    decoder, predictor, device)
        ce, mse = recon_losses(idx, emb, emb_rec, logits)
        ce_sum += ce.item() * idx.shape[0]
        mse_sum += mse.item() * idx.shape[0]
        pred = logits.view(-1, MAX_LEN, VOCAB_SIZE).argmax(-1)
        seq_ok += int(torch.all(pred == idx, dim=1).sum())
        tok_ok += int((pred == idx).sum())
        tok_n += idx.numel()
    n = max(n_total, 1)
    return {"ce": ce_sum / n, "mse": mse_sum / n, "seq_acc": seq_ok / n, "tok_acc": tok_ok / max(tok_n, 1)}


@torch.no_grad()
def evaluate_properties(loader, smi_ted, A_encoder, autoencoder, decoder, predictor,
                        regressors, stats, device):
    """Mean relative error |pred - y| / |y| per property (rows with non-NaN, non-zero y)."""
    for m in (A_encoder, autoencoder.encoder, decoder, predictor, *regressors.values()):
        m.eval()
    err = {k: 0.0 for k in PROPERTY_NAMES}
    cnt = {k: 0 for k in PROPERTY_NAMES}
    for batch in loader:
        smi, thk, flux, y = labeled_batch_to_tensors(batch, device)
        *_, z = forward_pass(smi, smi_ted, A_encoder, autoencoder, decoder, predictor, device)
        thk_n = (thk - stats["thk_mean"]) / stats["thk_std"]
        flux_n = (flux - stats["flux_mean"]) / stats["flux_std"]
        x770 = torch.cat([z, thk_n.view(-1, 1), flux_n.view(-1, 1)], dim=-1)
        for k, name in enumerate(PROPERTY_NAMES):
            x = x770 if PROP_USES_THK_FLUX[name] else z
            pred = regressors[name](x).view(-1) * stats["y_std"][k] + stats["y_mean"][k]
            mask = (~torch.isnan(y[:, k])) & (y[:, k].abs() > 1e-12)
            if mask.any():
                err[name] += float(((pred[mask] - y[mask, k]).abs() / y[mask, k].abs()).sum())
                cnt[name] += int(mask.sum())
    return {k: (err[k] / cnt[k] if cnt[k] else float("nan")) for k in PROPERTY_NAMES}, cnt


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="poly4mer joint pSMILES reconstruction + property training.")
    ap.add_argument("--train_smiles", required=True,
                    help="Comma-separated SMILES-only files (reconstruction loss only).")
    ap.add_argument("--train_labeled", required=True,
                    help="Comma-separated labeled files (reconstruction + property loss).")
    ap.add_argument("--val_data", required=True,
                    help="Comma-separated SMILES-only files; their seq_acc selects the best checkpoint.")
    ap.add_argument("--val_labeled", default=None,
                    help="Optional labeled files; per-property relative error is printed (not used for selection).")
    ap.add_argument("--init_recon_ckpt", default=os.path.join(REPO_DIR, "poly4mer_v3.ckpt"),
                    help="Initial weights for all networks incl. property heads (default: poly4mer_v3.ckpt).")
    ap.add_argument("--fresh_heads", action="store_true",
                    help="Do not load property heads from --init_recon_ckpt; start them random and "
                         "standardize targets with statistics of your labeled data.")
    ap.add_argument("--init_heads_dir", default=None,
                    help="Directory with pretrained head files; used only if "
                         "the heads are not loaded from --init_recon_ckpt.")
    ap.add_argument("--no_init", action="store_true", help="Ignore all initialization; start from random weights.")
    ap.add_argument("--checkpoint_dir", required=True, help="Where best checkpoints and loss history are written.")
    ap.add_argument("--checkpoint_path", default=None,
                    help="Checkpoint to resume from (with --resume). Default: newest *best*.ckpt in checkpoint_dir.")
    ap.add_argument("--resume", action="store_true",
                    help="Restore weights, Adam state, epoch counter and property stats from a checkpoint.")
    ap.add_argument("--lr", type=float, default=1e-5, help="Adam learning rate.")
    ap.add_argument("--scheduler_step", type=int, default=1000,
                    help="Decay lr every this many epochs (default 1000 = constant lr).")
    ap.add_argument("--scheduler_gamma", type=float, default=0.7)
    ap.add_argument("--wd", type=float, default=1e-5, help="Adam weight decay.")
    ap.add_argument("--lamb", type=float, default=1.0, help="Weight of the embedding-MSE reconstruction loss.")
    ap.add_argument("--property_lamb", type=float, default=0.1, help="Weight of the property loss.")
    ap.add_argument("--batch_size", type=int, default=48, help="SMILES-only and validation batch size.")
    ap.add_argument("--labeled_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10,
                    help="Train until this 0-based epoch index (exclusive). With --resume, training "
                         "continues from the checkpoint's epoch + 1 up to this value.")
    ap.add_argument("--seed", type=int, default=1995)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- networks ---------------------------------------------------------
    A_encoder = star_encoder(dim=1).to(device)
    autoencoder = AutoEncoderLayer3(feature_size=MAX_LEN * EMB_DIM, mid_size=EMB_DIM * 4,
                                    mid_size_2=EMB_DIM * 2, latent_size=EMB_DIM).to(device)  # only .encoder is used
    decoder = Decoder2(feature_size=MAX_LEN * EMB_DIM, mid_size=EMB_DIM * 4,
                       mid_size_2=EMB_DIM * 2, latent_size=EMB_DIM).to(device)
    predictor = prediction_Model(n_embd=EMB_DIM, mid_size=EMB_DIM * 4, n_vocab=VOCAB_SIZE).to(device)
    regressors = {n: build_regressor(770 if PROP_USES_THK_FLUX[n] else 768).to(device) for n in PROPERTY_NAMES}
    print(f">> params: star_encoder {count_params(A_encoder):,}  encoder {count_params(autoencoder.encoder):,}  "
          f"decoder {count_params(decoder):,}  predictor {count_params(predictor):,}  "
          f"heads {sum(count_params(r) for r in regressors.values()):,}")

    # Frozen smi-ted tokenizer/embedder.
    SmiTed, Tokenizer = load_smi_ted_explicit()
    smi_ted_dir = find_smi_ted_dir()
    smi_ted = SmiTed(Tokenizer(os.path.join(smi_ted_dir, "bert_vocab_curated.txt")))
    smi_ted.load_checkpoint(os.path.join(smi_ted_dir, "smi-ted-Light_40.pt"))
    smi_ted.eval()
    for p in smi_ted.parameters():
        p.requires_grad = False

    # ---- data ---------------------------------------------------------------
    print(">> SMILES-only training data")
    train_smiles = load_smiles(parse_paths(args.train_smiles))
    unlabeled_loader = DataLoader(SmilesDataset(train_smiles), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    print(">> Labeled training data")
    labeled_df = load_labeled(parse_paths(args.train_labeled))
    stats = compute_property_stats(labeled_df, device)
    labeled_loader = DataLoader(PropertyDataset(labeled_df), batch_size=args.labeled_batch_size,
                                shuffle=True, num_workers=args.num_workers)

    print(">> Validation data")
    val_smiles = load_smiles(parse_paths(args.val_data))
    val_loader = DataLoader(SmilesDataset(val_smiles), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    val_labeled_loader = None
    if args.val_labeled:
        val_labeled_loader = DataLoader(PropertyDataset(load_labeled(parse_paths(args.val_labeled))),
                                        batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)

    # ---- initialization / optimizer / resume ------------------------------
    if not args.no_init:
        ck_stats = init_from_pretrained(args.init_recon_ckpt, args.init_heads_dir, decoder, predictor,
                                        autoencoder, A_encoder, regressors, device,
                                        fresh_heads=args.fresh_heads)
        if ck_stats is not None:
            # Heads came from the checkpoint: keep the standardization they were trained with.
            stats = ck_stats
            print(">> Using property_stats from the checkpoint (thickness, flux, target mean/std).")

    modules = [A_encoder, autoencoder.encoder, decoder, predictor, *regressors.values()]
    params = [p for m in modules for p in m.parameters()]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    start_epoch, best_val_acc = 0, 0.0
    if args.resume:
        path = args.checkpoint_path or find_latest_best(args.checkpoint_dir)
        if path and os.path.isfile(path):
            print(f">> Resuming from {path}")
            ck = torch.load(path, map_location=device, weights_only=False)
            A_encoder.load_state_dict(ck["star_encoder"])
            autoencoder.encoder.load_state_dict(ck["autoencoder_encoder"])
            decoder.load_state_dict(ck["decoder"])
            predictor.load_state_dict(ck["predictor"])
            for n in PROPERTY_NAMES:
                regressors[n].load_state_dict(ck["property_regressors"][n])
            # Keep the standardization of the original run, so targets keep the same scale.
            stats = {k: torch.as_tensor(v, device=device) for k, v in ck["property_stats"].items()}
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = int(ck["epoch"]) + 1
            best_val_acc = float(ck.get("best_val_acc", 0.0))
            del ck
            print(f"   start_epoch {start_epoch}, best val seq_acc {best_val_acc * 100:.2f}%")
        else:
            print(">> Nothing to resume from; starting fresh.")

    total_params = sum(count_params(m) for m in modules)
    best_path = os.path.join(args.checkpoint_dir, f"model_params{total_params}_lamb{args.lamb}_best.ckpt")
    hist_path = os.path.join(args.checkpoint_dir,
                             f"losses_lr{args.lr:.6f}_gamma{args.scheduler_gamma:.1f}_wd{args.wd:.6f}_lamb{args.lamb}.txt")
    history = []

    # ---- training loop ----------------------------------------------------
    for ep in range(start_epoch, args.epochs):
        t0 = default_timer()
        set_lr(optimizer, lr_at_epoch(args.lr, ep, args.scheduler_step, args.scheduler_gamma))
        for m in modules:
            m.train()
        labeled_iter = iter(labeled_loader)

        pbar = tqdm(unlabeled_loader, desc=f"Epoch {ep + 1}/{args.epochs}")
        for u_smiles in pbar:
            # (a) SMILES-only batch: reconstruction losses.
            idx_u, emb_u, rec_u, logits_u, _ = forward_pass(u_smiles, smi_ted, A_encoder, autoencoder,
                                                            decoder, predictor, device)
            ce_u, mse_u = recon_losses(idx_u, emb_u, rec_u, logits_u)

            # (b) Labeled batch (cycled): reconstruction + property losses.
            try:
                l_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                l_batch = next(labeled_iter)
            l_smiles, thk, flux, y = labeled_batch_to_tensors(l_batch, device)
            idx_l, emb_l, rec_l, logits_l, z_l = forward_pass(l_smiles, smi_ted, A_encoder, autoencoder,
                                                              decoder, predictor, device)
            ce_l, mse_l = recon_losses(idx_l, emb_l, rec_l, logits_l)
            prop = property_loss(z_l, thk, flux, y, regressors, stats)

            # (c) Joint objective.
            loss = ((ce_u + ce_l) / 2.0
                    + args.lamb * (mse_u + mse_l) / 2.0
                    + args.property_lamb * prop)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---- validation and checkpointing ---------------------------------
        m = evaluate_reconstruction(val_loader, len(val_smiles), smi_ted, A_encoder, autoencoder,
                                    decoder, predictor, device)
        if m["seq_acc"] >= best_val_acc:
            best_val_acc = m["seq_acc"]
            torch.save({
                "star_encoder":        A_encoder.state_dict(),
                "autoencoder_encoder": autoencoder.encoder.state_dict(),
                "decoder":             decoder.state_dict(),
                "predictor":           predictor.state_dict(),
                "property_regressors": {n: regressors[n].state_dict() for n in PROPERTY_NAMES},
                "property_stats":      {k: v.detach().cpu() for k, v in stats.items()},
                "optimizer":           optimizer.state_dict(),
                "epoch":               ep,
                "best_val_acc":        best_val_acc,
                "best_train_acc":      0.0,
            }, best_path)
            print(f">> Saved best to {best_path}")

        history.append([m["mse"], m["ce"], m["seq_acc"], best_val_acc])
        np.savetxt(hist_path, history, delimiter=",")
        print(f"Epoch {ep}, {default_timer() - t0:.1f}s | val pred_loss {m['ce']:.6f}, "
              f"rec_loss {m['mse']:.6f}, seq_acc {m['seq_acc'] * 100:.2f}%, "
              f"tok_acc {m['tok_acc'] * 100:.2f}%, best {best_val_acc * 100:.2f}%")

        if val_labeled_loader is not None:
            rel, n = evaluate_properties(val_labeled_loader, smi_ted, A_encoder, autoencoder, decoder,
                                         predictor, regressors, stats, device)
            print("   val property rel_err: " + "  ".join(
                f"{k}={rel[k] * 100:6.2f}% (n={n[k]})" for k in PROPERTY_NAMES))
    print(">> End training.")


if __name__ == "__main__":
    main()
