"""
Fire-property prediction from pSMILES via poly4mer v3.0.0 + Top-K SAE features.

Pipeline
  pSMILES --[smi-ted + star encoder + autoencoder.encoder]--> z in R^768
  z       --[frozen Top-K SAE encoder]-------------------->  c in R^{alpha*768}, k nonzeros
  c (+ test conditions) --[linear head]------------------->  fire property

Four properties, from three files:
  tig, pkhrr : data/experimental_flux_thickness.smi  (203 rows, 49 unique pSMILES)
  sea        : data/experimental_sea_avg.smi         (44 rows)
  co         : data/experimental_co_avg.smi          (32 rows)

tig and pkhrr are measured per (polymer, thickness, flux): the same pSMILES appears
on several rows with very different targets (tig 251 at flux 35 vs 86 at flux 50),
so thickness and flux are fed to the head alongside c. Without them one input has
several correct answers and that part of the error is irreducible -- compare with
--no_conditions. sea and co are already averaged per polymer and have no conditions.
--aggregate mean collapses repeated measurements into one row per pSMILES.

There is no train/validation split: every pSMILES is used for training, and the
errors are then measured in-distribution on those same rows. The headline relative
error is the relative L2 norm, ||y_pred - y_true||_2 / ||y_true||_2.

Example
  python predict_fire_properties.py --alpha 8 --k 160
  python predict_fire_properties.py --alpha 32 --k 1000 --solver ridge
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import AutoEncoderLayer3, TopKSAE, star_encoder
from utils1 import StringDataset, load_smi_ted_explicit

BASE = os.path.dirname(os.path.abspath(__file__))
STAR_TOKEN_ID = 439
SEQ_LEN, TOK_DIM = 202, 768

# property -> (file, target column, extra input columns)
SPECS = {
    "tig":   ("experimental_flux_thickness.smi", "tig",   ["thickness", "flux"]),
    "pkhrr": ("experimental_flux_thickness.smi", "pkhrr", ["thickness", "flux"]),
    "sea":   ("experimental_sea_avg.smi",        "sea",   []),
    "co":    ("experimental_co_avg.smi",         "co",    []),
}


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_embeddings(smiles_list, args, device):
    """Frozen CLLM pipeline -> Z in R^{M x 768}. Cached by content hash."""
    key = hashlib.md5("\n".join(smiles_list).encode()).hexdigest()[:10]
    cache = os.path.join(args.cache_dir, f"fire_psmiles_embeddings_{key}.pt")
    if os.path.exists(cache) and not args.force_recompute:
        print(f">> Load cached embeddings from {cache}")
        return torch.load(cache, map_location="cpu")

    print(">> Load pretrained smi-ted.")
    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    tokenizer = MolTranBertTokenizer(f"{BASE}/smi_ted_light/bert_vocab_curated.txt")
    smi_ted = Smi_ted(tokenizer)
    smi_ted.load_checkpoint(f"{BASE}/smi_ted_light/smi-ted-Light_40.pt")
    smi_ted.eval()

    print(f">> Load star encoder + autoencoder.encoder from {args.cllm_ckpt}")
    A_encoder = star_encoder(dim=1).to(device).eval()
    autoencoder = AutoEncoderLayer3(
        feature_size=SEQ_LEN * TOK_DIM, mid_size=TOK_DIM * 4,
        mid_size_2=TOK_DIM * 2, latent_size=TOK_DIM,
    ).to(device).eval()
    ckpt = torch.load(args.cllm_ckpt, map_location="cpu", weights_only=False)
    autoencoder.encoder.load_state_dict(ckpt["autoencoder_encoder"])
    A_encoder.load_state_dict(ckpt["star_encoder"])
    del ckpt

    zs = []
    loader = DataLoader(StringDataset(smiles_list), batch_size=args.batch_size, shuffle=False)
    for batch in loader:
        idx_t, tok_t, _ = smi_ted.extract_embeddings(batch)
        idx_t, tok_t = idx_t.to(device), tok_t.to(device)
        pos = torch.nonzero(idx_t == STAR_TOKEN_ID, as_tuple=False)
        if pos.numel() > 0:
            star = A_encoder(idx_t[pos[:, 0], pos[:, 1]].view(-1, 1).float())
            tok_t[pos[:, 0], pos[:, 1], :] = star
        zs.append(autoencoder.encoder(tok_t.view(-1, SEQ_LEN * TOK_DIM)).cpu())

    Z = torch.cat(zs, dim=0)
    os.makedirs(args.cache_dir, exist_ok=True)
    torch.save(Z, cache)
    print(f">> Cached embeddings {tuple(Z.shape)} to {cache}")
    return Z


@torch.no_grad()
def sae_encode(Z, args, device):
    """z -> sparse code c, using the frozen Top-K SAE for (alpha, k)."""
    path = args.sae_ckpt or os.path.join(
        BASE, "checkpoint", "topk_sae", f"topk_sae_alpha{args.alpha}_k{args.k}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No SAE checkpoint at {path}. Train one with\n"
            f"  python train_topk_sae.py --expansion_factor {args.alpha} --k {args.k}\n"
            f"or pass --sae_ckpt explicitly.")
    print(f">> Load SAE from {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg["expansion_factor"] != args.alpha or cfg["k"] != args.k:
        raise ValueError(f"checkpoint is alpha={cfg['expansion_factor']} k={cfg['k']}, "
                         f"asked for alpha={args.alpha} k={args.k}")
    sae = TopKSAE(d=cfg["d"], expansion_factor=cfg["expansion_factor"], k=cfg["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    print(f"   d={cfg['d']}  D={cfg['D']}  alpha={cfg['expansion_factor']}  k={cfg['k']}")

    C = torch.cat([sae.encode(z.to(device)).cpu()
                   for (z,) in DataLoader(torch.utils.data.TensorDataset(Z),
                                          batch_size=256, shuffle=False)], dim=0)
    nz = (C != 0).any(dim=0).sum().item()
    print(f">> Sparse codes C: {tuple(C.shape)}   features active on >=1 molecule: {nz}")
    return C.numpy()


# --------------------------------------------------------------------------- #
# head
# --------------------------------------------------------------------------- #
def fit_linear_head(Xtr, ytr, args, device, n_cond=0):
    """Linear head on the sparse code.

    The SAE code c is never scaled. The last n_cond columns are test conditions
    (thickness, flux for tig/pkhrr) and are standardized.

    Targets are log-transformed (all four properties are positive and span orders
    of magnitude) then standardized; predictions are mapped back before scoring."""
    n_code = Xtr.shape[1] - n_cond
    cond_mu = np.zeros(Xtr.shape[1]); cond_sd = np.ones(Xtr.shape[1])
    if n_cond:
        cond_mu[n_code:] = Xtr[:, n_code:].mean(0)
        cond_sd[n_code:] = Xtr[:, n_code:].std(0)
        cond_sd[cond_sd < 1e-8] = 1.0
    Xs = (Xtr - cond_mu) / cond_sd

    yt = np.log(ytr) if args.target_transform == "log" else ytr.copy()
    mu_y, sd_y = yt.mean(), yt.std()
    sd_y = sd_y if sd_y > 1e-8 else 1.0
    ys = (yt - mu_y) / sd_y

    if args.solver == "ridge":
        # Closed-form ridge with an unpenalized intercept. Centre X and y for the
        # solve only (no scaling), fit the slope, then recover the intercept.
        xbar, ybar = Xs.mean(0), ys.mean()
        Xc, yc = Xs - xbar, ys - ybar
        n, dim = Xc.shape
        lam = args.ridge_lambda
        if n < dim:
            # dual form: w = Xc' (Xc Xc' + lam I_n)^-1 yc   -- n x n instead of d x d
            G = Xc @ Xc.T + lam * np.eye(n)
            w = Xc.T @ np.linalg.solve(G, yc)
        else:
            A = Xc.T @ Xc + lam * np.eye(dim)
            w = np.linalg.solve(A, Xc.T @ yc)
        b = float(ybar - xbar @ w)
        state = (w, b)
    else:
        torch.manual_seed(args.seed)
        head = nn.Linear(Xs.shape[1], 1).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=device)
        Yt = torch.tensor(ys, dtype=torch.float32, device=device).view(-1, 1)
        lossf = nn.MSELoss()
        for ep in range(args.epochs):
            opt.zero_grad()
            loss = lossf(head(Xt), Yt)
            loss.backward()
            opt.step()
            if args.verbose and (ep + 1) % max(1, args.epochs // 5) == 0:
                print(f"      epoch {ep+1:5d}/{args.epochs}  mse(norm)={loss.item():.5f}")
        state = head

    return {"state": state, "cond_mu": cond_mu, "cond_sd": cond_sd,
            "mu_y": mu_y, "sd_y": sd_y, "transform": args.target_transform}


def predict(model, X, device):
    Xs = (X - model["cond_mu"]) / model["cond_sd"]   # identity on code columns
    if isinstance(model["state"], tuple):
        w, b = model["state"]
        ys = Xs @ w + b
    else:
        with torch.no_grad():
            ys = model["state"](
                torch.tensor(Xs, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    yt = ys * model["sd_y"] + model["mu_y"]
    return np.exp(yt) if model["transform"] == "log" else yt


def relative_error(y_true, y_pred):
    """Relative L2 error: ||y_pred - y_true||_2 / ||y_true||_2, as a percentage."""
    return float(np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true) * 100.0)


# --------------------------------------------------------------------------- #
# per-property driver
# --------------------------------------------------------------------------- #
def build_xy(prop, code_of, args):
    fname, ycol, cond = SPECS[prop]
    df = pd.read_csv(os.path.join(args.data_dir, fname), sep=r"\s+")
    df = df.dropna(subset=[ycol] + (cond if args.use_conditions else []))
    if args.aggregate == "mean":
        agg = {ycol: "mean"} | {c: "mean" for c in (cond if args.use_conditions else [])}
        df = df.groupby("smiles_canonicalized", as_index=False).agg(agg)
    C = np.stack([code_of[s] for s in df.smiles_canonicalized])
    if cond and args.use_conditions:
        extra = df[cond].to_numpy(dtype=np.float64)
        if args.condition_transform == "log":
            # with a log target, log conditions let the head fit power laws
            extra = np.log(extra)
        X = np.hstack([C, extra])
    else:
        X = C
    return df, X, df[ycol].to_numpy(dtype=np.float64)

def save_prediction_head(model, prop, feature_mask, args):
    save_dir = os.path.join(BASE, "checkpoint", "fire_heads")
    os.makedirs(save_dir, exist_ok=True)

    state = model["state"]
    if isinstance(state, tuple):
        # Closed-form ridge head
        w, b = state
        head = {
            "kind": "ridge",
            "weight": torch.as_tensor(w).cpu(),
            "bias": float(b),
        }
    else:
        # Adam-trained nn.Linear head
        head = {
            "kind": "linear",
            "in_features": state.in_features,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in state.state_dict().items()
            },
        }

    checkpoint = {
        "property": prop,
        "head": head,
        "feature_mask": torch.as_tensor(feature_mask, dtype=torch.bool),
        "cond_mu": torch.as_tensor(model["cond_mu"]),
        "cond_sd": torch.as_tensor(model["cond_sd"]),
        "mu_y": float(model["mu_y"]),
        "sd_y": float(model["sd_y"]),
        "transform": model["transform"],
        "condition_columns": SPECS[prop][2] if args.use_conditions else [],
        "condition_transform": args.condition_transform,
        "config": dict(vars(args)),
        "sae_ckpt": os.path.abspath(
            args.sae_ckpt or os.path.join(
                BASE, "checkpoint", "topk_sae",
                f"topk_sae_alpha{args.alpha}_k{args.k}.pt",
            )
        ),
    }

    path = os.path.join(
        save_dir,
        f"{prop}_alpha{args.alpha}_k{args.k}_{args.solver}.pt",
    )
    torch.save(checkpoint, path)
    print(f"  Saved prediction head: {path}")

def run_property(prop, code_of, keep, args, device):
    cond = SPECS[prop][2] if args.use_conditions else []
    df, X, y = build_xy(prop, code_of, args)
    X = X[:, keep[prop]] if prop in keep else X
    n_code = X.shape[1] - len(cond)
    print(f"\n=== {prop} ===")
    print(f"  rows={len(y)}  unique pSMILES={df.smiles_canonicalized.nunique()}  "
          f"features={X.shape[1]} ({n_code} code + {len(cond)} condition)  "
          f"target range=[{y.min():.4g}, {y.max():.4g}]")

    model = fit_linear_head(X, y, args, device, n_cond=len(cond))
    
    # Mask refers to the original input: full SAE code + condition columns.
    feature_mask = (
        keep[prop]
        if prop in keep
        else np.ones(X.shape[1], dtype=bool)
    )
    
    save_prediction_head(model, prop, feature_mask, args)
    
    pred = predict(model, X, device)
    mse = float(np.mean((pred - y) ** 2))
    mre = relative_error(y, pred)
    print(f"  MSE : {mse:.6g}    RMSE : {mse ** 0.5:.6g}    "
          f"relative error ||d||/||y|| : {mre:.2f}%")

    out = df.copy()
    out["predicted"] = pred
    out["relative_error_pct"] = np.abs(pred - y) / np.abs(y) * 100.0
    return mse, mre, out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alpha", type=int, default=8,
                   help="SAE expansion factor (D = alpha*768). Trained: 4, 8, 16, 32")
    p.add_argument("--k", type=int, default=160,
                   help="SAE sparsity. Trained: 80, 100, 160, 200, 250, 500, 1000")
    p.add_argument("--sae_ckpt", default=None, help="override the alpha/k checkpoint path")
    p.add_argument("--cllm_ckpt", default=f"{BASE}/checkpoint/poly4mer_v3.0.0.ckpt")
    p.add_argument("--data_dir", default=f"{BASE}/data")
    p.add_argument("--cache_dir", default=f"{BASE}/checkpoint/sae_cache")
    p.add_argument("--properties", nargs="+", default=list(SPECS), choices=list(SPECS))
    p.add_argument("--solver", choices=["sgd", "ridge"], default="sgd",
                   help="sgd: Adam-trained nn.Linear; ridge: closed form")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--ridge_lambda", type=float, default=10.0)
    p.add_argument("--target_transform", choices=["log", "none"], default="log")
    p.add_argument("--condition_transform", choices=["log", "none"], default="log",
                   help="log: head fits power laws tig ~ flux^b thickness^c; "
                        "none: raw values. Standardized either way.")
    p.add_argument("--no_conditions", dest="use_conditions", action="store_false",
                   help="drop thickness/flux from the tig and pkhrr inputs "
                        "(on by default)")
    p.add_argument("--aggregate", choices=["none", "mean"], default="none",
                   help="mean: average repeated measurements of the same pSMILES "
                        "into one row before fitting")
    p.add_argument("--drop_dead_features", action="store_true", default=True,
                   help="drop SAE features that are zero for every molecule")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None, help="write per-row predictions here (csv)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # one embedding pass over every pSMILES appearing in any of the three files
    smiles = []
    for prop in args.properties:
        fname = SPECS[prop][0]
        df = pd.read_csv(os.path.join(args.data_dir, fname), sep=r"\s+")
        smiles.extend(df.smiles_canonicalized.tolist())
    uniq = sorted(set(smiles))
    print(f">> {len(uniq)} unique pSMILES across {len(set(SPECS[p][0] for p in args.properties))} files")

    Z = extract_embeddings(uniq, args, device)
    C = sae_encode(Z, args, device)
    code_of = dict(zip(uniq, C))

    # dead features carry no information and only slow the head down
    keep = {}
    if args.drop_dead_features:
        for prop in args.properties:
            _, X, _ = build_xy(prop, code_of, args)
            ncode = C.shape[1]
            alive = (X[:, :ncode] != 0).any(axis=0)
            keep[prop] = np.concatenate([alive, np.ones(X.shape[1] - ncode, dtype=bool)])

    results, frames = {}, {}
    for prop in args.properties:
        mse, mre, out = run_property(prop, code_of, keep, args, device)
        results[prop] = (mse, mre)
        frames[prop] = out

    print("\n" + "=" * 62)
    print(f"alpha={args.alpha}  k={args.k}  solver={args.solver}  "
          f"transform={args.target_transform}  conditions={args.use_conditions}")
    print(f"{'property':<10} {'n':>5} {'MSE':>14} {'RMSE':>12} {'rel err %':>10}")
    print("-" * 62)
    for prop in args.properties:
        mse, mre = results[prop]
        n = len(frames[prop])
        print(f"{prop:<10} {n:>5} {mse:>14.6g} {mse ** 0.5:>12.6g} {mre:>10.2f}")
    print("=" * 62)
    print("rel err % = ||y_pred - y_true||_2 / ||y_true||_2 x 100")
    if args.out:
        allf = pd.concat([f.assign(property=p) for p, f in frames.items()], ignore_index=True)
        allf.to_csv(args.out, index=False)
        print(f"\nwrote per-row predictions to {args.out}")


if __name__ == "__main__":
    main()
