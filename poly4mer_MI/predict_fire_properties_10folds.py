"""
Fire-property prediction over the 10-fold splits in data/10folds/.

For each property and each fold data_{0..9}: train a linear head on train.csv,
evaluate on test.csv, and keep the epoch with the lowest test relative error.
Then take the 5 folds with the lowest test error and report mean +/- std.

Pipeline is the same as predict_fire_properties.py and reuses its code:
  pSMILES --[frozen CLLM]--> z in R^768 --[frozen Top-K SAE]--> c --[linear head]--> property

Error metric (both train and test):
  rel err % = ||y_pred - y_true||_2 / ||y_true||_2 x 100

The sparse code is never scaled. Thickness and flux are log-transformed (so the
head can fit the power laws ignition physics follows; --condition_transform none
for raw) and standardized. Their statistics, the target statistics and the
dead-feature mask are all fit on the training split only. The epoch is
chosen on test error, as specified -- that is selection on the evaluation set,
so the `test@final` column is printed alongside for reference.

Each selected head is saved to
  checkpoint/fire_heads_10folds/alpha{A}_k{K}/<property>_fold<i>.pt
carrying the weights plus everything inference needs: the dead-feature mask, the
target log-statistics and the alpha/k the code came from.
See load_head() at the bottom of this file.

Example
  python predict_fire_properties_10folds.py --alpha 8 --k 160
  python predict_fire_properties_10folds.py --alpha 32 --k 1000 --epochs 5000 --top 5
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from predict_fire_properties import (
    BASE, extract_embeddings, relative_error, sae_encode,
)

# property -> (folder under data/10folds, target column, condition columns)
DATASETS = {
    "tig":   ("tig_exp_data",      "tig",   ["thickness", "flux"]),
    "pkhrr": ("pkhrr_exp_data",    "pkhrr", ["thickness", "flux"]),
    "sea":   ("SEA_exp_data_avg",  "SEA",   []),
    "co":    ("CO_exp_data_avg",   "CO",    []),
}


def fold_dir(args, prop, i):
    return os.path.join(args.folds_dir, DATASETS[prop][0], f"data_{i}")


def build_split(df, prop, code_of, args):
    """-> X (code | log conditions), y"""
    cond = DATASETS[prop][2] if args.use_conditions else []
    C = np.stack([code_of[s] for s in df.smiles_canonicalized])
    if cond:
        v = df[cond].to_numpy(dtype=np.float64)
        if args.condition_transform == "log":
            v = np.log(v)
        X = np.hstack([C, v])
    else:
        X = C
    return X, df[DATASETS[prop][1]].to_numpy(dtype=np.float64)


def run_fold(prop, i, code_of, args, device):
    tr = pd.read_csv(os.path.join(fold_dir(args, prop, i), "train.csv"))
    te = pd.read_csv(os.path.join(fold_dir(args, prop, i), "test.csv"))
    Xtr, ytr = build_split(tr, prop, code_of, args)
    Xte, yte = build_split(te, prop, code_of, args)

    n_cond = len(DATASETS[prop][2]) if args.use_conditions else 0
    n_code = Xtr.shape[1] - n_cond

    # dead-feature mask from TRAIN only
    if args.drop_dead_features:
        alive = (Xtr[:, :n_code] != 0).any(axis=0)
        mask = np.concatenate([alive, np.ones(n_cond, dtype=bool)])
        Xtr, Xte = Xtr[:, mask], Xte[:, mask]
        n_code = int(alive.sum())

    # the sparse code is never scaled; the condition columns are standardized
    # with training-split statistics
    cond_mu = np.zeros(Xtr.shape[1]); cond_sd = np.ones(Xtr.shape[1])
    if n_cond:
        cond_mu[n_code:] = Xtr[:, n_code:].mean(0)
        cond_sd[n_code:] = Xtr[:, n_code:].std(0)
        cond_sd[cond_sd < 1e-8] = 1.0
    Xtr_s, Xte_s = (Xtr - cond_mu) / cond_sd, (Xte - cond_mu) / cond_sd

    # target transform from TRAIN only
    log = args.target_transform == "log"
    ttr = np.log(ytr) if log else ytr
    mu_y, sd_y = ttr.mean(), ttr.std()
    sd_y = sd_y if sd_y > 1e-8 else 1.0

    Xt = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    Yt = torch.tensor((ttr - mu_y) / sd_y, dtype=torch.float32, device=device).view(-1, 1)
    Xv = torch.tensor(Xte_s, dtype=torch.float32, device=device)

    torch.manual_seed(args.seed + i)
    head = nn.Linear(Xt.shape[1], 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.MSELoss()

    def to_units(out):
        v = out.detach().cpu().numpy().ravel() * sd_y + mu_y
        return np.exp(v) if log else v

    best = {"test": np.inf, "train": np.inf, "epoch": -1}
    best_state = copy.deepcopy(head.state_dict())
    final_test = np.nan
    for ep in range(1, args.epochs + 1):
        head.train()
        opt.zero_grad()
        lossf(head(Xt), Yt).backward()
        opt.step()

        if ep % args.eval_every and ep != args.epochs:
            continue
        head.eval()
        with torch.no_grad():
            rtr = relative_error(ytr, to_units(head(Xt)))
            rte = relative_error(yte, to_units(head(Xv)))
        final_test = rte
        if rte < best["test"]:
            best = {"test": rte, "train": rtr, "epoch": ep}
            best_state = copy.deepcopy(head.state_dict())

    ckpt_path = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"{prop}_fold{i}.pt")
        torch.save({
            # the selected head (lowest test relative error) and the last one
            "state_dict": {k: v.cpu() for k, v in best_state.items()},
            "state_dict_final": {k: v.cpu() for k, v in head.state_dict().items()},
            "property": prop, "fold": i,
            "alpha": args.alpha, "k": args.k,
            "in_features": int(Xtr.shape[1]),
            # rebuild X: C[:, code_mask], append transformed conditions, then
            # (X - cond_mu) / cond_sd -- identity on the code columns
            "code_mask": alive if args.drop_dead_features else None,
            "condition_cols": DATASETS[prop][2] if args.use_conditions else [],
            "condition_transform": args.condition_transform,
            "n_code": int(n_code), "n_cond": int(n_cond),
            "cond_mu": cond_mu, "cond_sd": cond_sd,
            "mu_y": float(mu_y), "sd_y": float(sd_y),
            "target_transform": args.target_transform,
            "best_epoch": best["epoch"], "train_rel": best["train"],
            "test_rel": best["test"], "test_final": final_test,
            "train_csv": os.path.join(fold_dir(args, prop, i), "train.csv"),
            "test_csv": os.path.join(fold_dir(args, prop, i), "test.csv"),
        }, ckpt_path)

    return {"ckpt": ckpt_path, "fold": i, "n_train": len(tr), "n_test": len(te), "n_feat": Xtr.shape[1],
            "best_epoch": best["epoch"], "train_rel": best["train"],
            "test_rel": best["test"], "test_final": final_test}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--k", type=int, default=160)
    p.add_argument("--sae_ckpt", default=None)
    p.add_argument("--cllm_ckpt", default=f"{BASE}/checkpoint/poly4mer_v3.0.0.ckpt")
    p.add_argument("--folds_dir", default=f"{BASE}/data/10folds")
    p.add_argument("--cache_dir", default=f"{BASE}/checkpoint/sae_cache")
    p.add_argument("--properties", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--top", type=int, default=5, help="how many best folds to summarize")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--eval_every", type=int, default=1, help="epochs between evaluations")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--target_transform", choices=["log", "none"], default="log")
    p.add_argument("--no_conditions", dest="use_conditions", action="store_false")
    p.add_argument("--condition_transform", choices=["log", "none"], default="log",
                   help="log: head fits power laws tig ~ flux^b thickness^c; "
                        "none: raw values. Standardized either way.")
    p.add_argument("--drop_dead_features", action="store_true", default=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=None,
                   help="directory for the trained heads "
                        "(default checkpoint/fire_heads_10folds/alpha{A}_k{K})")
    p.add_argument("--no_save", dest="save_dir", action="store_const", const="",
                   help="do not save the heads")
    p.add_argument("--out", default=None, help="write the per-fold table here (csv)")
    args = p.parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.join(BASE, "checkpoint", "fire_heads_10folds",
                                     f"alpha{args.alpha}_k{args.k}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")
    if args.save_dir:
        print(f">> Heads will be saved to {args.save_dir}")

    # every fold of every property draws on the same molecule set: embed once
    smiles = set()
    for prop in args.properties:
        for i in range(args.folds):
            for split in ("train.csv", "test.csv"):
                df = pd.read_csv(os.path.join(fold_dir(args, prop, i), split))
                smiles |= set(df.smiles_canonicalized)
    uniq = sorted(smiles)
    print(f">> {len(uniq)} unique pSMILES across all folds")

    Z = extract_embeddings(uniq, args, device)
    C = sae_encode(Z, args, device)
    code_of = dict(zip(uniq, C))

    rows, summary = [], {}
    for prop in args.properties:
        print(f"\n{'=' * 74}\n{prop}   ({DATASETS[prop][0]})\n{'=' * 74}")
        print(f"{'fold':>4} {'n_tr':>5} {'n_te':>5} {'feat':>6} {'best_ep':>8} "
              f"{'train rel %':>12} {'test rel %':>11} {'test@final':>11}")
        print("-" * 74)
        recs = []
        for i in range(args.folds):
            r = run_fold(prop, i, code_of, args, device)
            r["property"] = prop
            recs.append(r)
            rows.append(r)
            print(f"{r['fold']:>4} {r['n_train']:>5} {r['n_test']:>5} {r['n_feat']:>6} "
                  f"{r['best_epoch']:>8} {r['train_rel']:>12.2f} {r['test_rel']:>11.2f} "
                  f"{r['test_final']:>11.2f}")

        srt = sorted(recs, key=lambda r: r["test_rel"])[:args.top]
        te = np.array([r["test_rel"] for r in srt])
        tr = np.array([r["train_rel"] for r in srt])
        all_te = np.array([r["test_rel"] for r in recs])
        summary[prop] = {
            "top_folds": [r["fold"] for r in srt],
            "test_mean": te.mean(), "test_std": te.std(ddof=1),
            "train_mean": tr.mean(), "train_std": tr.std(ddof=1),
            "all_mean": all_te.mean(), "all_std": all_te.std(ddof=1),
        }
        s = summary[prop]
        print("-" * 74)
        print(f"  all {args.folds} folds : test {s['all_mean']:.2f} +/- {s['all_std']:.2f} %")
        print(f"  top {args.top} folds {s['top_folds']}:")
        print(f"      train {s['train_mean']:.2f} +/- {s['train_std']:.2f} %")
        print(f"      test  {s['test_mean']:.2f} +/- {s['test_std']:.2f} %")

    print(f"\n{'=' * 74}")
    print(f"SUMMARY  alpha={args.alpha}  k={args.k}  epochs={args.epochs}  "
          f"top={args.top} of {args.folds} folds")
    print(f"{'=' * 74}")
    print(f"{'property':<10} {'top folds':<20} {'train rel %':>18} {'test rel %':>18}")
    print("-" * 74)
    for prop in args.properties:
        s = summary[prop]
        print(f"{prop:<10} {str(s['top_folds']):<20} "
              f"{s['train_mean']:>10.2f} +/- {s['train_std']:<5.2f} "
              f"{s['test_mean']:>10.2f} +/- {s['test_std']:<5.2f}")
    print("=" * 74)
    print("rel err % = ||y_pred - y_true||_2 / ||y_true||_2 x 100 ; std over the "
          f"{args.top} selected folds (ddof=1)")

    if args.save_dir:
        n = sum(1 for r in rows if r.get("ckpt"))
        print(f"\nsaved {n} trained heads to {args.save_dir}/<property>_fold<i>.pt")

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"wrote per-fold results to {args.out}")


def load_head(path, device="cpu", which="state_dict"):
    """Reload a saved head. Returns (head, meta); apply it to a code matrix with

        C = sae.encode(z)                       # [N, alpha*768]
        X = C[:, meta["code_mask"]]             # drop dead features
        X = np.hstack([X, cond])                # log(cond) if condition_transform=="log"
        X = (X - meta["cond_mu"]) / meta["cond_sd"]   # identity on code columns
        y = head(X) * meta["sd_y"] + meta["mu_y"]
        y = np.exp(y) if meta["target_transform"] == "log" else y
    """
    meta = torch.load(path, map_location="cpu", weights_only=False)
    head = nn.Linear(meta["in_features"], 1)
    head.load_state_dict(meta[which])
    return head.to(device).eval(), meta


if __name__ == "__main__":
    main()
