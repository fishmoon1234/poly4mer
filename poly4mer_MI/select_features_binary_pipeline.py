"""
Essential-feature selection pipeline for a trained binary Group-ID classifier
(from train_group_binary_classifier.py).

Produces ONE GLOBAL feature subset of size --top_n, plus a LASSO-retrained
comparison subset, and validates sufficiency/necessity.

Method 1: activation x weight
  score1[j] = E_val[c_j] * |W[0, j]|

Method 2: marginal AUROC
  score2[j] = AUROC(y_val, c_val[:, j])

Combined (Borda): mean rank across (1) and (2), pick the top --top_n.
  Alternative --combine_mode intersection: top-2N from each, intersect,
  then keep the --top_n best by mean rank within the intersection.

Validation (held-out val, classifier weights FROZEN):
  - baseline AUROC                      : full c, no masking
  - sufficiency AUROC                    : keep only selected features (zero others)
                                           Stays close to baseline -> subset is sufficient.
  - necessity AUROC                      : zero only selected features (keep others)
                                           Drops sharply -> subset is necessary.

LASSO retrain (Method 3, --l1_lambda required to trigger):
  L_lambda = BCE(pos_weight) + lambda * sum_j |W[0, j]|
  Warm-started from the input classifier. The selected set at strength lambda
  is {j : |W^lambda[0, j]| > --l1_select_threshold}.

Outputs (default out_dir = the input ckpt's directory):
  feature_selection_<base>.tsv   one row per feature: scores, ranks, selected flags
  validation_<base>.json         baseline / sufficiency / necessity + LASSO summary
  <base>_l1lambda<L>.pt          retrained sparse head per lambda (if --save_l1_ckpts)
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from models import TopKSAE
from train_group_binary_classifier import auroc, load_binary_labels
from train_group_classifier import load_sae



## Collect activation statistics , encoding each batch of polymer embeddings through the frozen SAE to get the sparse feature codes c

@torch.no_grad()
def collect_activation_stats(sae, Z_subset, y_subset, device, batch_size, D):
    """Per-feature stats + full c matrix on the subset.

    Returns dict:
      mean_act      : (D,)
      alive_rate    : (D,)
      cond_mean     : (2, D) -- E[c_j | y=0] and E[c_j | y=1] (NaN where class absent)
      all_c         : (M_sub, D) float16, for per-feature AUROC downstream
    """
    M = Z_subset.shape[0]
    mean_act = torch.zeros(D, dtype=torch.float64, device=device)
    alive    = torch.zeros(D, dtype=torch.float64, device=device)
    cond_sum = torch.zeros((2, D), dtype=torch.float64, device=device)
    cond_n   = torch.zeros(2, dtype=torch.float64, device=device)
    all_c    = np.zeros((M, D), dtype=np.float16)

    for i in tqdm(range(0, M, batch_size), desc="activation stats"):
        z = Z_subset[i:i + batch_size].to(device, non_blocking=True)
        y = torch.from_numpy(y_subset[i:i + batch_size]).to(device)
        c = sae.encode(z)                                # (B, D), float32
        mean_act += c.double().sum(dim=0)
        alive    += (c > 0).double().sum(dim=0)
        for cls in (0, 1):
            mask = (y == cls)
            n = int(mask.sum().item())
            if n == 0:
                continue
            cond_sum[cls] += c[mask].double().sum(dim=0)
            cond_n[cls]  += n
        all_c[i:i + c.shape[0]] = c.cpu().numpy().astype(np.float16)

    mean_act = (mean_act / max(M, 1)).cpu().numpy()
    alive    = (alive    / max(M, 1)).cpu().numpy()
    cond_mean = np.full((2, D), np.nan, dtype=np.float64)
    for cls in (0, 1):
        if cond_n[cls] > 0:
            cond_mean[cls] = (cond_sum[cls] / cond_n[cls]).cpu().numpy()
    return {"mean_act": mean_act, "alive_rate": alive,
            "cond_mean_0": cond_mean[0], "cond_mean_1": cond_mean[1],
            "all_c": all_c}

## Compute per-feature AUROC for each feature c_j as a score for y, returning NaN where c_j fires fewer than alive_min times.
# for each SAE feature, ask how well that feature alone would rank polymers by whether they contain the target group.
def per_feature_auroc(all_c, y, alive_min=10):
    """AUROC of each c_j as a score for y. NaN where c_j fires fewer than alive_min times."""
    M, D = all_c.shape
    aucs = np.full(D, np.nan, dtype=np.float64)
    for j in tqdm(range(D), desc="per-feature AUROC"):
        col = all_c[:, j].astype(np.float64)
        if int((col > 0).sum()) < alive_min:
            continue
        aucs[j] = auroc(y, col)
    return aucs

## combines the Borda count ranking of the two methods (activation x weight and per-feature AUROC) to select the top_n features based on the specified mode (borda or intersection).
def combine_rankings(score_general, score_auroc, mode, top_n):
    """Return: dense ranks for each method (1-best), combined mean rank, selected mask."""
    finite1 = np.isfinite(score_general)
    finite2 = np.isfinite(score_auroc)
    finite  = finite1 & finite2

    D = score_general.shape[0]
    rank1 = np.full(D, D + 1, dtype=np.float64)
    rank2 = np.full(D, D + 1, dtype=np.float64)
    rank1[finite1] = (-score_general[finite1]).argsort().argsort() + 1
    rank2[finite2] = (-score_auroc  [finite2]).argsort().argsort() + 1
    mean_rank = (rank1 + rank2) / 2.0

    if mode == "borda":
        order = np.argsort(mean_rank, kind="mergesort")
        selected = np.zeros(D, dtype=bool)
        selected[order[:top_n]] = True
    elif mode == "intersection":
        top_each = max(top_n * 2, top_n + 50)
        s1 = np.argsort(rank1, kind="mergesort")[:top_each]
        s2 = np.argsort(rank2, kind="mergesort")[:top_each]
        inter = np.intersect1d(s1, s2)
        order = inter[np.argsort(mean_rank[inter], kind="mergesort")]
        selected = np.zeros(D, dtype=bool)
        selected[order[:top_n]] = True
    else:
        raise ValueError(f"unknown combine_mode {mode}")
    return rank1, rank2, mean_rank, selected

## runs validation AUROC  through frozen classifier 
# on the val set after applying a multiplicative mask to the SAE-encoded features

@torch.no_grad()
def val_auroc_with_mask(classifier, sae, Z_val, y_val_np, device, batch_size, mask):
    """val AUROC after applying mask (multiplicative, shape (D,)) to c."""
    classifier.eval()
    M = Z_val.shape[0]
    probs = np.zeros(M, dtype=np.float32)
    for i in range(0, M, batch_size):
        z = Z_val[i:i + batch_size].to(device, non_blocking=True)
        c = sae.encode(z) * mask.unsqueeze(0)
        logits = classifier(c).squeeze(-1)
        probs[i:i + z.shape[0]] = torch.sigmoid(logits).cpu().numpy()
    return auroc(y_val_np, probs)

"""classifier_init_state: dict of original trained classifiers param, λ run begins from these
Z: matrix if shape (M, D), M is the total number of polymers in your dataset 
y: (M,) float32, matrix corresponding to the target group ID
train_idx: (N_train,) long tensor of training row indices
val_idx: (N_val,) long tensor of validation row indices
sae: TopKSAE instance (frozen)
device: torch.device
args: argparse.Namespace with hyperparameters
l1_lambda: float, L1 regularization strength
Returns: (new_state_dict, val_curve, w_abs)
  new_state_dict: dict of nn.Linear(D, 1) state_dict after LASSO retrain
  val_curve: list of dicts with keys "epoch" and "auroc" for each epoch
  w_abs: (D,) numpy array of absolute weights after retrain
"""

def l1_retrain(classifier_init_state, Z, y, train_idx, val_idx, sae, device,
               args, l1_lambda):
    """LASSO logistic retrain (warm start) at given lambda."""
    D = sae.D  ## dictionary size
    head = nn.Linear(D, 1).to(device)
    head.load_state_dict(classifier_init_state)

    ## sets up the build with loss, BCEWithLogitsLoss with pos_weight to handle class imbalance
    y_train = y[train_idx.numpy()]
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pw = min(args.pos_weight_cap, n_neg / max(n_pos, 1.0))
    pos_weight = torch.tensor(pw, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)

    Y = torch.from_numpy(y)
    n_train = train_idx.shape[0]
    n_val = val_idx.shape[0]
    g = torch.Generator().manual_seed(args.random_seed)

    val_curve = []
    for epoch in range(args.l1_epochs):
        head.train()  # puts head on training mode 
        perm = train_idx[torch.randperm(n_train, generator=g)]
        for start in tqdm(range(0, n_train, args.batch_size),
                          desc=f"LASSO lambda={l1_lambda} ep {epoch+1}/{args.l1_epochs}",
                          leave=False):
            idx = perm[start:start + args.batch_size]
            z = Z[idx].to(device, non_blocking=True)
            y_b = Y[idx].to(device, non_blocking=True)
            with torch.no_grad():
                c = sae.encode(z)
            logits = head(c).squeeze(-1)
            ce = bce(logits, y_b) ## FORWARD PASS
            penalty = l1_lambda * head.weight.abs().sum()
            loss = ce + penalty # total loss = cross entropy + l1 penalty   
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # val AUROC for this epoch
        head.eval()
        val_probs = np.zeros(n_val, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n_val, args.batch_size):
                idx = val_idx[start:start + args.batch_size]
                z = Z[idx].to(device, non_blocking=True)
                c = sae.encode(z)
                logits = head(c).squeeze(-1)
                val_probs[start:start + idx.shape[0]] = torch.sigmoid(logits).cpu().numpy()
        val_y = y[val_idx.numpy()]
        val_curve.append({"epoch": epoch + 1, "auroc": auroc(val_y, val_probs)})

    head.eval()
    w_abs = head.weight.detach().abs().squeeze(0).cpu().numpy()
    return head.state_dict(), val_curve, w_abs


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="Single-group BINARY classifier ckpt from "
                             "train_group_binary_classifier.py.")
    parser.add_argument("--data_path", type=str,
                        default=f"{base_dir}/data/v1_fragment_ids_correct.smi")
    parser.add_argument("--labels_path", type=str,
                        default=f"{base_dir}/data/v1_fragments_ids_correct_binary.smi")
    parser.add_argument("--cache_dir", type=str,
                        default=f"{base_dir}/checkpoint/sae_cache")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Default: directory containing --classifier_ckpt.")

    # Selection
    parser.add_argument("--top_n", type=int, default=100,
                        help="Size of the global feature subset.")
    parser.add_argument("--combine_mode", type=str, default="borda",
                        choices=["borda", "intersection"])
    parser.add_argument("--act_sample_size", type=int, default=200_000,
                        help="Subsample size for activation stats and "
                             "per-feature AUROC (drawn from the val split).")
    parser.add_argument("--auroc_alive_min", type=int, default=10,
                        help="Features firing on fewer val rows than this get "
                             "AUROC=NaN (excluded from method 2).")

    # Match training-time hyperparams to reproduce the train/val split
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4096)

    # LASSO retrain
    parser.add_argument("--l1_lambda", type=float, action="append", default=[],
                        help="L1 strength. Repeat to sweep. Empty -> no LASSO retrain.")
    parser.add_argument("--l1_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--l1_select_threshold", type=float, default=1e-3)
    ## added snippet 
    parser.add_argument("--l1_top_n", type=int, default=0,
                        help="If >0, keep only the l1_top_n largest |W| after LASSO.")
    parser.add_argument("--save_l1_ckpts", action="store_true")

    # Validation
    parser.add_argument("--no_validate", action="store_true")

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")

    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(args.classifier_ckpt))
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.classifier_ckpt))[0]

    # 1. Load classifier
    print(f">> Load classifier from {args.classifier_ckpt}")
    cckpt = torch.load(args.classifier_ckpt, map_location=device, weights_only=False)
    cfg = cckpt["config"]
    if cfg.get("head") != "linear_binary":
        raise ValueError(
            f"Expected a binary classifier with head='linear_binary'; got "
            f"head={cfg.get('head')!r}. This pipeline's Method 1 score relies "
            f"on the single weight row of nn.Linear(D, 1); an MLP head needs a "
            f"different attribution (gradient x input / integrated gradients). "
            f"Retrain a linear baseline via `train_group_binary_classifier.py "
            f"--head_type linear` if you need feature selection on this ID."
        )
    sel_ids = cfg.get("selected_ids", [])
    if len(sel_ids) != 1:
        raise ValueError(f"Expected single Group ID; got selected_ids={sel_ids}")
    target_id = sel_ids[0]
    print(f"   target group ID: {target_id}")

    # 2. SAE
    sae_path = cfg["sae_ckpt"]
    print(f">> Load SAE from {sae_path}")
    sae, scfg = load_sae(sae_path, device)
    D = scfg["D"]

    head = nn.Linear(D, 1).to(device)
    head.load_state_dict(cckpt["state_dict"])
    head.eval()
    W = head.weight.detach().cpu().numpy().squeeze(0)   # (D,)
    b = float(head.bias.detach().cpu().item())
    print(f"   classifier: Linear({D}, 1), |W|.max={np.abs(W).max():.4f}, b={b:+.4f}")

    # 3. Cached embeddings
    tag = hashlib.md5(os.path.abspath(args.data_path).encode()).hexdigest()[:10]
    cache_path = os.path.join(args.cache_dir, f"psmiles_embeddings_{tag}.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"No cached embeddings at {cache_path}")
    print(f">> Load cached Z from {cache_path}")
    Z = torch.load(cache_path, map_location="cpu")

    # 4. Binary labels for the target ID
    _, Y_full = load_binary_labels(args.labels_path)
    assert Y_full.shape[0] == Z.shape[0]
    y = Y_full[:, target_id - 1].astype(np.float32)

    # 5. Reproduce the train/val split (matches train_group_binary_classifier.py)
    g_split = torch.Generator().manual_seed(args.random_seed)
    M = Z.shape[0]
    n_val = int(round(args.val_frac * M))
    perm = torch.randperm(M, generator=g_split)
    train_idx = perm[:M - n_val]
    val_idx = perm[M - n_val:]
    print(f">> Train/val split: {train_idx.numel()} / {val_idx.numel()}")

    # 6. Activation stats + per-feature AUROC on a val subsample
    val_Z_full = Z[val_idx]
    val_y_full = y[val_idx.numpy()]
    sample_size = min(args.act_sample_size, val_idx.numel())
    if sample_size < val_idx.numel():
        rng = np.random.default_rng(args.random_seed)
        sub_idx = np.sort(rng.choice(val_idx.numel(), size=sample_size, replace=False))
        val_Z_sub = val_Z_full[torch.from_numpy(sub_idx)]
        val_y_sub = val_y_full[sub_idx]
    else:
        val_Z_sub = val_Z_full
        val_y_sub = val_y_full
    print(f">> Activation stats on {sample_size} val samples (pos_rate={val_y_sub.mean():.4f})")
    stats = collect_activation_stats(sae, val_Z_sub, val_y_sub, device,
                                     args.batch_size, D)
    aucs = per_feature_auroc(stats["all_c"], val_y_sub, alive_min=args.auroc_alive_min)

    # 7. Scores
    score_general = stats["mean_act"] * np.abs(W)
    score_auroc   = aucs

    rank1, rank2, mean_rank, selected = combine_rankings(
        score_general, score_auroc, args.combine_mode, args.top_n
    )
    sel_idx = np.flatnonzero(selected)
    print(f">> Combined ranking ({args.combine_mode}): selected top {len(sel_idx)} features.")

    # 8. Validation: baseline / sufficiency / necessity (on the FULL val split)
    val_results = {}
    if not args.no_validate:
        ones = torch.ones(D, dtype=torch.float32, device=device)
        keep_only_selected = torch.zeros(D, dtype=torch.float32, device=device)
        keep_only_selected[torch.from_numpy(sel_idx.astype(np.int64)).to(device)] = 1.0
        keep_others = 1.0 - keep_only_selected

        print(">> Validating on full val split...")
        auc_baseline = val_auroc_with_mask(head, sae, val_Z_full, val_y_full,
                                           device, args.batch_size, ones)
        auc_sufficient = val_auroc_with_mask(head, sae, val_Z_full, val_y_full,
                                             device, args.batch_size, keep_only_selected)
        auc_necessary = val_auroc_with_mask(head, sae, val_Z_full, val_y_full,
                                            device, args.batch_size, keep_others)
        val_results = {
            "baseline_auroc": auc_baseline,
            "sufficiency_auroc_keep_only_selected": auc_sufficient,
            "necessity_auroc_zero_selected_keep_others": auc_necessary,
            "delta_baseline_minus_sufficient": auc_baseline - auc_sufficient,
            "delta_baseline_minus_necessary": auc_baseline - auc_necessary,
        }
        print(f"   baseline AUROC                       : {auc_baseline:.6f}")
        print(f"   sufficiency AUROC (keep selected)    : {auc_sufficient:.6f}  "
              f"(drop {auc_baseline - auc_sufficient:+.6f}; close to 0 -> subset is sufficient)")
        print(f"   necessity AUROC (zero selected)      : {auc_necessary:.6f}  "
              f"(drop {auc_baseline - auc_necessary:+.6f}; large drop -> subset is necessary)")

    # 9. LASSO retrain at each requested lambda
    lasso_results = {}
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    for lam in args.l1_lambda:
        print(f"\n>> LASSO retrain at lambda={lam} (epochs={args.l1_epochs})")
        new_state, val_curve, w_abs = l1_retrain(
            init_state, Z, y, train_idx, val_idx, sae, device, args, lam
        )
        # n_sel_l1 = int((w_abs > args.l1_select_threshold).sum())
        # sel_l1 = np.flatnonzero(w_abs > args.l1_select_threshold)
        if args.l1_top_n > 0:
            sel_l1 = np.argsort(-w_abs)[:args.l1_top_n]
            sel_l1 = sel_l1[w_abs[sel_l1] > args.l1_select_threshold]  # drop dead ones
            sel_l1 = np.sort(sel_l1)
        else:
            sel_l1 = np.flatnonzero(w_abs > args.l1_select_threshold)
        n_sel_l1 = int(sel_l1.size)
        overlap = int(np.intersect1d(sel_l1, sel_idx).size)
        print(f"   n_selected@thresh={args.l1_select_threshold}: {n_sel_l1}/{D}  "
              f"overlap with method (1+2) top-{args.top_n}: {overlap}")
        lasso_results[str(lam)] = {
            "n_selected": n_sel_l1,
            "selected_feature_indices": sel_l1.tolist(),
            "select_threshold": args.l1_select_threshold,
            "overlap_with_combined_subset": overlap,
            "val_auroc_final": val_curve[-1]["auroc"],
            "val_curve_auroc": [v["auroc"] for v in val_curve],
        }
        if args.save_l1_ckpts:
            l1_path = os.path.join(args.out_dir, f"{base}_l1lambda{lam}.pt")
            torch.save({
                "state_dict": new_state,
                "config": {**cfg, "l1_lambda": lam},
                "n_selected": n_sel_l1,
                "select_threshold": args.l1_select_threshold,
                "val_curve_auroc": [v["auroc"] for v in val_curve],
                "init_classifier_ckpt": args.classifier_ckpt,
            }, l1_path)
            print(f"   saved retrained head -> {l1_path}")

    # 10. Per-feature TSV
    df_cols = {
        "feat_idx":     np.arange(D),
        "mean_act":     stats["mean_act"],
        "alive_rate":   stats["alive_rate"],
        "mean_act_y0":  stats["cond_mean_0"],
        "mean_act_y1":  stats["cond_mean_1"],
        "abs_weight":   np.abs(W),
        "weight":       W,
        "score_general": score_general,
        "score_auroc":   score_auroc,
        "rank_general":  rank1,
        "rank_auroc":    rank2,
        "mean_rank":     mean_rank,
        "selected":      selected.astype(np.int8),
    }
    # One per-lambda boolean column flagging LASSO survivors
    for lam, res in lasso_results.items():
        col = np.zeros(D, dtype=np.int8)
        col[np.asarray(res["selected_feature_indices"], dtype=np.int64)] = 1
        df_cols[f"selected_l1lambda{lam}"] = col
    df = pd.DataFrame(df_cols)
    df.sort_values(["selected", "mean_rank"], ascending=[False, True], inplace=True)
    tsv_path = os.path.join(args.out_dir, f"feature_selection_{base}.tsv")
    df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6e")
    print(f"\n>> Wrote feature ranking to {tsv_path}")

    # 11. Validation JSON
    val_json_path = os.path.join(args.out_dir, f"validation_{base}.json")
    out_obj = {
        "classifier_ckpt": os.path.abspath(args.classifier_ckpt),
        "target_id": target_id,
        "D": D, "top_n": args.top_n, "combine_mode": args.combine_mode,
        "selected_feature_indices": sel_idx.tolist(),
        "validation": val_results,
        "lasso": lasso_results,
    }
    with open(val_json_path, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f">> Wrote validation summary to {val_json_path}")

    # 12. Print top-15 from the combined subset for a quick eyeball
    print(f"\n>> Top 15 from combined subset:")
    print(f"   {'feat_idx':>8}  {'mean_act':>10}  {'|W|':>9}  {'AUROC':>7}  "
          f"{'rank1':>5}  {'rank2':>5}  {'mean_rank':>9}")
    head_df = df[df["selected"] == 1].head(15)
    for _, r in head_df.iterrows():
        print(f"   {int(r.feat_idx):>8}  {r.mean_act:>10.4e}  {r.abs_weight:>9.3e}  "
              f"{r.score_auroc:>7.4f}  {int(r.rank_general):>5}  {int(r.rank_auroc):>5}  "
              f"{r.mean_rank:>9.1f}")


if __name__ == "__main__":
    main()
