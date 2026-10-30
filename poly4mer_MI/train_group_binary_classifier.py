"""
Per-Group-ID binary logistic-regression classifier on top of a trained Top-K SAE.

For each pSMILES, predict whether fragment ID g appears in its fragment_ids list
(binary: 0 = absent, 1 = present). When --group_ids selects more than one ID,
an *independent* classifier is trained per ID -- each gets its own optimizer,
LR schedule, ckpt, and metrics TSV; only the frozen SAE features and the
random train/val split are shared across IDs.

Pipeline (per Group ID):
  z (cached pSMILES embedding, R^768)
    -> SAE.encode(.)  -> c in R^D       [frozen, D = alpha * 768]
    -> nn.Linear(D, 1)                  [trainable: one binary logit]
    -> sigmoid + 0.5 threshold          -> prediction in {0, 1}

This is the textbook weighted (binary) logistic regression with a frozen
feature transform: the loss is BCEWithLogitsLoss(pos_weight=n_neg/n_pos),
where the pos_weight upweights the positive class to counteract the
heavy negative-class skew (the rare class is "ID g is present").

Inputs:
  --sae_ckpt    : SAE checkpoint produced by train_topk_sae.py
  --data_path   : the .smi whose absolute path keys the cached embeddings
                  (md5 of abspath). Contents are NOT read unless
                  --verify_alignment is passed.
  --labels_path : (1 + N_GROUPS)-column headerless smi from
                  extract_fragment_binary_labels.py -- 197 columns for the
                  196-fragment reference library. Row order MUST match the
                  row order of the embeddings in the cache.

Outputs (one set per selected Group ID):
  checkpoint/group_binary_classifier/group_bin_clf_<sae_tag>_ids_<id>.pt
      classifier ckpt (state_dict, config, pos_weight, val_metrics, args).
  checkpoint/group_binary_classifier/group_bin_clf_<sae_tag>_ids_<id>.metrics.tsv
      per-epoch loss / metrics / lr (line-buffered).
"""

import argparse
import hashlib
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models import TopKSAE
from train_group_classifier import load_sae  # reuse


# Number of fragment IDs in data/fragments_reference.csv (ids are 1..196).
# Override at the CLI with --n_groups if the reference library changes.
N_GROUPS = 196


def build_head(D, head_type, mlp_hidden_dims=(1024, 256), mlp_dropout=0.1):
    """Build the binary classification head.

    head_type='linear': nn.Linear(D, 1) -- weighted logistic regression.
    head_type='mlp'   : Linear(D, h_1) -> GELU -> Dropout(p)
                        -> Linear(h_1, h_2) -> GELU -> Dropout(p) -> ...
                        -> Linear(h_{L-1}, 1).
    """
    if head_type == "linear":
        return nn.Linear(D, 1)
    if head_type == "mlp":
        dims = [D] + list(mlp_hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:                 # all but the final projection
                layers.append(nn.GELU())
                layers.append(nn.Dropout(mlp_dropout))
        return nn.Sequential(*layers)
    raise ValueError(f"unknown head_type: {head_type!r}")


def detect_label_width(path):
    """Number of label columns in a headerless <smiles v_1 ... v_n> file."""
    with open(path) as f:
        for line in f:
            if line.strip():
                return len(line.split()) - 1
    raise ValueError(f"{path} contains no non-blank lines")


def read_smiles_column(path, col=0):
    """First whitespace field of every row, skipping a 'smiles' header if present.

    Used only by --verify_alignment. Tolerates both the 4-column
    v1_fragment_ids_correct.smi (which may carry a header) and a plain
    2-column .smi.
    """
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            parts = line.split()
            if i == 0 and parts[col].strip().lower() == "smiles":
                continue                              # header line
            out.append(parts[col].strip())
    return out


def load_binary_labels(path, n_groups=N_GROUPS):
    """Parse <smiles bin_1 ... bin_n_groups>; values must be 0 or 1."""
    print(f">> Loading binary labels from {path}")

    width = detect_label_width(path)
    print(f"   detected {width} label columns ({width + 1} total)")
    if width != n_groups:
        raise ValueError(
            f"{path} has {width} label columns but n_groups={n_groups}. "
            f"pandas would silently absorb the surplus columns into the index "
            f"rather than raising, so this is a hard error. Pass "
            f"--n_groups {width} if the reference library really has {width} "
            f"fragments, or regenerate the labels file."
        )

    cols = ["smiles"] + [f"id_{i}" for i in range(1, n_groups + 1)]
    dtypes = {c: np.float32 for c in cols[1:]}
    df = pd.read_csv(path, header=None, delimiter=" ", names=cols, dtype=dtypes)
    labels = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    bad = ~np.isin(labels, [0.0, 1.0])
    if bad.any():
        n_bad = int(bad.sum())
        print(f"   WARNING: {n_bad} cells are not in {{0, 1}}; clipping to {{0, 1}}")
        labels = (labels > 0).astype(np.float32)
    return df["smiles"].tolist(), labels


def auroc(y_true, y_score):
    """Mann-Whitney U formulation of AUROC. Both inputs are 1-D numpy arrays."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    # average ranks for ties
    ys_sorted = y_score[order]
    i = 0
    while i < len(ys_sorted):
        k = i
        while k + 1 < len(ys_sorted) and ys_sorted[k + 1] == ys_sorted[i]:
            k += 1
        if k > i:
            avg = ranks[order[i:k + 1]].mean()
            ranks[order[i:k + 1]] = avg
        i = k + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


@torch.no_grad()
def predict_probs(classifier, sae, Z, device, batch_size):
    classifier.eval()
    M = Z.shape[0]
    probs = np.zeros(M, dtype=np.float32)
    for i in range(0, M, batch_size):
        z = Z[i:i + batch_size].to(device, non_blocking=True)
        c = sae.encode(z)
        logits = classifier(c).squeeze(-1)            # (B,)
        probs[i:i + z.shape[0]] = torch.sigmoid(logits).cpu().numpy()
    return probs


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    yt = y_true.astype(np.int64)
    tp = int(((y_pred == 1) & (yt == 1)).sum())
    fp = int(((y_pred == 1) & (yt == 0)).sum())
    fn = int(((y_pred == 0) & (yt == 1)).sum())
    tn = int(((y_pred == 0) & (yt == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    acc = (tp + tn) / max(len(yt), 1)
    pos_rate = float(yt.mean())
    return {
        "pos_rate": pos_rate,
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "auroc": auroc(yt, y_prob),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def print_metrics(target_id, m, label="val"):
    print(f"\n>> Binary metrics for ID {target_id} ({label})")
    print(f"   pos_rate     : {m['pos_rate']:.4f}")
    print(f"   accuracy     : {m['acc']:.4f}")
    print(f"   precision    : {m['prec']:.4f}")
    print(f"   recall       : {m['rec']:.4f}")
    print(f"   F1           : {m['f1']:.4f}")
    print(f"   AUROC        : {m['auroc']:.4f}")
    print(f"   confusion    : TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")


def train_one_group(target_id, args, sae, D, Z, y_g, train_idx, val_idx,
                    save_path, device, base_dir):
    """Train one independent binary classifier for Group ID `target_id`."""
    n_train = train_idx.shape[0]
    n_val = val_idx.shape[0]

    y_train_np = y_g[train_idx.numpy()]
    n_pos_train = float(y_train_np.sum())
    n_neg_train = float(n_train - n_pos_train)
    pos_rate_train = n_pos_train / max(n_train, 1)

    if args.no_pos_weight:
        pos_weight = torch.tensor(1.0, device=device)
        print(f">> [ID {target_id}] Imbalance handling: DISABLED (uniform BCE)")
    else:
        if n_pos_train == 0:
            pos_weight = torch.tensor(1.0, device=device)
            print(f">> [ID {target_id}] WARNING: no positive samples in train; "
                  f"pos_weight=1.0")
        else:
            pw = min(args.pos_weight_cap, n_neg_train / n_pos_train)
            pos_weight = torch.tensor(pw, device=device)
            print(f">> [ID {target_id}] pos_rate(train)={pos_rate_train:.4f}, "
                  f"n_pos={int(n_pos_train):>9}, n_neg={int(n_neg_train):>9}, "
                  f"pos_weight={pos_weight.item():.3f}")

    Y = torch.from_numpy(y_g)
    classifier = build_head(D, args.head_type,
                            mlp_hidden_dims=args.mlp_hidden_dims,
                            mlp_dropout=args.mlp_dropout).to(device)
    n_params = sum(p.numel() for p in classifier.parameters())
    if args.head_type == "linear":
        head_desc = f"nn.Linear({D}, 1)"
    else:
        head_desc = (f"MLP({D} -> "
                     f"{' -> '.join(str(h) for h in args.mlp_hidden_dims)} -> 1, "
                     f"GELU, dropout={args.mlp_dropout})")
    print(f">> [ID {target_id}] head: {head_desc}  params={n_params}")

    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume_from not found: {args.resume_from}")
        prior = torch.load(args.resume_from, map_location=device, weights_only=False)
        prior_cfg = prior.get("config", {})
        if prior_cfg.get("D") != D:
            raise ValueError(
                f"--resume_from has D={prior_cfg.get('D')} but current SAE has D={D}"
            )
        prior_head = prior_cfg.get("head_type", "linear")
        if prior_head != args.head_type:
            raise ValueError(
                f"--resume_from head_type={prior_head!r} but current "
                f"--head_type={args.head_type!r}; warm-start across architectures "
                f"is not supported."
            )
        print(f">> [ID {target_id}] warm start from {args.resume_from}")
        classifier.load_state_dict(prior["state_dict"])

    optimizer = torch.optim.AdamW(classifier.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_steps = args.epochs * max(1, (n_train + args.batch_size - 1) // args.batch_size)
    warmup = min(args.warmup_steps, n_steps // 10)
    lr_min_frac = max(0.0, min(1.0, args.lr_decay_min))

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, n_steps - warmup)
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return lr_min_frac + (1.0 - lr_min_frac) * cos

    print(f">> [ID {target_id}] LR schedule: lr={args.lr}, warmup={warmup}/{n_steps}, "
          f"lr_decay_min={lr_min_frac}")

    log_path = os.path.splitext(save_path)[0] + ".metrics.tsv"
    best_val_f1 = -1.0
    best_state = None
    best_metrics = None
    step = 0
    g_loader = torch.Generator().manual_seed(args.random_seed + target_id)

    with open(log_path, "w", buffering=1) as flog:
        flog.write("epoch\ttrain_loss\tval_loss\tval_acc\tval_prec\tval_rec"
                   "\tval_f1\tval_auroc\tlr\n")
        for epoch in range(args.epochs):
            classifier.train()
            t0 = time.time()
            train_idx_perm = train_idx[torch.randperm(n_train, generator=g_loader)]
            train_loss_sum = 0.0
            train_n = 0
            cur_lr = args.lr
            pbar = tqdm(range(0, n_train, args.batch_size),
                        desc=f"[ID {target_id}] Epoch {epoch + 1}/{args.epochs}",
                        leave=False)
            for start in pbar:
                idx = train_idx_perm[start:start + args.batch_size]
                z = Z[idx].to(device, non_blocking=True)
                y = Y[idx].to(device, non_blocking=True)         # (B,)
                cur_lr = args.lr * lr_at(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
                with torch.no_grad():
                    c = sae.encode(z)
                logits = classifier(c).squeeze(-1)                # (B,)
                loss = bce(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                bs = idx.shape[0]
                train_loss_sum += loss.item() * bs
                train_n += bs
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            train_loss = train_loss_sum / max(train_n, 1)

            # Validation: BCE loss + thresholded metrics
            classifier.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for start in range(0, n_val, args.batch_size):
                    idx = val_idx[start:start + args.batch_size]
                    z = Z[idx].to(device, non_blocking=True)
                    y = Y[idx].to(device, non_blocking=True)
                    c = sae.encode(z)
                    logits = classifier(c).squeeze(-1)
                    vl = bce(logits, y)
                    val_loss_sum += vl.item() * idx.shape[0]
            val_loss = val_loss_sum / max(n_val, 1)

            val_probs = predict_probs(classifier, sae, Z[val_idx], device,
                                      args.batch_size)
            val_true = y_g[val_idx.numpy()]
            m = compute_binary_metrics(val_true, val_probs)

            dt = time.time() - t0
            print(f"   [ID {target_id} | epoch {epoch + 1}] "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={m['acc']:.3f}  val_prec={m['prec']:.3f}  "
                  f"val_rec={m['rec']:.3f}  val_f1={m['f1']:.3f}  "
                  f"val_auroc={m['auroc']:.3f}  ({dt:.1f}s)")
            flog.write(f"{epoch + 1}\t{train_loss:.6e}\t{val_loss:.6e}\t"
                       f"{m['acc']:.6f}\t{m['prec']:.6f}\t{m['rec']:.6f}\t"
                       f"{m['f1']:.6f}\t{m['auroc']:.6f}\t{cur_lr:.6e}\n")

            if m["f1"] > best_val_f1:
                best_val_f1 = m["f1"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in classifier.state_dict().items()}
                best_metrics = m

    classifier.load_state_dict(best_state)
    print_metrics(target_id, best_metrics, label=f"val (best by F1)")

    torch.save({
        "state_dict": classifier.state_dict(),
        "config": {
            "D": D,
            "selected_ids": [target_id],
            "sae_ckpt": args.sae_ckpt,
            "head": f"{args.head_type}_binary",
            "head_type": args.head_type,
            "mlp_hidden_dims": list(args.mlp_hidden_dims),
            "mlp_dropout": args.mlp_dropout,
            "n_groups": args.n_groups,
            "labels_path": args.labels_path,
        },
        "pos_weight": float(pos_weight.item()),
        "val_metrics": best_metrics,
        "args": vars(args),
    }, save_path)
    print(f">> [ID {target_id}] saved -> {save_path}")
    print(f">> [ID {target_id}] metrics -> {log_path}")
    return best_metrics


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)

    # Inputs
    parser.add_argument("--sae_ckpt", type=str,
                        default=f"{base_dir}/checkpoint/topk_sae/topk_sae_alpha4_k1000.pt")
    parser.add_argument("--data_path", type=str,
                        default=f"{base_dir}/data/v1_fragment_ids_correct.smi")
    parser.add_argument("--labels_path", type=str,
                        default=f"{base_dir}/data/v1_fragments_ids_correct_binary.smi")
    parser.add_argument("--cache_dir", type=str,
                        default=f"{base_dir}/checkpoint/sae_cache")
    parser.add_argument("--n_groups", type=int, default=N_GROUPS,
                        help=f"Number of fragment IDs / label columns "
                             f"(default {N_GROUPS}, matching "
                             f"data/fragments_reference.csv).")
    parser.add_argument("--verify_alignment", action="store_true",
                        help="Cross-check the SMILES column of --labels_path "
                             "against --data_path before training. Off by "
                             "default because --data_path may not be a plain "
                             "2-column .smi.")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay_min", type=float, default=0.0,
                        help="Cosine-decay floor as a fraction of --lr. "
                             "0.0 = decay to 0 (default), 1.0 = constant LR.")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--no_pos_weight", action="store_true",
                        help="Disable per-class pos_weight in BCE (uniform loss).")
    parser.add_argument("--pos_weight_cap", type=float, default=20.0,
                        help="Cap pos_weight to avoid extreme upweighting "
                             "when positives are very rare (default 20).")
    parser.add_argument("--group_ids", type=str, default="",
                        help="Comma-separated group IDs to train on (1-indexed). "
                             "'7' (single), '1,7,40' (subset), '1-10' (range), "
                             "'' or 'all' (default = all IDs). One independent "
                             "classifier is trained per ID.")

    # Head architecture
    parser.add_argument("--head_type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="'linear' = nn.Linear(D, 1) -- weighted logistic "
                             "regression (default). 'mlp' = 3-layer MLP head "
                             "with sizes from --mlp_hidden_dims.")
    parser.add_argument("--mlp_hidden_dims", type=str, default="1024,256",
                        help="Comma-separated hidden widths for --head_type mlp "
                             "(default '1024,256' -> Linear(D,1024)->GELU->"
                             "Dropout->Linear(1024,256)->GELU->Dropout->Linear(256,1)).")
    parser.add_argument("--mlp_dropout", type=float, default=0.1,
                        help="Dropout probability between MLP layers.")

    # Outputs / continuation
    parser.add_argument("--save_path", type=str, default=None,
                        help="Only honored with a single Group ID; otherwise "
                             "per-ID auto-naming is used.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Warm-start head weights from a prior ckpt; "
                             "optimizer + LR schedule reset. Single ID only.")

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")

    n_groups = args.n_groups

    # Parse --mlp_hidden_dims into a tuple of ints, validate
    try:
        args.mlp_hidden_dims = tuple(int(x) for x in args.mlp_hidden_dims.split(",") if x.strip())
    except ValueError as e:
        raise ValueError(f"--mlp_hidden_dims must be comma-separated ints, got "
                         f"{args.mlp_hidden_dims!r}: {e}")
    if args.head_type == "mlp" and len(args.mlp_hidden_dims) == 0:
        raise ValueError("--head_type mlp requires --mlp_hidden_dims with >=1 width")

    # Parse --group_ids selection
    sel = args.group_ids.strip().lower()
    if sel in ("", "all"):
        selected_ids = list(range(1, n_groups + 1))
    elif "-" in sel and "," not in sel:
        a, b = sel.split("-", 1)
        selected_ids = list(range(int(a), int(b) + 1))
    else:
        selected_ids = [int(x) for x in sel.split(",") if x]
    for i in selected_ids:
        if not (1 <= i <= n_groups):
            raise ValueError(f"Group ID {i} outside [1, {n_groups}]")
    if len(selected_ids) < n_groups:
        print(f">> Selected {len(selected_ids)} group ID(s): {selected_ids}")
    else:
        print(f">> Training on all {n_groups} group IDs.")

    out_dir = f"{base_dir}/checkpoint/group_binary_classifier"
    os.makedirs(out_dir, exist_ok=True)
    sae_tag = os.path.splitext(os.path.basename(args.sae_ckpt))[0]

    if args.save_path is not None and len(selected_ids) > 1:
        print(f">> WARNING: --save_path is ignored because multiple Group IDs "
              f"were selected; per-ID ckpts will go under {out_dir}.")
        args.save_path = None
    if args.resume_from is not None and len(selected_ids) > 1:
        print(f">> WARNING: --resume_from is only honored when training a "
              f"single Group ID; it will be ignored here.")
        args.resume_from = None

    # Cached embeddings
    tag = hashlib.md5(os.path.abspath(args.data_path).encode()).hexdigest()[:10]
    cache_path = os.path.join(args.cache_dir, f"psmiles_embeddings_{tag}.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cached embeddings at {cache_path}. Run train_topk_sae.py "
            f"--data_path {args.data_path} first."
        )
    print(f">> Load cached embeddings from {cache_path}")
    Z = torch.load(cache_path, map_location="cpu")
    print(f">> Embeddings Z: {tuple(Z.shape)}")

    # SAE
    sae, cfg = load_sae(args.sae_ckpt, device)
    D = cfg["D"]
    assert Z.shape[1] == cfg["d"], f"Z dim {Z.shape[1]} != SAE d {cfg['d']}"

    # Binary labels
    label_smiles, Y_full = load_binary_labels(args.labels_path, n_groups=n_groups)
    assert Y_full.shape == (Z.shape[0], n_groups), (
        f"Labels shape {Y_full.shape} != ({Z.shape[0]}, {n_groups}); "
        f"row order must align."
    )

    if args.verify_alignment:
        data_smiles = read_smiles_column(args.data_path)
        if len(data_smiles) != len(label_smiles):
            raise ValueError(
                f"--verify_alignment: {args.data_path} has {len(data_smiles)} "
                f"rows but {args.labels_path} has {len(label_smiles)}."
            )
        mismatch = [i for i in range(len(label_smiles))
                    if data_smiles[i] != label_smiles[i]]
        if mismatch:
            i = mismatch[0]
            raise ValueError(
                f"--verify_alignment: {len(mismatch)} row(s) disagree. "
                f"First at row {i}: data={data_smiles[i]!r} "
                f"labels={label_smiles[i]!r}"
            )
        print(f">> Alignment verified: {len(label_smiles)} SMILES match "
              f"row-for-row between --data_path and --labels_path")

    # Shared train/val split
    g_split = torch.Generator().manual_seed(args.random_seed)
    M = Z.shape[0]
    n_val = int(round(args.val_frac * M))
    n_train = M - n_val
    perm = torch.randperm(M, generator=g_split)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    print(f">> Train/val split (shared across IDs): {n_train} / {n_val} "
          f"({args.val_frac:.1%})")

    # Train one independent classifier per Group ID
    # Tag non-linear heads so MLP ckpts don't clobber existing linear ones.
    head_tag = "" if args.head_type == "linear" else f"_{args.head_type}"
    summary = []
    for k, target_id in enumerate(selected_ids):
        if len(selected_ids) > 1 or args.save_path is None:
            save_path = f"{out_dir}/group_bin_clf_{sae_tag}_ids_{target_id}{head_tag}.pt"
        else:
            save_path = args.save_path

        y_g = Y_full[:, target_id - 1]
        print(f"\n========================================================")
        print(f">> [{k + 1}/{len(selected_ids)}] Training binary classifier for Group ID {target_id}")
        print(f"========================================================")
        m = train_one_group(target_id, args, sae, D, Z, y_g,
                            train_idx, val_idx, save_path, device, base_dir)
        summary.append({"id": target_id, **m, "save_path": save_path})

    if len(selected_ids) > 1:
        print("\n>> Per-ID validation summary (best epoch per ID, by F1):")
        print(f"   {'ID':>3}  {'pos':>7}  {'acc':>6}  {'prec':>6}  {'rec':>6}  "
              f"{'f1':>6}  {'auroc':>6}  ckpt")
        for r in summary:
            print(f"   {r['id']:>3}  {r['pos_rate']:>7.3f}  {r['acc']:>6.3f}  "
                  f"{r['prec']:>6.3f}  {r['rec']:>6.3f}  {r['f1']:>6.3f}  "
                  f"{r['auroc']:>6.3f}  {r['save_path']}")


if __name__ == "__main__":
    main()