"""
Train a Top-K Sparse Autoencoder on pSMILES embeddings z in R^d (d=768).

Pipeline (Section 2.1 of Understanding_Chemical_LMM_via_molar_groups.pdf):
  1. Run pretrained CLLM pipeline (smi-ted + star encoder + autoencoder.encoder)
     over an M-row SMILES dataset to produce Z in R^{M x 768}.  Cached to disk.
  2. Train TopKSAE:  z_hat = W_dec TopK(W_enc z + b_enc) + b_dec
     minimizing L_SAE = ||z - z_hat||_2^2.
  3. Report reconstruction loss, fraction-alive, and (optional) pSMILES accuracy
     through the frozen decoder + language-model head.
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import (
    AutoEncoderLayer3,
    Decoder2,
    TopKSAE,
    prediction_Model,
    star_encoder,
)
from utils1 import StringDataset, count_params, load_smi_ted_explicit


@torch.no_grad()
def extract_psmiles_embeddings(smiles_list, smi_ted, A_encoder, autoencoder, device,
                               batch_size=128, return_idx=False):
    """Run frozen CLLM pipeline and return z in R^{M x 768} (and token idx if asked)."""
    loader = DataLoader(StringDataset(smiles_list), batch_size=batch_size, shuffle=False)
    zs, idxs = [], []
    for smiles_batch in tqdm(loader, desc="Embedding"):
        idx_t, token_emb_t, _ = smi_ted.extract_embeddings(smiles_batch)
        idx_t, token_emb_t = idx_t.to(device), token_emb_t.to(device)

        mask_439 = (idx_t == 439)
        pos = torch.nonzero(mask_439, as_tuple=False)
        if pos.numel() > 0:
            star_emb = A_encoder(idx_t[pos[:, 0], pos[:, 1]].view(-1, 1).float())
            token_emb_t[pos[:, 0], pos[:, 1], :] = star_emb

        z = autoencoder.encoder(token_emb_t.view(-1, 202 * 768))  # [B, 768]
        zs.append(z.cpu())
        if return_idx:
            idxs.append(idx_t.cpu())

    Z = torch.cat(zs, dim=0)
    if return_idx:
        return Z, torch.cat(idxs, dim=0)
    return Z


## helper function to load SMILES from a file, ignoring multiple columns 
def load_smiles_list(data_path):
    """Load SMILES from a file, ignoring multiple columns."""
    smiles_list = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smiles_list.append(line.split()[0])
    return smiles_list

def load_or_cache_embeddings(args, device):
    """Cache psmiles embeddings keyed by data_path + size, re-use across SAE sweeps."""
    tag = hashlib.md5(os.path.abspath(args.data_path).encode()).hexdigest()[:10]
    cache_path = os.path.join(args.cache_dir, f"psmiles_embeddings_{tag}.pt")
    if os.path.exists(cache_path) and not args.force_recompute:
        print(f">> Load cached embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f">> No cache at {cache_path}, extracting embeddings.")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(">> Load pretrained smi-ted.")
    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    tokenizer = MolTranBertTokenizer(f"{base_dir}/smi_ted_light/bert_vocab_curated.txt")
    smi_ted = Smi_ted(tokenizer)
    smi_ted.load_checkpoint(f"{base_dir}/smi_ted_light/smi-ted-Light_40.pt")
    smi_ted.eval()

    print(">> Load star encoder.")
    A_encoder = star_encoder(dim=1).to(device).eval()

    print(">> Load autoencoder.encoder from checkpoint.")
    autoencoder = AutoEncoderLayer3(
        feature_size=202 * 768, mid_size=768 * 4, mid_size_2=768 * 2, latent_size=768
    ).to(device).eval()
    ckpt = torch.load(args.cllm_ckpt, map_location=device, weights_only=False)
    autoencoder.encoder.load_state_dict(ckpt["autoencoder_encoder"])
    A_encoder.load_state_dict(ckpt["star_encoder"]) # Load star encoder weights from the same checkpoint change made for accuracy by diya 

    # df = pd.read_csv(
    #     args.data_path, header=None, delimiter=" ",
    #     names=["smiles_canonicalized", "Group_IDs"],
    # )
    # smiles_list = df["smiles_canonicalized"].tolist()
    smiles_list = load_smiles_list(args.data_path)
    print(f">> {len(smiles_list)} SMILES loaded from {args.data_path}")

    Z = extract_psmiles_embeddings(
        smiles_list, smi_ted, A_encoder, autoencoder, device,
        batch_size=args.batch_size,
    )
    os.makedirs(args.cache_dir, exist_ok=True)
    torch.save(Z, cache_path)
    print(f">> Cached embeddings {tuple(Z.shape)} to {cache_path}")
    return Z


def train_sae(Z, args, device):
    M, d = Z.shape
    assert d == 768, f"Expected d=768 embeddings, got {d}"

    # No train/val split — train and validate on the full embedding set.
    full_ds = TensorDataset(Z)
    train_loader = DataLoader(full_ds, batch_size=args.sae_batch_size, shuffle=True)
    val_loader = DataLoader(full_ds, batch_size=args.sae_batch_size, shuffle=False)

    sae = TopKSAE(d=d, expansion_factor=args.expansion_factor, k=args.k).to(device)
    print(f">> TopKSAE: d={d}, D={sae.D} (alpha={args.expansion_factor}), K={args.k}, "
          f"params={count_params(sae)}")

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    n_steps = args.epochs * max(1, len(train_loader))
    warmup = min(args.warmup_steps, n_steps // 10)

    lr_min_frac = max(0.0, min(1.0, args.lr_decay_min))

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, n_steps - warmup)
        cos = 0.5 * (1 + np.cos(np.pi * progress))  # 1 -> 0 over remaining steps
        return lr_min_frac + (1.0 - lr_min_frac) * cos

    best_val = float("inf")
    best_state = None
    best_l0 = float("nan")
    step = 0
    # fires_ever: [D], True if feature ever fired anywhere in training set
    fires_ever = torch.zeros(sae.D, dtype=torch.bool, device=device)

    # Per-epoch metrics file: <save_path>.metrics.tsv (sibling of the ckpt).
    metrics_path = os.path.splitext(args.save_path)[0] + ".metrics.tsv"
    metrics_f = open(metrics_path, "w", buffering=1)  # line-buffered for live tailing
    metrics_f.write("epoch\ttrain_mse\tval_mse\tL0\talive_train\talive_val\tlr\n")

    for epoch in range(args.epochs):
        sae.train()
        train_loss_sum, train_n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for (z_batch,) in pbar:
            z_batch = z_batch.to(device)
            for g_ in optimizer.param_groups:
                g_["lr"] = args.lr * lr_at(step)

            z_hat, c = sae(z_batch, output_features=True)
            loss = F.mse_loss(z_hat, z_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder_()

            with torch.no_grad():
                fires_ever |= (c != 0).any(dim=0)

            bs = z_batch.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs
            step += 1
            pbar.set_postfix(mse=f"{loss.item():.3e}")

        train_mse = train_loss_sum / max(1, train_n)

        val_mse, val_fires, val_l0 = evaluate(sae, val_loader, device)
        frac_alive_val = val_fires.float().mean().item()
        frac_alive_train = fires_ever.float().mean().item()
        cur_lr = args.lr * lr_at(step - 1) if step > 0 else 0.0
        print(f"   [epoch {epoch + 1}] train_mse={train_mse:.3e}  val_mse={val_mse:.3e}  "
              f"alive(train)={frac_alive_train:.3f}  alive(val)={frac_alive_val:.3f}  "
              f"L0={val_l0:.2f}")
        metrics_f.write(
            f"{epoch + 1}\t{train_mse:.6e}\t{val_mse:.6e}\t{val_l0:.4f}\t"
            f"{frac_alive_train:.4f}\t{frac_alive_val:.4f}\t{cur_lr:.6e}\n"
        )

        if val_mse < best_val:
            best_val = val_mse
            best_l0 = val_l0
            best_state = {k: v.detach().cpu().clone() for k, v in sae.state_dict().items()}

    metrics_f.close()
    print(f">> Per-epoch metrics: {metrics_path}")

    if best_state is not None:
        sae.load_state_dict(best_state)
    return sae, best_val, fires_ever.float().mean().item(), best_l0


@torch.no_grad()
def evaluate(sae, loader, device):
    """Returns (mean_mse, fires_mask, mean_L0) over the loader.

    L0 = average number of non-zero features per sample (||c||_0 averaged over
    samples). Upper-bounded by K but can be smaller when fewer than K
    pre-activations survive the ReLU inside TopKSAE.
    """
    sae.eval()
    mse_sum, n = 0.0, 0
    fires = torch.zeros(sae.D, dtype=torch.bool, device=device)
    l0_sum = 0.0
    for (z_batch,) in loader:
        z_batch = z_batch.to(device)
        z_hat, c = sae(z_batch, output_features=True)
        bs = z_batch.size(0)
        mse_sum += F.mse_loss(z_hat, z_batch).item() * bs
        n += bs
        fires |= (c != 0).any(dim=0)
        l0_sum += (c != 0).sum().item()
    return mse_sum / max(1, n), fires, l0_sum / max(1, n)


@torch.no_grad()
def evaluate_psmiles_accuracy(sae, Z, args, device):
    """pSMILES reconstruction accuracy: a sample is 'correct' only if ALL 202
    predicted tokens match the original tokenization exactly."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    decoder = Decoder2(feature_size=202 * 768, mid_size=768 * 4,
                       mid_size_2=768 * 2, latent_size=768).to(device).eval()
    lang_model = prediction_Model(n_embd=768, mid_size=768 * 4, n_vocab=2393).to(device).eval()
    ckpt = torch.load(args.cllm_ckpt, map_location=device, weights_only=False)
    decoder.load_state_dict(ckpt["decoder"])
    lang_model.load_state_dict(ckpt["predictor"])
    #del ckpt # added diya for manage memory

    # df = pd.read_csv(
    #     args.data_path, header=None, delimiter=" ",
    #     names=["smiles_canonicalized", "Group_IDs"],
    # )
    # smiles_list = df["smiles_canonicalized"].tolist()

    smiles_list = load_smiles_list(args.data_path)

    print(">> Re-tokenizing to compute pSMILES accuracy against SAE reconstructions.")
    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    tokenizer = MolTranBertTokenizer(f"{base_dir}/smi_ted_light/bert_vocab_curated.txt")
    smi_ted = Smi_ted(tokenizer)
    smi_ted.load_checkpoint(f"{base_dir}/smi_ted_light/smi-ted-Light_40.pt")
    smi_ted.eval()

    exact, total = 0, 0
    sae.eval()
    for i in tqdm(range(0, len(smiles_list), args.batch_size), desc="pSMILES acc"):
        batch = smiles_list[i:i + args.batch_size]
        idx_t, _, _ = smi_ted.extract_embeddings(batch)
        idx_t = idx_t.to(device)

        z = Z[i:i + args.batch_size].to(device)
        z_hat = sae(z)
        token_emb_rec = decoder(z_hat).view(-1, 202, 768)
        logits = lang_model(token_emb_rec.view(-1, 768))
        pred_idx = logits.view(-1, 202, 2393).argmax(-1)

        exact += torch.all(pred_idx == idx_t, dim=1).sum().item()
        total += idx_t.size(0)

    return exact / total


def main():
    parser = argparse.ArgumentParser(description="Train a Top-K SAE on pSMILES embeddings")

    # Top-K SAE hyperparameters
    parser.add_argument("--expansion_factor", type=int, default=8,
                        help="alpha in D = alpha * 768. Sweep: {4, 8, 16, 32}")
    parser.add_argument("--k", type=int, default=80,
                        help="K in TopK. Sweep: {40, 80, 100, 160, 200, 250, 500, 1000, 2000}")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--sae_batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_decay_min", type=float, default=0.0,
                        help="Cosine-decay floor as a fraction of --lr. "
                             "0.0=decay to 0 (default), 1.0=constant after warmup.")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=42)

    # Embedding extraction
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for CLLM embedding extraction")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--cllm_ckpt", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--force_recompute", action="store_true")

    # Output + optional eval
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--eval_psmiles", action="store_true",
                        help="Also compute pSMILES reconstruction accuracy.")

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_path is None:
        # args.data_path = f"{base_dir}/data/homo_groups_1.6M.smi"
        args.data_path = f"{base_dir}/data/PSMILES.smi"
    if args.cllm_ckpt is None:
        args.cllm_ckpt = (f"{base_dir}/checkpoint/"
                          f"poly4mer_v3.0.0.ckpt")
    if args.cache_dir is None:
        args.cache_dir = f"{base_dir}/checkpoint/sae_cache"
    if args.save_path is None:
        out_dir = f"{base_dir}/checkpoint/topk_sae"
        os.makedirs(out_dir, exist_ok=True)
        args.save_path = (f"{out_dir}/topk_sae_alpha{args.expansion_factor}_k{args.k}.pt")

    Z = load_or_cache_embeddings(args, device)
    print(f">> Embeddings Z: {tuple(Z.shape)}  "
          f"mean_norm={Z.norm(dim=1).mean().item():.3f}")

    sae, best_val, frac_alive, best_l0 = train_sae(Z, args, device)
    print(f"\n>> Best val MSE: {best_val:.3e}   fraction-alive (train): {frac_alive:.3f}   "
          f"L0: {best_l0:.2f}")

    torch.save(
        {
            "state_dict": sae.state_dict(),
            "config": {
                "d": sae.d, "D": sae.D,
                "expansion_factor": args.expansion_factor,
                "k": args.k,
            },
            "best_val_mse": best_val,
            "fraction_alive_train": frac_alive,
            "L0": best_l0,
        },
        args.save_path,
    )
    print(f">> Saved SAE to {args.save_path}")

    if args.eval_psmiles:
        exact_acc = evaluate_psmiles_accuracy(sae, Z, args, device)
        print(f">> pSMILES reconstruction accuracy: {exact_acc * 100:.3f}%")


if __name__ == "__main__":
    main()
