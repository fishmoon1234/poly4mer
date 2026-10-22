"""
Standalone eval: compute pSMILES reconstruction accuracy through a trained Top-K SAE,
reusing cached CLLM embeddings and poly4mer_v3.0.0.ckpt

No training. Expects:
  --sae_ckpt : a checkpoint saved by train_topk_sae.py (contains state_dict + config)
  --data_path: the same SMILES file the SAE was trained on (so the md5-keyed
               embedding cache resolves, and smi-ted re-tokenization aligns 1:1
               with the cached Z rows).
"""

import argparse
import hashlib
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from models import TopKSAE
from train_topk_sae import evaluate, evaluate_psmiles_accuracy


def main(): 
    parser = argparse.ArgumentParser(description="Eval pSMILES accuracy of a trained Top-K SAE")
    parser.add_argument("--sae_ckpt", type=str, required=True,
                        help="Path to SAE checkpoint (produced by train_topk_sae.py)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="SMILES file the SAE was trained on (must match cache key)")
    parser.add_argument("--cllm_ckpt", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_path is None:
        args.data_path = f"{base_dir}/data/PSMILES.smi"
    if args.cllm_ckpt is None:
        args.cllm_ckpt = (f"{base_dir}/checkpoint/poly4mer_v3.0.0.ckpt")
    if args.cache_dir is None:
        args.cache_dir = f"{base_dir}/checkpoint/sae_cache"

    tag = hashlib.md5(os.path.abspath(args.data_path).encode()).hexdigest()[:10]
    cache_path = os.path.join(args.cache_dir, f"psmiles_embeddings_{tag}.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cached embeddings at {cache_path}. Run train_topk_sae.py "
            f"--data_path {args.data_path} first to populate the cache."
        )
    print(f">> Load cached embeddings from {cache_path}")
    Z = torch.load(cache_path, map_location="cpu")
    print(f">> Embeddings Z: {tuple(Z.shape)}")

    print(f">> Load SAE checkpoint from {args.sae_ckpt}")
    ckpt = torch.load(args.sae_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(d=cfg["d"], expansion_factor=cfg["expansion_factor"], k=cfg["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    print(f"   d={cfg['d']}, D={cfg['D']}, alpha={cfg['expansion_factor']}, K={cfg['k']}")
    if "best_val_mse" in ckpt:
        print(f"   best_val_mse(train): {ckpt['best_val_mse']:.3e}")
    # del ckpt 
    # torch.cuda.empty_cache()
    assert Z.shape[0] > 0
    assert Z.shape[1] == cfg["d"], f"Embedding dim {Z.shape[1]} != SAE d {cfg['d']}"

    z_loader = DataLoader(TensorDataset(Z), batch_size=4096, shuffle=False) ## changed batch size  from 4096 diya
    mse, fires, l0 = evaluate(sae, z_loader, device)
    print(f">> SAE MSE: {mse:.3e}   fraction-alive: {fires.float().mean().item():.3f}   "
          f"L0: {l0:.2f}")

    acc = evaluate_psmiles_accuracy(sae, Z, args, device)
    print(f"\n>> pSMILES reconstruction accuracy (through SAE): "
          f"{acc * 100:.3f}%")


if __name__ == "__main__":
    main() 
