"""
dCE, L_clean, L_sae, L_zero, and loss recovered for trained TopK SAEs (eval only).

Pipeline per molecule i:
    z_i -> Decoder2 -> prediction_Model (LM head) -> logits [202, 2393]
    targets = smi-ted token ids x_i [202]

Cross-entropy uses torch.nn.CrossEntropyLoss(reduction="sum"), summed over the FULL dataset
and divided by total tokens (M * 202) once at the end:

    CE(z')         = sum_{i,t} -log softmax(logits(z'_i)_t)[x_it] / (M * 202)
    dCE            = CE(SAE(z)) - CE(z)
    loss_recovered = 1 - (CE(SAE(z)) - CE(z)) / (CE(0) - CE(z))

Usage:
    python eval_sae_ce.py --sae_ckpt checkpoint/topk_sae/topk_sae_alpha*_k*.pt
"""

import argparse
import hashlib
import os
import re

import torch
import torch.nn as nn
from tqdm import tqdm

from models import Decoder2, TopKSAE, prediction_Model
from utils1 import load_smi_ted_explicit
from train_topk_sae import load_or_cache_embeddings, load_smiles_list

SEQ, EMB, VOCAB = 202, 768, 2393


def load_or_cache_tokens(smiles, args, base_dir):
    """Token ids [M, 202] from smi-ted, cached next to the embedding cache (same md5 key)."""
    tag = hashlib.md5(os.path.abspath(args.data_path).encode()).hexdigest()[:10]
    path = os.path.join(args.cache_dir, f"psmiles_token_ids_{tag}.pt")
    if os.path.exists(path) and not args.force_recompute:
        print(f">> Load cached token ids from {path}")
        return torch.load(path, map_location="cpu")

    print(f">> No token cache at {path}, tokenizing with smi-ted (one time).")
    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    tokenizer = MolTranBertTokenizer(f"{base_dir}/smi_ted_light/bert_vocab_curated.txt")
    smi_ted = Smi_ted(tokenizer)
    smi_ted.load_checkpoint(f"{base_dir}/smi_ted_light/smi-ted-Light_40.pt")
    smi_ted.eval()

    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles), args.tok_batch_size), desc="Tokenize"):
            idx_t, _, _ = smi_ted.extract_embeddings(smiles[i:i + args.tok_batch_size])
            out.append(idx_t.cpu().to(torch.int16))       # vocab 2393 fits in int16
    idx = torch.cat(out, dim=0)

    os.makedirs(args.cache_dir, exist_ok=True)
    torch.save(idx, path + ".tmp")
    os.replace(path + ".tmp", path)                       # no half-written cache file
    print(f">> Cached token ids {tuple(idx.shape)} to {path}")
    return idx


def load_sae(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sae = TopKSAE(d=cfg["d"], expansion_factor=cfg["expansion_factor"], k=cfg["k"])
    sae.load_state_dict(ck["state_dict"])
    return sae.to(device).eval(), cfg


def alpha_k_key(p):
    m = re.search(r"alpha_?([\d.]+)_k_?(\d+)", os.path.basename(p))
    return (float(m.group(1)), int(m.group(2))) if m else (float("inf"), 0)


@torch.no_grad()
def ce_pass(latent_fn, Z, idx, decoder, lang, ce_loss, bs, device, desc):
    """Returns total summed CE over all token positions for the dataset."""
    total_ce = 0.0
    for i in tqdm(range(0, Z.size(0), bs), desc=desc):
        z = Z[i:i + bs].to(device)
        target = idx[i:i + bs].to(device).long().reshape(-1)                  # [B*202]
        logits = lang(decoder(latent_fn(z)).view(-1, EMB)).float()            # [B*202, 2393]
        total_ce += ce_loss(logits, target).item()
    return total_ce


def main():
    ap = argparse.ArgumentParser(description="dCE, L_clean, L_sae, L_zero & loss recovered for TopK SAEs")
    ap.add_argument("--sae_ckpt", type=str, nargs="+", required=True)
    ap.add_argument("--data_path", type=str, default=None,
                    help="Must be the same path string used for training (caches are keyed on it).")
    ap.add_argument("--cllm_ckpt", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--force_recompute", action="store_true")
    ap.add_argument("--batch_size", type=int, default=256, help="Molecules per decode batch.")
    ap.add_argument("--tok_batch_size", type=int, default=128, help="Batch size for smi-ted tokenization.")
    ap.add_argument("--out_tsv", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_path is None:
        args.data_path = f"{base_dir}/data/PSMILES.smi"
    if args.cllm_ckpt is None:
        args.cllm_ckpt = f"{base_dir}/checkpoint/poly4mer_v3.0.0.ckpt"
    if args.cache_dir is None:
        args.cache_dir = f"{base_dir}/checkpoint/sae_cache"
    if args.out_tsv is None:
        args.out_tsv = f"{base_dir}/checkpoint/topk_sae/sae_ce_metrics.tsv"

    # ---- data: cached z + token-id targets
    smiles = load_smiles_list(args.data_path)
    Z = load_or_cache_embeddings(args, device).float()
    idx = load_or_cache_tokens(smiles, args, base_dir)
    assert Z.shape[0] == len(smiles) == idx.shape[0], \
        f"Row mismatch: {len(smiles)} SMILES, {Z.shape[0]} z rows, {idx.shape[0]} token rows"
    assert idx.shape[1] == SEQ, f"Expected {SEQ} tokens per molecule, got {idx.shape[1]}"

    n_all = idx.numel()
    print(f">> {Z.shape[0]} molecules | total tokens evaluated: {n_all}")

    # ---- frozen decoder + LM head
    ck = torch.load(args.cllm_ckpt, map_location="cpu", weights_only=False)
    decoder = Decoder2(feature_size=SEQ * EMB, mid_size=EMB * 4, mid_size_2=EMB * 2, latent_size=EMB)
    decoder.load_state_dict(ck["decoder"])
    lang = prediction_Model(n_embd=EMB, mid_size=EMB * 4, n_vocab=VOCAB)
    lang.load_state_dict(ck["predictor"])
    del ck
    decoder, lang = decoder.to(device).eval(), lang.to(device).eval()

    ce_loss = nn.CrossEntropyLoss(reduction="sum")
    run = lambda fn, desc: ce_pass(fn, Z, idx, decoder, lang, ce_loss, args.batch_size, device, desc)

    # ---- baselines, computed once and shared by every SAE
    l_clean = run(lambda z: z, "CE clean") / n_all
    l_zero = run(torch.zeros_like, "CE zero-ablation") / n_all
    print(f">> L_clean: {l_clean:.6f} | L_zero: {l_zero:.6f}")

    # ---- one SAE at a time; rows written as each finishes
    os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)), exist_ok=True)
    with open(args.out_tsv, "w", buffering=1) as f:
        f.write("alpha\tk\tL_clean\tL_sae\tL_zero\tdCE\tloss_rec\n")
        for p in sorted(args.sae_ckpt, key=alpha_k_key):
            sae, cfg = load_sae(p, device)
            
            l_sae = run(sae, f"alpha={cfg['expansion_factor']} k={cfg['k']}") / n_all
            dce = l_sae - l_clean
            
            denom = l_zero - l_clean
            loss_rec = 1.0 - dce / (denom if abs(denom) > 1e-8 else 1e-8)

            print(f"   alpha={cfg['expansion_factor']:<3} k={cfg['k']:<5} "
                  f"L_clean={l_clean:.6f} L_sae={l_sae:.6f} L_zero={l_zero:.6f} "
                  f"dCE={dce:+.4e} loss_rec={loss_rec:.4f}")
            
            f.write(f"{cfg['expansion_factor']}\t{cfg['k']}\t"
                    f"{l_clean:.6e}\t{l_sae:.6e}\t{l_zero:.6e}\t"
                    f"{dce:.6e}\t{loss_rec:.6e}\n")

            del sae
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f">> Wrote {args.out_tsv}")


if __name__ == "__main__":
    main()