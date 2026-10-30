"""
Batch zero-ablation sweep for one fragment group ID.

For each pSMILES that carries the target fragment:
  1. encode  -> z -> c  (SAE code, |c|_0 <= K)
  2. shortlist = top-5 of (Borda set  n  active features), ranked by activation
  3. decode the un-ablated code as a baseline row
  4. zero ONE shortlisted feature at a time, decode, record the new pSMILES

Rows are appended to a single CSV per group ID:
    row_idx, smiles, n_active, n_intersect, rank, feature_id,
    ablated_psmiles, valid_smiles, exact_match_to_original

The baseline row is rank=0 / feature_id=-1: it is the SAE round-trip with no
ablation, so anything that changes there is reconstruction error, not the
ablation.

Example
-------
python zero_ablate_sweep_pipeline.py \
    --group_id 96 \
    --sae_ckpt checkpoint/topk_sae/topk_sae_alpha4_k1000.pt \
    --borda_json checkpoint/select_features_binary_pipeline_ID/target_id_96/validation.json \
    --limit 100
"""

import argparse
import csv
import json
import os

import torch
from rdkit import Chem, RDLogger

from models import (
    AutoEncoderLayer3,
    Decoder2,
    TopKSAE,
    prediction_Model,
    star_encoder,
)
from utils1 import load_smi_ted_explicit

RDLogger.DisableLog("rdApp.*")

N_FRAGMENT_IDS = 196


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_positive_rows(binary_smi, group_id, limit):
    """Rows of *_binary.smi whose column for `group_id` is 1.

    File layout: smiles + 196 binary labels, whitespace separated.
    Column index for fragment id g is exactly g (col 0 is the SMILES).
    """
    rows = []
    with open(binary_smi) as f:
        for i, line in enumerate(f):
            tok = line.split()
            if not tok:
                continue
            if len(tok) != N_FRAGMENT_IDS + 1:
                raise ValueError(
                    f"{binary_smi}:{i + 1} has {len(tok)} columns, expected "
                    f"{N_FRAGMENT_IDS + 1} (smiles + {N_FRAGMENT_IDS} labels). "
                    "Wrong file?"
                )
            if tok[group_id] == "1":
                rows.append((i, tok[0]))
                if limit and len(rows) >= limit:
                    break
    return rows


def load_borda_set(args):
    """Borda selection for the group, optionally intersected with LASSO(lambda)."""
    if args.borda_indices:
        borda = [int(x) for x in args.borda_indices.split(",") if x.strip()]
        src = "--borda_indices"
    else:
        with open(args.borda_json) as f:
            obj = json.load(f)
        borda = None
        for key in ("selected_feature_indices", "borda_selected"):
            if isinstance(obj.get(key), (list, tuple)):
                borda = list(obj[key])
                break
        if borda is None:
            raise ValueError(
                f"No top-level 'selected_feature_indices' in {args.borda_json}"
            )
        src = f"{args.borda_json}:selected_feature_indices"
        if args.borda_top_n > 0:
            borda = borda[:args.borda_top_n]

        if args.lasso_lambda is not None:
            lasso = obj.get("lasso", {})
            if args.lasso_lambda not in lasso:
                raise ValueError(
                    f"lambda {args.lasso_lambda!r} not in lasso block. "
                    f"Available: {sorted(lasso.keys(), key=float)}"
                )
            lasso_set = set(lasso[args.lasso_lambda]["selected_feature_indices"])
            before = len(borda)
            borda = [j for j in borda if j in lasso_set]
            src += f" n lasso@lambda={args.lasso_lambda} ({before} -> {len(borda)})"

    if not borda:
        raise ValueError("Borda set is empty after filtering.")
    return set(borda), src


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_to_z(smi_ted, A_encoder, autoencoder, smi, device):
    idx_t, token_emb_t, _ = smi_ted.extract_embeddings([smi])
    idx_t = idx_t.to(device)
    token_emb_t = token_emb_t.to(device)
    pos = torch.nonzero(idx_t == 439, as_tuple=False)
    if pos.numel() > 0:
        star_emb = A_encoder(idx_t[pos[:, 0], pos[:, 1]].view(-1, 1).float())
        token_emb_t[pos[:, 0], pos[:, 1], :] = star_emb
    return autoencoder.encoder(token_emb_t.view(-1, 202 * 768))


@torch.no_grad()
def decode_z_batch(decoder2, lang_model, tokenizer, Z):
    """Z in R^{n x 768} -> list of n SMILES strings."""
    token_emb_rec = decoder2(Z).view(-1, 202, 768)
    logits = lang_model(token_emb_rec.view(-1, 768))
    pred_idx = logits.view(-1, 202, 2393).argmax(-1).cpu().numpy()
    return [tokens_to_smiles(row, tokenizer) for row in pred_idx]


def tokens_to_smiles(token_ids, tokenizer):
    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.vocab.get("[PAD]", 0)
    cls_id = getattr(tokenizer, "cls_token_id", None) or tokenizer.vocab.get("[CLS]", -1)
    sep_id = getattr(tokenizer, "sep_token_id", None) or tokenizer.vocab.get("[SEP]", -1)
    skip = {pad_id, cls_id, sep_id}
    pieces = []
    for tid in token_ids:
        tid = int(tid)
        if tid in skip:
            if tid == sep_id:
                break
            continue
        if hasattr(tokenizer, "ids_to_tokens"):
            pieces.append(tokenizer.ids_to_tokens.get(tid, ""))
        elif hasattr(tokenizer, "convert_ids_to_tokens"):
            pieces.append(tokenizer.convert_ids_to_tokens(tid))
    return "".join(pieces)


def load_models(args, device):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    tokenizer = MolTranBertTokenizer(f"{base_dir}/smi_ted_light/bert_vocab_curated.txt")
    smi_ted = Smi_ted(tokenizer)
    smi_ted.load_checkpoint(f"{base_dir}/smi_ted_light/smi-ted-Light_40.pt")
    smi_ted.eval()

    # map_location to CPU storage first, then .to(device) -- avoids the OOM from
    # deserializing CUDA-tagged tensors while other weights are already resident.
    cckpt = torch.load(args.cllm_ckpt, map_location=lambda s, l: s, weights_only=False)

    A_encoder = star_encoder(dim=1)
    A_encoder.load_state_dict(cckpt["star_encoder"])   # <-- REQUIRED. see note below.
    A_encoder = A_encoder.to(device).eval()

    autoencoder = AutoEncoderLayer3(202 * 768, 768 * 4, 768 * 2, 768)
    autoencoder.encoder.load_state_dict(cckpt["autoencoder_encoder"])
    autoencoder = autoencoder.to(device).eval()

    decoder2 = Decoder2(202 * 768, 768 * 4, 768 * 2, 768)
    decoder2.load_state_dict(cckpt["decoder"])
    decoder2 = decoder2.to(device).eval()

    lang_model = prediction_Model(n_embd=768, mid_size=768 * 4, n_vocab=2393)
    lang_model.load_state_dict(cckpt["predictor"])
    lang_model = lang_model.to(device).eval()

    sckpt = torch.load(args.sae_ckpt, map_location=lambda s, l: s, weights_only=False)
    scfg = sckpt["config"]
    sae = TopKSAE(d=scfg["d"], expansion_factor=scfg["expansion_factor"], k=scfg["k"])
    sae.load_state_dict(sckpt["state_dict"])
    sae = sae.to(device).eval()

    for m in (A_encoder, autoencoder, decoder2, lang_model, sae):
        for p in m.parameters():
            p.requires_grad_(False)

    return smi_ted, tokenizer, A_encoder, autoencoder, decoder2, lang_model, sae, scfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--group_id", type=int, required=True,
                    help=f"Fragment ID, 1..{N_FRAGMENT_IDS} (e.g. 3 = chloro)")
    ap.add_argument("--binary_smi", type=str,
                    default=f"{base_dir}/data/v1_fragments_ids_correct_binary.smi")
    ap.add_argument("--limit", type=int, default=100,
                    help="Number of positive pSMILES to process. 0 = all.")

    # Feature shortlist
    ap.add_argument("--borda_json", type=str, default=None,
                    help="validation_<...>.json for this group id.")
    ap.add_argument("--borda_indices", type=str, default=None,
                    help="Comma-separated Borda indices, bypasses --borda_json.")
    ap.add_argument("--borda_top_n", type=int, default=30,
                    help="Truncate the Borda list to the first N. 0 = keep all.")
    ap.add_argument("--lasso_lambda", type=str, default=None,
                    help="If set, also intersect Borda with lasso.<lambda>."
                         "selected_feature_indices from the same JSON.")
    ap.add_argument("--top_k_features", type=int, default=5,
                    help="How many shortlisted features to ablate per molecule.")

    # Checkpoints / output
    ap.add_argument("--sae_ckpt", type=str, required=True)
    ap.add_argument("--cllm_ckpt", type=str,
                    default=f"{base_dir}/checkpoint/poly4mer_v3.0.0.ckpt")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Default: results/zero_ablation/ablation_group<ID>.csv")

    args = ap.parse_args()
    if not (1 <= args.group_id <= N_FRAGMENT_IDS):
        raise ValueError(f"--group_id must be in 1..{N_FRAGMENT_IDS}")
    if (args.borda_json is None) == (args.borda_indices is None):
        raise ValueError("Pass exactly one of --borda_json / --borda_indices.")
    if args.lasso_lambda is not None and args.borda_json is None:
        raise ValueError("--lasso_lambda needs --borda_json.")

    if args.out_csv is None:
        out_dir = f"{base_dir}/results/zero_ablation"
        os.makedirs(out_dir, exist_ok=True)
        args.out_csv = f"{out_dir}/ablation_group{args.group_id}.csv"
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible. This script is GPU-only.")
    device = torch.device("cuda")
    print(f">> Device: {device} ({torch.cuda.get_device_name(0)})")

    borda_set, borda_src = load_borda_set(args)
    print(f">> Borda set: {len(borda_set)} features  [{borda_src}]")

    rows = load_positive_rows(args.binary_smi, args.group_id, args.limit)
    print(f">> Positives for id {args.group_id}: {len(rows)} "
          f"(from {args.binary_smi})")
    if not rows:
        raise ValueError(f"No rows with label 1 for id {args.group_id}.")

    (smi_ted, tokenizer, A_encoder, autoencoder,
     decoder2, lang_model, sae, scfg) = load_models(args, device)
    D = scfg["D"]
    print(f">> SAE: D={D}, K={scfg['k']}, alpha={scfg['expansion_factor']} "
          f"({args.sae_ckpt})")
    bad = [j for j in borda_set if not 0 <= j < D]
    if bad:
        raise ValueError(f"Borda indices outside [0,{D}): {bad[:5]}")

    fields = ["row_idx", "smiles", "n_active", "n_intersect", "rank",
              "feature_id", "ablated_psmiles", "valid_smiles",
              "exact_match_to_original"]
    f_out = open(args.out_csv, "w", newline="", buffering=1)
    writer = csv.DictWriter(f_out, fieldnames=fields)
    writer.writeheader()

    n_done = n_failed = 0
    for n, (row_idx, smi) in enumerate(rows, start=1):
        try:
            z = encode_to_z(smi_ted, A_encoder, autoencoder, smi, device)
            c = sae.encode(z)                                  # [1, D]
            c_flat = c.squeeze(0)

            active = torch.nonzero(c_flat > 0).squeeze(-1).tolist()
            inter = [j for j in active if j in borda_set]
            inter.sort(key=lambda j: -c_flat[j].item())
            shortlist = inter[:args.top_k_features]

            # batch: row 0 = baseline (no ablation), rows 1.. = one feature each
            codes = [c_flat]
            for fid in shortlist:
                c_ab = c_flat.clone()
                c_ab[fid] = 0.0
                codes.append(c_ab)
            C = torch.stack(codes, dim=0)                      # [1+len, D]
            Z = sae.decode(C)
            out_smiles = decode_z_batch(decoder2, lang_model, tokenizer, Z)

            for rank, (fid, new_smi) in enumerate(
                    zip([-1] + shortlist, out_smiles)):
                writer.writerow({
                    "row_idx": row_idx,
                    "smiles": smi,
                    "n_active": len(active),
                    "n_intersect": len(inter),                        # 0 = baseline
                    "feature_id": fid,                         # -1 = baseline
                    "ablated_psmiles": new_smi,
                    "exact_match_to_original": int(new_smi == smi),
                })
            n_done += 1
            print(f"   [{n}/{len(rows)}] row {row_idx}  |c|_0={len(active)}  "
                  f"inter={len(inter)}  ablated={shortlist}")
        except Exception as e:
            n_failed += 1
            print(f"   [{n}/{len(rows)}] row {row_idx} FAILED on {smi!r}: {e}")
            writer.writerow({
                "row_idx": row_idx, "smiles": smi, "n_active": "",
                "n_intersect": "", "rank": "", "feature_id": "",
                "ablated_psmiles": "", "valid_smiles": "",
                "exact_match_to_original": "",
            })

    f_out.close()
    print(f"\n>> {n_done} molecules done, {n_failed} failed")
    print(f">> {args.out_csv}")


if __name__ == "__main__":
    main()