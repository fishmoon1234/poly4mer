"""
fire_inverse_design_v3.py: latent-space inverse design with poly4mer_v3.ckpt
============================================================================

The seed pSMILES is encoded to a single 768-d latent vector z. z is then the only
free variable: it is moved by plain gradient descent on a fire objective built from
the four property heads (tig, pHRR, SEA, CO) carried in the checkpoint. For every
step the current z is also decoded back to a token sequence, so the run yields a
stream of candidate pSMILES alongside the continuous property path. Decoding can 
optionally be deferred and performed in batches, giving increased performance at
the expense of memory usage.

Nothing except z is updated. smi-ted, the encoder, the decoder, the token predictor
and the four property heads are all frozen and in eval mode.

Latent convention
-----------------
The property heads consume the RAW latent z. Do not standardize z before passing it
in. Only thickness and flux are standardized, using the thk/flux statistics stored
in the checkpoint alongside the heads.

    tig    MLP([z, thk_norm, flux_norm])   input dim 770
    pHRR   MLP([z, thk_norm, flux_norm])   input dim 770
    SEA    MLP(z)                          input dim 768
    CO     MLP(z)                          input dim 768

    real value = output * y_std[k] + y_mean[k],  k = tig, pkhrr, sea, co

Objective
---------
    L =   c_tig  * relu(TIG_TARGET - tig)^2 / TIG_TARGET^2
        + c_qpua * pHRR^2 / PKHRR_MAX^2
        + c_sea  * SEA^2  / SEA_MAX^2
        + c_co   * CO^2   / CO_MAX^2

The tig term is a one-sided hinge: it stops pushing once the predicted ignition time
reaches the target. The other three are plain minimization, each divided by the
dataset maximum of that property so all four terms are dimensionless and comparable.

    z <- z - lr_start * dL/dz        (raw gradient descent, no momentum)

INPUT FORM. The seed must be an RDKit-canonical pSMILES with bare '*' attachment
points, for example '*CC(*)c1ccccc1'. Bracketed stars and non-canonical orderings of
the same molecule do not encode or decode faithfully. See canonical_psmiles() in
evaluate_reconstruction_v3.py.

--clamp
-------
Optional, off by default. Clips each prediction to its physical range before the loss.
clamp() has zero gradient outside the range, so a property that leaves its band stops
contributing instead of dragging the latent further out. It does NOT keep the latent
inside the decodable region: the head can still be predicting a negative pHRR while
the clipped value reads 22 kW/m^2. The property log therefore carries both the raw and
the clipped value in every row, so a clipped trajectory cannot hide a diverged latent.

Example (run from inside this folder):
    python fire_inverse_design_v3.py \
        --seed_smi '*N1C(=O)c2ccc(Oc3ccc(C(C)(C)c4ccc(Oc5ccc6c(c5)C(=O)N(*)C6=O)cc4)cc3)cc2C1=O' \
        --name PEI --lr_start 0.01 --thickness 3.0 --flux 50.0
"""

# Deep learning
import torch
import torch.nn.functional as F

# Data
import pandas as pd
import numpy as np

# Standard library
import argparse
import os
import sys
import csv
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # repository root
for _p in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import prediction_Model, Decoder2, star_encoder, AutoEncoderLayer3
from utils1 import load_smi_ted_explicit, find_smi_ted_dir
from train_poly4mer_v3_property_reg import (build_regressor, MAX_LEN, EMB_DIM,
                                            VOCAB_SIZE, STAR_TOKEN_ID)
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- constants --
NAMES = ["tig", "pkhrr", "sea", "co"]

# which heads also take thickness and flux (input dim 770 instead of 768)
USES_THK_FLUX = {"tig": True, "pkhrr": True, "sea": False, "co": False}

# target and the per-property scales that make the four loss terms dimensionless
TIG_TARGET, PKHRR_MAX, SEA_MAX, CO_MAX = 900.0, 2748.0, 0.149, 0.45

# physical ranges used by --clamp, in real units
PROP_RANGE = {"tig": (11.0, 898.0), "pkhrr": (22.0, 2748.0),
              "sea": (0.001, 0.149), "co": (0.01, 0.45)}

SPECIAL = ("<pad>", "<unk>", "<s>", "</s>", "<bos>", "<eos>", "<mask>", "<cls>", "<sep>")
N_STARS_MIN = 2            # 2 = linear repeat unit, 3 = branch point, 4 or 6 = network
                           # former; all occur in the experimental set. 0 or 1 cannot
                           # propagate a chain and is rejected.



def resolve_heads(ck, which, label="heads"):
    """Pick a head set from a checkpoint that may carry more than one.

    'selected' prefers property_regressors_selected and falls back to property_regressors
    for older checkpoints that only have the one set.
    """
    if which == "selected" and "property_regressors_selected" in ck:
        print(f">> {label}: property_regressors_selected (cross-validation selected)")
        return ck["property_regressors_selected"]
    if which == "selected":
        print(f">> {label}: property_regressors "
              f"(this checkpoint carries no selected set)")
    else:
        print(f">> {label}: property_regressors (the model's own heads)")
    return ck["property_regressors"]


def canonical_psmiles(s):
    """RDKit-canonical pSMILES with bare '*'; None if the string does not parse.
    '*' is read as a dummy atom, it is not substituted away."""
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


def count_stars(s):
    """Number of attachment points, or -1 if the string does not parse."""
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return -1
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 0)


def get_parser():
    p = argparse.ArgumentParser(description="poly4mer v3 latent inverse design")

    # seed: a literal pSMILES, or a row index into a csv with a 'smiles' column
    p.add_argument("--seed_smi", default=None,
                   help="Seed pSMILES; overrides --init_file / --smiles_idx.")
    p.add_argument("--init_file", default=None,
                   help="CSV with a 'smiles' column, used when --seed_smi is absent.")
    p.add_argument("--smiles_idx", type=int, default=1, help="1-based row index into --init_file.")
    p.add_argument("--name", default="seed", help="Label used in printouts and file names.")

    # optimization
    p.add_argument("--lr_start", type=float, default=0.01, help="Latent step size.")
    p.add_argument("--max_epochs", type=int, default=5000, help="Number of descent steps.")
    p.add_argument("--seed", type=int, default=1995)
    p.add_argument("--decode_batch_size", type=int, default=1,
                   help="Latents decoded together; 1 for sequential decoding (default).")
    p.add_argument("--threads", type=int, default=None, help="Number of CPU compute threads.")

    # objective weights
    p.add_argument("--c_tig", type=float, default=1.0)
    p.add_argument("--c_qpua", type=float, default=1.0, help="pHRR weight.")
    p.add_argument("--c_sea", type=float, default=1.0)
    p.add_argument("--c_co", type=float, default=1.0)

    # fire test conditions
    p.add_argument("--thickness", type=float, default=3.0, help="Sample thickness, mm.")
    p.add_argument("--flux", type=float, default=50.0, help="Incident heat flux, kW/m^2.")

    # weights
    p.add_argument("--checkpoint", default=os.path.join(SCRIPT_DIR, "checkpoint", "poly4mer_v3.ckpt"))
    p.add_argument("--heads", default="selected", choices=["selected", "joint"],
                   help="Which head set in --checkpoint to use. 'selected' is the cross-validation-selected set and the default; 'joint' is the model's own set, produced with the rest of the model. Ignored when --heads_ckpt names an external file.")
    p.add_argument("--heads_ckpt", default=None,
                   help="Optional separate file holding property_regressors and "
                        "property_stats; by default the heads in --checkpoint are used.")

    # behaviour
    p.add_argument("--clamp", action="store_true",
                   help="Clip predictions to their physical range before the loss.")
    p.add_argument("--keep_unphysical", action="store_true",
                   help="Keep designs whose predicted pHRR is negative; off by default.")
    p.add_argument("--results_dir", default=os.path.join(SCRIPT_DIR, "results"))
    p.add_argument("--save_embeddings", action="store_true",
                   help="Dump the full latent trajectory; large, off by default.")
    return p


def main(config):
    if config.decode_batch_size < 1:
        raise SystemExit("--decode_batch_size must be positive")
    if config.threads is not None:
        if config.threads < 1:
            raise SystemExit("--threads must be positive")
        torch.set_num_threads(config.threads)

    # ------------------------------------------------------------- 1. Device --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device being used: {device}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ------------------------------------------- 2. Load models and statistics --
    print(f">> checkpoint {config.checkpoint}")
    ck = torch.load(config.checkpoint, map_location="cpu", weights_only=False, mmap=True)

    A_encoder = star_encoder(dim=1)
    autoencoder = AutoEncoderLayer3(MAX_LEN * EMB_DIM, EMB_DIM * 4, EMB_DIM * 2, EMB_DIM)
    decoder = Decoder2(MAX_LEN * EMB_DIM, EMB_DIM * 4, EMB_DIM * 2, EMB_DIM)
    lang_model = prediction_Model(n_embd=EMB_DIM, mid_size=EMB_DIM * 4, n_vocab=VOCAB_SIZE)

    # The star encoder must be loaded with the rest. An unloaded star_encoder leaves
    # every attachment-point embedding at its random initialization.
    A_encoder.load_state_dict(ck["star_encoder"])
    autoencoder.encoder.load_state_dict(ck["autoencoder_encoder"])
    decoder.load_state_dict(ck["decoder"])
    lang_model.load_state_dict(ck["predictor"])
    for m in (A_encoder, autoencoder, decoder, lang_model):
        m.to(device)

    # Property heads, from --heads_ckpt when given, otherwise from the checkpoint.
    hk = ck if config.heads_ckpt is None else torch.load(
        config.heads_ckpt, map_location="cpu", weights_only=False)
    if config.heads_ckpt is not None:
        print(f">> heads: {os.path.basename(config.heads_ckpt)}")
    src = (hk["property_regressors"] if config.heads_ckpt is not None
           else resolve_heads(ck, config.heads))
    regressors = {}
    for n in NAMES:
        r = build_regressor(770 if USES_THK_FLUX[n] else 768)
        r.load_state_dict(src[n])
        regressors[n] = r.to(device)

    stats = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
             for k, v in hk["property_stats"].items()}
    y_mean, y_std = stats["y_mean"].view(-1), stats["y_std"].view(-1)

    # Frozen smi-ted-Light.
    smi_ted_dir = find_smi_ted_dir()
    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    vocab_file = os.path.join(smi_ted_dir, "bert_vocab_curated.txt")
    smi_ted = Smi_ted(MolTranBertTokenizer(vocab_file))
    smi_ted.load_checkpoint(os.path.join(smi_ted_dir, "smi-ted-Light_40.pt"))

    # Freeze everything. z is the only tensor that will carry a gradient.
    for m in [decoder, lang_model, autoencoder.encoder, A_encoder, smi_ted,
              *regressors.values()]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    with open(vocab_file) as f:
        id_to_token = {i: line.strip() for i, line in enumerate(f)}

    # ------------------------------------------------- 3. Input conditions --
    thk_norm = ((torch.tensor(config.thickness, device=device, dtype=torch.float32)
                 - stats["thk_mean"]) / stats["thk_std"]).view(1, 1)
    flux_norm = ((torch.tensor(config.flux, device=device, dtype=torch.float32)
                  - stats["flux_mean"]) / stats["flux_std"]).view(1, 1)
    print(f">> conditions: thickness {config.thickness} mm, flux {config.flux} kW/m^2, "
          f"tig target {TIG_TARGET} s")

    # --------------------------------------------------- 4. Helper closures --
    def encode(smiles):
        """pSMILES -> latent z of shape [1, 768]."""
        with torch.no_grad():
            idx_t, token_emb_t, _ = smi_ted.extract_embeddings(smiles)
            idx_t, token_emb_t = idx_t.to(device), token_emb_t.to(device)
            pos = torch.nonzero(idx_t == STAR_TOKEN_ID, as_tuple=False)
            if pos.numel():
                token_emb_t[pos[:, 0], pos[:, 1], :] = \
                    A_encoder(idx_t[pos[:, 0], pos[:, 1]].view(-1, 1).float())
            return autoencoder.encoder(token_emb_t.view(-1, MAX_LEN * EMB_DIM))

    def predict(z):
        """Latent -> the four properties in real units. Differentiable in z."""
        in770 = torch.cat([z, thk_norm, flux_norm], dim=-1)
        return {n: regressors[n](in770 if USES_THK_FLUX[n] else z).view(-1)
                   * y_std[k] + y_mean[k]
                for k, n in enumerate(NAMES)}

    def decode_batch(latents):
        """Decode every independent row, in order."""
        with torch.no_grad():
            rec = decoder(latents).view(-1, MAX_LEN, EMB_DIM)
            ids = lang_model(rec.reshape(-1, EMB_DIM)).view(-1, MAX_LEN, VOCAB_SIZE).argmax(-1)
            return ["".join(id_to_token.get(int(i), "") for i in row
                            if id_to_token.get(int(i), "") not in SPECIAL)
                    for row in ids.tolist()]

    # -------------------------------------------------- 5. Initialization --
    if config.seed_smi is not None:
        smiles_init = config.seed_smi
    elif config.init_file is not None:
        smiles_init = pd.read_csv(config.init_file, delimiter=",")["smiles"][config.smiles_idx - 1]
    else:
        raise SystemExit("give either --seed_smi or --init_file")

    seed_canon = canonical_psmiles(smiles_init)
    if seed_canon is None:
        raise SystemExit(f"seed does not parse: {smiles_init}")
    if seed_canon != smiles_init:
        print(f">> note: seed is not in canonical bare-star form; canonical is {seed_canon}")

    embeddings = encode(smiles_init).clone().detach().requires_grad_(True)
    print(f">> seed [{config.name}]: {smiles_init}")

    save_embeddings = [embeddings.clone().detach()] if config.save_embeddings else []
    save_smiles = [(-1, smiles_init)]
    save_fire_props = []            # raw real-unit prediction, one row per step
    save_clamped_props = []         # what the clamped loss actually saw
    save_canonicalized_smiles = []
    designs = {}                    # canonical pSMILES -> (step, properties at that step)

    pending_latents = []

    def flush_decodes():
        strings = decode_batch(torch.cat(pending_latents, dim=0))
        first = len(save_fire_props) - len(pending_latents)
        for ep, smiles_string in enumerate(strings, start=first):
            raw = dict(zip(NAMES, save_fire_props[ep]))
            canon_smiles = canonical_psmiles(smiles_string)

            if smiles_string != save_smiles[-1][-1]:
                save_smiles.append((ep, smiles_string))
                if canon_smiles is not None:
                    save_canonicalized_smiles.append((ep, canon_smiles))

            if (canon_smiles is not None and canon_smiles != seed_canon
                    and canon_smiles not in designs
                    and count_stars(canon_smiles) >= N_STARS_MIN
                    and (config.keep_unphysical or config.clamp or raw["pkhrr"] >= 0)):
                designs[canon_smiles] = (ep, {n: raw[n] for n in NAMES})

        pending_latents.clear()

    # ------------------------------------------------- 6. Optimization loop --
    for ep in range(config.max_epochs):
        # (a) predict at the CURRENT latent
        raw = predict(embeddings)

        # (b) snapshot the SAME latent for deferred decoding, so the structure and the 
        #     numbers recorded at a given step always correspond to each other
        pending_latents.append(embeddings.detach().clone())

        # (c) optional clip to physical ranges
        used = ({n: raw[n].clamp(PROP_RANGE[n][0], PROP_RANGE[n][1]) for n in NAMES}
                if config.clamp else raw)

        # (d) fire objective
        loss = (config.c_tig * F.relu(TIG_TARGET - used["tig"]).pow(2) / TIG_TARGET ** 2
                + config.c_qpua * used["pkhrr"].pow(2) / PKHRR_MAX ** 2
                + config.c_sea * used["sea"].pow(2) / SEA_MAX ** 2
                + config.c_co * used["co"].pow(2) / CO_MAX ** 2).sum()

        # log the raw prediction too, so clipping can never hide a diverged latent
        save_fire_props.append([raw[n].item() for n in NAMES])
        save_clamped_props.append([used[n].item() for n in NAMES])

        # (e) one descent step, on the latent only
        embeddings.grad = None
        loss.backward()
        with torch.no_grad():
            embeddings -= config.lr_start * embeddings.grad

        if config.save_embeddings:
            save_embeddings.append(embeddings.clone().detach())

        if len(pending_latents) == config.decode_batch_size:
            flush_decodes()

    if pending_latents:
        flush_decodes()

    print(">> Optimization Finished.")

    # ------------------------------------------------------- 7. Report --
    seed_props = {n: save_fire_props[0][k] for k, n in enumerate(NAMES)}
    new = sorted(designs.items(), key=lambda kv: kv[1][0])
    n_linear = sum(1 for cs, _ in new if count_stars(cs) == 2)

    print(f"\n### {config.name}  lr={config.lr_start:g}  clamp={config.clamp}   "
          f"{len(new)} design(s): {n_linear} linear, {len(new) - n_linear} branched")
    print("    seed props (predicted): "
          + "  ".join(f"{n} {seed_props[n]:.4g}" for n in NAMES))
    for k, (cs, (ep, pr)) in enumerate(new, 1):
        stars = count_stars(cs)
        flag = "linear" if stars == 2 else f"{stars}-star"
        print(f"    d{k}  ep{ep:>5}  [{flag:>7}]  {cs}  "
              + "  ".join(f"{n} {pr[n]:.4g}" for n in NAMES))

    # ------------------------------------------------------- 8. Saving --
    save_file = (f"tig{config.c_tig}_phrr{config.c_qpua}_sea{config.c_sea}_co{config.c_co}"
                 f"_flux{config.flux}_thickness{config.thickness}"
                 + ("_clamp" if config.clamp else ""))
    base_paths = ["optimized_smiles", "optimized_smiles_canonicalized",
                  "log_optimized_prop", "designs"]
    if config.save_embeddings:
        base_paths.append("optimized_embeddings")
    for subdir in base_paths:
        os.makedirs(os.path.join(config.results_dir, subdir, save_file), exist_ok=True)

    tag = f"{config.name}_{config.smiles_idx}_lr_{config.lr_start:.4f}"

    def out_path(subdir, stem):
        return os.path.join(config.results_dir, subdir, save_file, stem)

    if config.save_embeddings:
        with open(out_path("optimized_embeddings", f"optimized_embeddings_{tag}.txt"), "w") as f:
            for embd in save_embeddings:
                np.savetxt(f, embd.cpu().numpy(), fmt="%.6f")

    with open(out_path("optimized_smiles", f"optimized_smiles_{tag}.txt"), "w") as f:
        for ep, smiles in save_smiles:
            f.write(f"Epoch {ep}: {smiles}\n")

    with open(out_path("optimized_smiles_canonicalized",
                       f"optimized_smiles_canonicalized_{tag}.txt"), "w") as f:
        for ep, smiles in save_canonicalized_smiles:
            f.write(f"Epoch {ep}: {smiles}\n")

    with open(out_path("log_optimized_prop", f"log_optimized_props_{tag}.txt"), "w") as f:
        f.write("tig, qpua_pk, sea, co, tig_clipped, qpua_pk_clipped, sea_clipped, co_clipped\n")
        writer = csv.writer(f)
        for row, crow in zip(save_fire_props, save_clamped_props):
            writer.writerow([f"{x:.6f}" for x in row] + [f"{x:.6f}" for x in crow])

    with open(out_path("designs", f"designs_{tag}.json"), "w") as f:
        json.dump({
            "name": config.name,
            "seed_smi": smiles_init,
            "seed_canon": seed_canon,
            "seed_props": seed_props,
            "lr_start": config.lr_start,
            "max_epochs": config.max_epochs,
            "clamp": config.clamp,
            "thickness": config.thickness,
            "flux": config.flux,
            "tig_target": TIG_TARGET,
            "coef": {"tig": config.c_tig, "pkhrr": config.c_qpua,
                     "sea": config.c_sea, "co": config.c_co},
            "designs": [{"ep": ep, "smi": cs, "stars": count_stars(cs),
                         "linear": count_stars(cs) == 2, **pr}
                        for cs, (ep, pr) in new],
        }, f, indent=2)

    print(f">> saved -> {os.path.join(config.results_dir, '*', save_file)}")


if __name__ == "__main__":
    main(get_parser().parse_args())
