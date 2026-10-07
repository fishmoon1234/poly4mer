"""
evaluate_reconstruction_v3.py: pSMILES reconstruction accuracy of poly4mer_v3.ckpt
==================================================================================

For every input pSMILES: encode -> z -> decode -> token argmax -> string, and compare
with the input. Reported per file:

  exact         decoded token sequence identical to the input token sequence
                (all 202 positions; this is the training-time "seq_acc")
  token_acc     fraction of identical token positions
  valid         decoded string parses with RDKit
  same_molecule RDKit canonical form of the decoded string equals the canonical form of
                the input (the same molecule, even if the string is written differently)
  input_canon   fraction of INPUT strings that are already in RDKit canonical form with
                bare '*' (the form the model was trained on). A low value here usually
                explains "bad reconstruction" reports: the model reproduces canonical
                bare-star pSMILES; other spellings of the same molecule are not expected
                to round-trip character for character.

Input files: .smi / .txt (one pSMILES per line, first whitespace token), the 10-column
v2 format (pSMILES in the third column, detected automatically), or comma-separated
files with a smiles_canonicalized / smiles column (any extension). Use --max_rows N to
evaluate a random sample.

REQUIRED INPUT FORM. The model was trained on RDKit-canonical pSMILES with bare '*'
attachment points (e.g. '*CC(*)c1ccccc1'). Bracketed stars '[*]' and non-canonical atom
orderings of the same molecule do NOT round-trip. Pass --canonicalize_inputs to convert
inputs with canonical_psmiles() first; the script then also reports how many inputs
had to be changed. Use the same function on anything you feed to the encoder.

Example (run from inside this folder):
    python evaluate_reconstruction_v3.py --files exp=../data/experimental_data.smi,v2=../data/v2_110k_FDS.smi
    python evaluate_reconstruction_v3.py --files v1=../data/v1_training_data.smi --max_rows 100000
"""
import argparse, csv, os, random, sys, time
import torch
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # repository root
for _p in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from models import star_encoder, AutoEncoderLayer3, Decoder2, prediction_Model
from utils1 import load_smi_ted_explicit, find_smi_ted_dir
from train_poly4mer_v3_property_reg import forward_pass, MAX_LEN, EMB_DIM, VOCAB_SIZE
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SPECIAL = {"<pad>", "<unk>", "<s>", "</s>", "<bos>", "<eos>", "<mask>", "<cls>", "<sep>"}


def canonical_psmiles(s):
    """RDKit-canonical pSMILES with bare '*' (the model's native form); None if unparsable."""
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


def read_smiles(path):
    """Return the list of pSMILES in `path`, auto-detecting the column."""
    with open(path) as f:
        first = next((ln for ln in f if ln.strip()), "")
    if path.lower().endswith(".csv") or ("," in first and "smiles" in first.lower()):
        import pandas as pd
        df = pd.read_csv(path)
        for c in ("smiles_canonicalized", "smiles", "SMILES"):
            if c in df.columns:
                return df[c].astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()
    out = []
    col = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if col is None:                                   # decide once from the first row
                if "smiles_canonicalized" in t:
                    col = t.index("smiles_canonicalized"); continue
                col = 2 if (len(t) >= 3 and "*" not in t[0] and "*" in t[2]) else 0
            if len(t) > col:
                out.append(t[col])
    return out


def canon(s):
    m = Chem.MolFromSmiles(s) if s else None
    return Chem.MolToSmiles(m) if m is not None else None


def main():
    ap = argparse.ArgumentParser(description="pSMILES reconstruction accuracy of poly4mer_v3.")
    ap.add_argument("--checkpoint", default=os.path.join(SCRIPT_DIR, "poly4mer_v3.ckpt"))
    ap.add_argument("--files", required=True, help="Comma-separated name=path (or just path) entries.")
    ap.add_argument("--max_rows", type=int, default=0, help="Random sample size per file; 0 = all rows.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1995)
    ap.add_argument("--save_mismatches", default=None, help="CSV path to write every non-exact row (input, decoded, flags).")
    ap.add_argument("--show", type=int, default=5, help="Print this many mismatch examples per file.")
    ap.add_argument("--canonicalize_inputs", action="store_true",
                    help="Convert every input to RDKit-canonical bare-star form before encoding (unparsable inputs are dropped).")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device {dev}; checkpoint {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    A = star_encoder(dim=1); A.load_state_dict(ck["star_encoder"])
    ae = AutoEncoderLayer3(MAX_LEN * EMB_DIM, EMB_DIM * 4, EMB_DIM * 2, EMB_DIM); ae.encoder.load_state_dict(ck["autoencoder_encoder"])
    dec = Decoder2(MAX_LEN * EMB_DIM, EMB_DIM * 4, EMB_DIM * 2, EMB_DIM); dec.load_state_dict(ck["decoder"])
    tok = prediction_Model(n_embd=EMB_DIM, mid_size=EMB_DIM * 4, n_vocab=VOCAB_SIZE); tok.load_state_dict(ck["predictor"])
    del ck
    for m in (A, ae.encoder, dec, tok):
        m.to(dev).eval()
    SmiTed, Tokenizer = load_smi_ted_explicit()
    d = find_smi_ted_dir()
    smi_ted = SmiTed(Tokenizer(os.path.join(d, "bert_vocab_curated.txt")))
    smi_ted.load_checkpoint(os.path.join(d, "smi-ted-Light_40.pt")); smi_ted.eval()
    id2tok = {i: t.strip() for i, t in enumerate(open(os.path.join(d, "bert_vocab_curated.txt")))}

    writer = None
    if args.save_mismatches:
        fh = open(args.save_mismatches, "w", newline="")
        writer = csv.writer(fh); writer.writerow(["file", "input", "decoded", "valid", "same_molecule", "input_canonical"])

    print(f"\n{'file':<8}{'n':>9}{'exact':>9}{'token_acc':>11}{'valid':>9}{'same_mol':>10}{'input_canon':>13}   time")
    for entry in args.files.split(","):
        name, path = entry.split("=", 1) if "=" in entry else (os.path.basename(entry), entry)
        smiles = read_smiles(path)
        if args.max_rows and len(smiles) > args.max_rows:
            random.Random(args.seed).shuffle(smiles); smiles = smiles[:args.max_rows]
        if args.canonicalize_inputs:
            conv = [(s, canonical_psmiles(s)) for s in smiles]
            dropped = sum(c is None for _, c in conv); changed = sum(c is not None and c != s for s, c in conv)
            smiles = [c for _, c in conv if c is not None]
            print(f"   [{name}] canonicalized inputs: {changed} changed, {dropped} unparsable dropped, {len(smiles)} kept")
        t0 = time.time()
        n = len(smiles); exact = tok_ok = tok_n = valid = same = in_canon = 0
        examples = []
        with torch.no_grad():
            for i in range(0, n, args.batch_size):
                batch = smiles[i:i + args.batch_size]
                idx, _, _, logits, _ = forward_pass(batch, smi_ted, A, ae, dec, tok, dev)
                pred = logits.view(-1, MAX_LEN, VOCAB_SIZE).argmax(-1)
                ok_rows = torch.all(pred == idx, dim=1).cpu().tolist()
                exact += sum(ok_rows); tok_ok += int((pred == idx).sum()); tok_n += idx.numel()
                for s, row_ok, ids in zip(batch, ok_rows, pred.cpu().tolist()):
                    c_in = canon(s)
                    in_canon += int(c_in is not None and c_in == s)
                    if row_ok:
                        valid += int(c_in is not None); same += int(c_in is not None)
                        continue
                    decoded = "".join(id2tok.get(j, "") for j in ids if id2tok.get(j, "") not in SPECIAL)
                    c_out = canon(decoded)
                    v = c_out is not None; sm = v and c_in is not None and c_out == c_in
                    valid += int(v); same += int(sm)
                    if len(examples) < args.show:
                        examples.append((s, decoded, v, sm))
                    if writer:
                        writer.writerow([name, s, decoded, int(v), int(sm), int(c_in == s)])
        print(f"{name:<8}{n:>9}{100 * exact / n:>8.2f}%{100 * tok_ok / tok_n:>10.3f}%{100 * valid / n:>8.2f}%{100 * same / n:>9.2f}%{100 * in_canon / n:>12.2f}%   {time.time() - t0:.0f}s", flush=True)
        for s, decoded, v, sm in examples:
            print(f"      input   {s}\n      decoded {decoded}   valid={v} same_molecule={sm}")
    if writer:
        fh.close(); print(f">> mismatches written to {args.save_mismatches}")


if __name__ == "__main__":
    main()
