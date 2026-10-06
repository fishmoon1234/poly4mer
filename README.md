# poly4mer v3 (dev branch)

A multi-physics synthesis based chemical language model for polymer fire property prediction and inverse design.

![Poly4mer architecture.](https://github.com/fishmoon1234/poly4mer/blob/main/poly4mer.png)

This branch carries **poly4mer v3**. It maps a polymer pSMILES string to a single 768-dimensional latent vector `z`. The latent can be decoded back into the pSMILES, and it predicts four fire properties: time to ignition (tig), peak heat release rate (pHRR), smoke extinction area (SEA), and carbon monoxide yield (CO). The released v1.0.0 and v2.0.0 code stays available under the tags of the same names.

**Highlights**

1. The model is developed based on [SMI-TED](https://github.com/IBM/materials/tree/main), and trained to predict 4 flammability metrics from polymer SMILES.
2. We introduce a principled multi-physics synthesis framework to data-scarce learning by embedding domain knowledge directly into language model training, enabling robust generalization beyond pure data-driven chemical language models.
3. We design a new strategy for introducing new tokens into pretrained language models via encoder-decoder architecture expansion, leveraging monomer representations to capture complex polymer semantics.
4. We architect an autoencoding system coupling predictive modeling with generative design, enabling inverse polymer design via latent-space exploration and structure reconstruction for targeted applications.

> **Two kinds of "predictor".** In the checkpoint, the key `predictor` is the **token predictor**, the last part of the decoder module. The **property predictor** is stored under `property_regressors`.

## Requirements

- Python 3.10.18 and a CUDA GPU (about 4 GB for inference, about 16 GB for joint training).
- `pip install -r requirements.txt` installs the exact package versions the model was validated with, including `torch==2.8.0+cu128` from the PyTorch wheel index named at the top of the file. On a machine with a different CUDA version, install a matching torch build and keep the other pins. The core packages are torch, numpy, pandas, rdkit, transformers, regex, and tqdm.
- Reconstruction accuracy depends on exact package versions, in particular `transformers`, because the SMILES tokenizer subclasses `BertTokenizer`. If reconstruction is near 0% in an environment of your own, re-create the environment from `requirements.txt` before anything else.

## Pretrained model

The SMI-TED Python package is bundled under `smi_ted_light/`; only the weights are downloaded. Download from our [HuggingFace mirror](https://huggingface.co/fishmoon1234/poly4mer) and place the files at the paths shown:

- `poly4mer_v3.ckpt` → repository root (next to the scripts)
- `smi-ted-Light_40.pt` → `smi_ted_light/`

Both files are ignored by git. After downloading, the layout is:

```
poly4mer/
|-- poly4mer_v3.ckpt
|-- smi_ted_light/
|   |-- smi-ted-Light_40.pt
|   |-- bert_vocab_curated.txt
|   |-- load.py
|   |-- fast_transformers/
|-- models.py
|-- utils1.py
|-- train_poly4mer_v3.py
|-- phase1_finetune_heads.py
|-- evaluate_reconstruction_v3.py
|-- requirements.txt
```

## Files

| File | Purpose |
|---|---|
| `poly4mer_v3.ckpt` | Trained weights: encoder module, decoder module, property predictor (3.9 GB, downloaded separately) |
| `models.py` | Network definitions |
| `utils1.py` | Loads the frozen smi-ted encoder and provides small helpers |
| `train_poly4mer_v3.py` | Joint training script (all networks); also defines the forward pass and the property heads |
| `phase1_finetune_heads.py` | Phase I: fine-tune only the property heads on labeled data, encoder frozen |
| `evaluate_reconstruction_v3.py` | Measures pSMILES reconstruction accuracy on any list of pSMILES; also provides `canonical_psmiles()` |
| `smi_ted_light/` | Frozen smi-ted-Light encoder from IBM: loader, vocabulary, and the downloaded weights |
| `requirements.txt` | Exact package versions |

## Model components

| Module | Parts, in order | Where the weights are |
|---|---|---|
| **Encoder module** | smi-ted token embedder, then `star_encoder`, then the autoencoder encoder | smi-ted: `smi_ted_light/smi-ted-Light_40.pt`, frozen. The rest: `star_encoder` and `autoencoder_encoder` in `poly4mer_v3.ckpt` |
| **Decoder module** | autoencoder decoder, then the token predictor | `decoder` and `predictor` in `poly4mer_v3.ckpt` |
| **Property predictor** | four MLP heads on the latent `z`, for tig, pHRR, SEA, CO | `property_regressors` and `property_stats` in `poly4mer_v3.ckpt` |

## What is saved in `poly4mer_v3.ckpt`

The file is a Python dict saved with `torch.save`. All tensors are float32.

| Key | Class | Parameters | Role |
|---|---|---:|---|
| `star_encoder` | `star_encoder(dim=1)` in `models.py` | 197,888 | Learned embedding for the `*` attachment-point token |
| `autoencoder_encoder` | `AutoEncoderLayer3(...).encoder` in `models.py` | 482,489,856 | Token embeddings, 202 × 768, to latent `z`, 768 |
| `decoder` | `Decoder2(...)` in `models.py` | 482,489,856 | Latent `z` back to token embeddings, 202 × 768 |
| `predictor` | `prediction_Model(...)` in `models.py` | 9,719,808 | Token predictor: token embedding to logits over the 2393-token vocabulary |
| `property_regressors` | `build_regressor(...)` in `train_poly4mer_v3.py`, one per property | 3,284,996 in total | Property predictor: dict with heads `tig`, `pkhrr`, `sea`, `co` |
| `property_stats` | n/a | 12 values | Standardization used by the heads: `thk_mean`, `thk_std`, `flux_mean`, `flux_std`, `y_mean`, `y_std` |
| `meta` | n/a | n/a | Model dimensions and descriptive notes |

**Using the decoder module.** The decoder and token predictor work as a pair. Always load both from this checkpoint.

**Using the property predictor.** The tig and pHRR heads take `[z, (thickness - thk_mean) / thk_std, (flux - flux_mean) / flux_std]` as input, 770 values. The SEA and CO heads take `z` alone, 768 values. Each head outputs a standardized value. Convert it with `raw = output * y_std[k] + y_mean[k]`, where `k` runs over tig, pkhrr, sea, co in that order.

## Architecture

smi-ted-Light is used only as a frozen tokenizer and token embedder. Its weights never change.

```
pSMILES
  -> smi-ted (frozen)     token ids [B, 202], token embeddings [B, 202, 768]
  -> star_encoder         replaces the embedding of each '*' token (id 439)
  -> encoder              flatten 202*768 -> z [B, 768]          (the latent)
  -> decoder              z -> reconstructed embeddings [B, 202, 768]
  -> token predictor      embeddings -> token logits [B*202, 2393]
  -> property predictor   tig, pHRR: MLP([z, thickness, flux])   (770 inputs)
                          SEA, CO:   MLP(z)                      (768 inputs)
```

Sequences are padded or truncated to 202 tokens, so very long pSMILES cannot be represented.

## Required input form: canonical pSMILES with bare `*`

The model reads and writes **RDKit-canonical pSMILES with bare `*` attachment points**, for example `*CC(*)c1ccccc1`. Two other spellings of the same molecule are common and both break reconstruction:

- bracketed stars such as `[*]CC([*])c1ccccc1`: exact reconstruction drops to 0%, because the attachment-point token is only recognised as a bare `*`;
- any non-canonical atom ordering such as `C(c1ccccc1)(*)C*`: exact reconstruction drops to a few percent.

Always canonicalize first. One line with RDKit does it and is idempotent on inputs that are already canonical:

```python
from rdkit import Chem
def canonical_psmiles(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None   # bare '*' comes out automatically
```

The same function is available as `canonical_psmiles()` in `evaluate_reconstruction_v3.py`. The model is built for single repeat units with two attachment points. Multi-component entries written with `.` are not reconstructed reliably.

## Encode, decode, and predict properties

Run this from the repository root. Inputs must be in the canonical form described above.

```python
import os, torch
from models import star_encoder, AutoEncoderLayer3, Decoder2, prediction_Model
from utils1 import load_smi_ted_explicit, find_smi_ted_dir
from train_poly4mer_v3 import forward_pass

dev = torch.device("cuda")
ck = torch.load("poly4mer_v3.ckpt", map_location="cpu", weights_only=False)
A = star_encoder(dim=1);                                       A.load_state_dict(ck["star_encoder"])
ae = AutoEncoderLayer3(202*768, 768*4, 768*2, 768);            ae.encoder.load_state_dict(ck["autoencoder_encoder"])
dec = Decoder2(202*768, 768*4, 768*2, 768);                    dec.load_state_dict(ck["decoder"])
tok = prediction_Model(n_embd=768, mid_size=768*4, n_vocab=2393); tok.load_state_dict(ck["predictor"])
for m in (A, ae.encoder, dec, tok):
    m.to(dev).eval()
del ck

SmiTed, Tokenizer = load_smi_ted_explicit()
d = find_smi_ted_dir()
smi_ted = SmiTed(Tokenizer(os.path.join(d, "bert_vocab_curated.txt")))
smi_ted.load_checkpoint(os.path.join(d, "smi-ted-Light_40.pt"))
smi_ted.eval()
id2tok = {i: t.strip() for i, t in enumerate(open(os.path.join(d, "bert_vocab_curated.txt")))}
SPECIAL = {"<pad>", "<unk>", "<s>", "</s>", "<bos>", "<eos>"}

smiles = ["*CC(*)c1ccccc1", "*CC(*)(C)C(=O)OC"]
with torch.no_grad():
    idx, emb, emb_rec, logits, z = forward_pass(smiles, smi_ted, A, ae, dec, tok, dev)
print(z.shape)                                   # torch.Size([2, 768]): the latent vectors

# Decode any latent (here, z itself) back to pSMILES.
with torch.no_grad():
    ids = tok(dec(z).view(-1, 768)).view(len(z), 202, 2393).argmax(-1).tolist()
decoded = ["".join(id2tok[i] for i in row if id2tok[i] not in SPECIAL) for row in ids]
print(decoded)                                   # ['*CC(*)c1ccccc1', '*CC(*)(C)C(=O)OC']

# Predict the four fire properties at a chosen thickness (mm) and heat flux (kW/m^2).
from train_poly4mer_v3 import build_regressor, PROPERTY_NAMES, PROP_USES_THK_FLUX
ck = torch.load("poly4mer_v3.ckpt", map_location="cpu", weights_only=False, mmap=True)
heads = {}
for n in PROPERTY_NAMES:
    heads[n] = build_regressor(770 if PROP_USES_THK_FLUX[n] else 768)
    heads[n].load_state_dict(ck["property_regressors"][n])
    heads[n].to(dev).eval()
st = {k: torch.as_tensor(v, device=dev) for k, v in ck["property_stats"].items()}
del ck

thickness, flux = 3.0, 50.0
thk_n = torch.full((len(z), 1), (thickness - st["thk_mean"].item()) / st["thk_std"].item(), device=dev)
flux_n = torch.full((len(z), 1), (flux - st["flux_mean"].item()) / st["flux_std"].item(), device=dev)
x770 = torch.cat([z, thk_n, flux_n], dim=-1)
with torch.no_grad():
    for k, n in enumerate(PROPERTY_NAMES):
        out = heads[n](x770 if PROP_USES_THK_FLUX[n] else z).view(-1)
        print(n, (out * st["y_std"][k] + st["y_mean"][k]).tolist())
```

## Check reconstruction on your own polymers

```bash
python evaluate_reconstruction_v3.py --files mine=my_polymers.smi --canonicalize_inputs
```

The script reports exact-match, token, validity, and same-molecule rates, and how many inputs it had to canonicalize. Input files are `.smi` or `.txt` with one pSMILES per line, or comma-separated files with a `smiles` column.

## Phase I fine-tuning: property heads only, encoder frozen

Use this to adapt the property predictor to new labeled data without touching the encoder or decoder. Because the latent `z` of a polymer does not change, the script embeds every unique pSMILES once with the frozen encoder, stores the latents in a cache file, and trains the heads on the cached latents. After the embedding step an epoch takes seconds even for 100k rows.

What is trained: the four heads under `property_regressors` only. What is frozen: smi-ted, `star_encoder`, and the autoencoder encoder. The decoder module is not loaded.

### Command

```bash
python phase1_finetune_heads.py \
    --train_labeled my_data/train.csv \
    --val_labeled   my_data/val.csv \
    --out_dir       runs/phase1 \
    --epochs 50 --lr 1e-4 --batch_size 256 --patience 8
```

The labeled file format is the same as for `train_poly4mer_v3.py` (see Data under Joint training below). Rows may have missing property values; each head is trained only on rows where its label is present.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint` | `poly4mer_v3.ckpt` | Provides the frozen encoder and, by default, the initial heads and their standardization statistics |
| `--properties` | `tig,pkhrr,sea,co` | Subset of heads to train, for example `tig,pkhrr` |
| `--fresh_heads` | off | Start the heads from random weights; statistics are then computed from the training data |
| `--recompute_stats` | off | Keep the checkpoint heads but recompute the statistics from the training data |
| `--epochs`, `--batch_size`, `--lr`, `--wd` | 50, 256, 1e-4, 1e-5 | Adam settings |
| `--scheduler_step`, `--scheduler_gamma` | 10, 0.7 | lr × gamma every `scheduler_step` epochs |
| `--patience` | 0 | Stop when the validation loss has not improved for this many epochs; 0 disables |
| `--embed_batch_size` | 64 | Batch size of the frozen encoder during the embedding step |
| `--latent_cache` | `<out_dir>/latent_cache.pt` | Cache file; reuse it across runs on the same data to skip embedding |
| `--save_full` | off | Also write `poly4mer_v3_phase1.ckpt`, a full drop-in replacement for `poly4mer_v3.ckpt` with the new heads |
| `--eval_only` | off | Only evaluate the heads in `--checkpoint` on `--val_labeled` |

The loss is the sum over the selected properties of the mean squared error in standardized units. By default the heads and the standardization statistics come from the checkpoint, so training continues from the shipped predictor. With `--fresh_heads` the statistics are recomputed from your training file, which is the right choice when your data covers a different range than the shipped heads were built for.

### Outputs

| File | Content |
|---|---|
| `heads_best.ckpt` | `property_regressors`, `property_stats`, the trained property list, the best epoch, and its validation metrics (about 13 MB) |
| `history.csv` | Per-epoch training loss per head, validation loss, and validation relative errors (median and mean) per head |
| `latent_cache.pt` | Latents of every embedded pSMILES |
| `poly4mer_v3_phase1.ckpt` | Only with `--save_full`: all networks plus the new heads, loadable exactly like `poly4mer_v3.ckpt` |

If `--val_labeled` is given, the epoch with the lowest validation loss is kept. Otherwise the last epoch is kept. Read the **median** relative error for pHRR and CO: near-zero targets make the mean relative error of those two properties meaningless.

To use the new heads, load `heads_best.ckpt` and read `property_regressors` and `property_stats` from it exactly as in the prediction example above. The encoder and decoder still come from `poly4mer_v3.ckpt`. With `--save_full` you can instead point the example at `poly4mer_v3_phase1.ckpt`.

To evaluate any heads file on a labeled set without training:

```bash
python phase1_finetune_heads.py --eval_only --checkpoint runs/phase1/heads_best.ckpt \
    --val_labeled my_data/val.csv --out_dir runs/phase1_eval
```

## Joint training

### Data

Training needs the following files. Pass several files of one kind as a comma-separated list.

| Argument | Content | Used for |
|---|---|---|
| `--train_smiles` | pSMILES only | Reconstruction |
| `--train_labeled` | pSMILES, thickness, flux, and properties | Reconstruction and property prediction |
| `--val_data` | pSMILES only | Reconstruction accuracy; selects the best checkpoint |
| `--val_labeled` (optional) | Same format as labeled training data | Prints property relative error each epoch |

Accepted formats:

- **pSMILES-only files**: a `.smi` file with the pSMILES as the first whitespace-separated token on each line. Alternatively, a `.csv` file with a `smiles_canonicalized` or `smiles` column.
- **Labeled files**: a `.csv` file, or a whitespace-separated `.smi` file with a header line. Either one needs the columns `smiles_canonicalized thickness flux tig pkhrr sea co`. A headerless 10-column `.smi` file is also accepted, in the order `index name smiles thickness flux tig pkhrr sea co igt`.
- Units: thickness in mm, flux in kW/m², tig in s, pHRR in kW/m², CO yield in kg/kg. SEA must use the same units as the property predictor's outputs.
- Missing property values may be left as `NaN`. They are skipped property by property. Rows with no property value at all are dropped.

### Command

```bash
python train_poly4mer_v3.py \
    --train_smiles  my_data/train_smiles.smi \
    --train_labeled my_data/train_labeled.csv \
    --val_data      my_data/val_smiles.smi \
    --val_labeled   my_data/val_labeled.csv \
    --checkpoint_dir runs/my_run \
    --epochs 10
```

Key options and their defaults:

| Option | Default | Meaning |
|---|---|---|
| `--init_recon_ckpt` | `poly4mer_v3.ckpt` | Starting weights for all networks, including the property heads |
| `--fresh_heads` | off | Start the property heads from random weights instead of the checkpoint |
| `--no_init` | off | Start everything from random weights |
| `--lr` | 1e-5 | Adam learning rate |
| `--scheduler_step`, `--scheduler_gamma` | 1000, 0.7 | lr × gamma every `scheduler_step` epochs; 1000 keeps lr constant |
| `--wd` | 1e-5 | Adam weight decay |
| `--lamb` | 1.0 | Weight of the embedding-reconstruction term |
| `--property_lamb` | 0.1 | Weight of the property term |
| `--batch_size` | 48 | pSMILES-only batch size, also used for validation |
| `--labeled_batch_size` | 32 | Labeled batch size |
| `--epochs` | 10 | Number of epochs for a new run |
| `--seed` | 1995 | Random seed |

Property targets are standardized. When the heads are loaded from `poly4mer_v3.ckpt`, the script keeps the checkpoint's `property_stats`, because the heads expect that scaling. With `--fresh_heads`, it computes the statistics from your labeled training data.

### Outputs

After every epoch the script measures exact reconstruction accuracy on `--val_data`. When the accuracy matches or beats the best so far, it saves `<checkpoint_dir>/model_params<N>_lamb<lamb>_best.ckpt`. That file holds all networks, the property heads, the property standardization statistics, the Adam state, and the epoch number. It is about 12 GB because of the Adam state. A loss history is also written to `losses_*.txt` in the same folder.

To continue an interrupted run, add `--resume` to the same command. That restores weights, optimizer state, and the epoch counter from the newest best checkpoint, or from `--checkpoint_path` if given. With `--resume`, `--epochs` is the final epoch index, not a count of extra epochs.

### Hardware

Joint training updates about 978M parameters with Adam. It needs about 16 GB of GPU memory for weights, gradients, and optimizer state, plus activations.

## Citation

If you find our models useful, please consider citing our papers:

```
@article{liu2025harnessing,
  title={Harnessing large language models for data-scarce learning of polymer properties},
  author={Liu, Ning and Jafarzadeh, Siavash and Lattimer, Brian Y and Ni, Shuna and Lua, Jim and Yu, Yue},
  journal={Nature Computational Science},
  volume={5},
  number={3},
  pages={245--254},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
@inproceedings{yin2025fake,
  title={Fake It Till You Make It: Multi-Physics Synthesis Breaks the Data Barrier in Chemical Language Models},
  author={Yin, Naiyu and Liu, Ning and Chen, Jiuzhou and Lattimer, Brian Y and Lua, Jim and Yu, Yue},
  booktitle={NeurIPS 2025 Machine Learning and the Physical Sciences Workshop}
}
```
