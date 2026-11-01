# Release title

poly4mer v3.0.0 — one checkpoint for prediction, reconstruction and inverse design

# Release note

poly4mer v3.0.0 maps a polymer pSMILES to a single 768-dimensional latent and uses that one
representation for three things: reconstructing the pSMILES, predicting four
cone-calorimeter properties, and running gradient-based inverse design in latent space.
Predictions are conditioned on the test setup, so the same polymer can be queried at
different sample thicknesses and heat fluxes.

Properties: time to ignition, peak heat release rate, smoke extinction area and
carbon monoxide yield.

## What is new since v2.0.0

**One checkpoint instead of six files.** `poly4mer_v3.ckpt` carries the encoder, the decoder,
the token predictor and the property heads. No separate scaler file is needed: the heads read
the latent directly and their standardization statistics travel with them.

**Reconstruction that works.** On canonical bare-star pSMILES the decoder returns the input
exactly for 99.99% of a 100,000-string sample of the training corpus, 100.00% of the 109,993
simulated structures, and all 16 commodity homopolymers, seven of which are absent from the
training corpus.

**Inverse design.** `fire_inverse_design_v3.py` runs gradient descent on the latent with every
weight frozen, decoding each step into a candidate pSMILES. `run_design.sh` reproduces five
worked examples. Designs are accepted at two or more attachment points, so branch points and
network formers count alongside linear repeat units.

**Optional monotone heads.** Alternative tig and pHRR heads that are hard-monotone in heat
flux by construction: tig can only fall as flux rises, pHRR can only rise. The constraint is
structural rather than a penalty, so it holds for every input. See `poly4mer_monotone/`.

**Runnable demos.** `run_property_prediction.sh`, `run_design.sh` and
`run_demo_property_heads.sh`, the last of which scores the bundled per-fold heads, re-trains
one property's folds and prints the two side by side.

**Cross-validation reproduction.** `data/10folds_data/` ships the experimental splits for all
four properties, with `evaluate_fire_prediction_heads.py` to score heads and a `--cv` mode in
the training script to re-run the protocol.

## Property prediction

Mean absolute relative error on experimental data, 10-fold cross-validation.

| Property | Best 5 folds | All 10 folds |
|---|---:|---:|
| tig | 15.76% | 19.11% |
| pHRR | 12.61% | 16.39% |
| SEA | 1.58% | 3.63% |
| CO | 0.93% | 3.94% |

The training epoch and learning rate are selected on each fold's own test rows, so the
best-5 column is optimistically biased. Prefer the all-10 column for a single number. The
repository ships the fold data and the per-fold heads so both can be checked.

## Getting started

```bash
git clone -b dev https://github.com/fishmoon1234/poly4mer.git && cd poly4mer
pip install -r requirements.txt
# download poly4mer_v3.ckpt into checkpoint/ and smi-ted-Light_40.pt into smi_ted_light/
python evaluate_reconstruction_v3.py --files homo=example_pSMILES_homo.txt   # expect 16/16
python fire_property_prediction_V3.py --smiles '*CC(*)C' --thickness 3 --flux 50
./run_design.sh
```

Weights: https://huggingface.co/fishmoon1234/poly4mer under `poly4mer_pretrained_v3/`.

## Known limitations

**Inputs must be RDKit-canonical pSMILES with bare `*`.** Bracketed stars and non-canonical
orderings of the same molecule do not encode faithfully. This is the most common cause of an
installation reporting near-zero reconstruction. Pass `--canonicalize` or use
`canonical_psmiles()`.

**Multi-component strings are not supported.** The model is trained on single repeat units and
does not reproduce the `.` separator used for blends and some thermoset formulations.

**Report design properties after re-encoding.** Values along the optimization path are
predictions at a point in latent space and do not match the predictions for the molecule that
point decodes to. Across a 17-design benchmark the median surviving fraction of the predicted
gain was about one third, and four of 17 reversed sign.

**Design yield is modest.** On 16 commodity homopolymers the loop returns about one valid
design per seed, and most are chemically implausible on inspection. Large, already
fire-resistant seeds yield little, because the objective terms they already satisfy supply
almost no gradient.

**Not a substitute for cone-calorimeter testing.** The experimental data behind the property
heads is a few hundred rows and errors on unseen polymers are tens of percent. Suitable for
screening and ranking, not for certification or any decision where a wrong estimate carries
risk.

## Reproducibility

`requirements.txt` pins the exact versions the model was validated with. Reconstruction
accuracy depends on the `transformers` version in particular, because the SMILES tokenizer
subclasses `BertTokenizer`. If reconstruction comes out near zero, rebuild the environment
from that file before investigating anything else.

v1.0.0 and v2.0.0 remain available under the tags of the same names.
