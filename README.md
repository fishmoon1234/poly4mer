# poly4mer `v3.0.0`

A multi-physics synthesis based chemical language model for polymer fire property prediction and inverse design, with conditioning on environmental parameters (heat flux and sample thickness).

![Poly4mer overview.](fig1_poly4mer.png)

*Overview. **Top:** pre-training puts physics in three places at once, in the data, in the
joint objective, and in the architecture, where the property heads are made monotone in heat
flux by taking a running minimum or maximum over 17 flux knots and interpolating at the
requested flux. **Bottom left:** fine-tuning for property prediction freezes the encoder,
decoder and token predictor and updates only the property heads on experimental data.
**Bottom right:** inverse design runs gradient descent on the latent with every weight frozen,
decoding each iterate into a candidate pSMILES. Source figure:
[`fig1_poly4mer.pdf`](fig1_poly4mer.pdf).*

This repository houses the code for our work in forward prediction and inverse polymer design using chemical language models. The released `v1.0.0` and `v2.0.0` code stays available under the tags of the same names.

**Highlights**:

1. The model is developed based on [SMI-TED](https://github.com/IBM/materials/tree/main), and trained to predict 4 flammability metrics: time to ignition (Tig), peak heat release rate (pHRR), smoke extinction area (SEA), and carbon monoxide yield (CO) from polymer SMILES.
2. We introduce a principled multi-physics synthesis framework to data-scarce learning by embedding domain knowledge directly into language model training, enabling robust generalization beyond pure data-driven chemical language models.
3. We design a new strategy for introducing new tokens into pretrained language models via encoder-decoder architecture expansion, leveraging monomer representations to capture complex polymer semantics.
4. We architect an autoencoding system coupling predictive modeling with generative design, enabling inverse polymer design via latent-space exploration and structure reconstruction for targeted applications.
5. Predictions and inverse design are conditioned on the heat flux and sample thickness of the test setup, so the same polymer can be evaluated under multiple fire-test scenarios.
6. The encoder, decoder and the four property predictors now ship as a **single checkpoint**, and the property predictors read the latent directly, so no separate normalization file is needed at inference.
7. Optional **hard-monotone** property heads for Tig and pHRR enforce the physical dependence on heat flux by construction; see `[poly4mer_monotone/](poly4mer_monotone/README.md)`.

> **Two kinds of "predictor".** In the checkpoint, the key `predictor` is the **token predictor**, the last part of the decoder module. The **property predictors** are stored under `property_regressors`.



## What changed since `v2.0.0`


|                          | `v2.0.0`                                                    | `v3.0.0`                                                      |
| ------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------- |
| Weight files             | 6 (4 property heads, 1 decoder bundle, 1 scaler)            | 1, `poly4mer_v3.ckpt`                                         |
| Property predictor input | whitened latent, needs `scalers.pt`                         | raw latent, statistics travel inside the checkpoint           |
| SEA predictor            | 770 inputs (latent + thickness + flux)                      | 768 inputs (latent only)                                      |
| Input pSMILES            | as written                                                  | must be RDKit-canonical with bare `*`                         |
| Scripts                  | `fire_property_predict_v2.py`, `fire_optimize_smiles_v2.py` | `fire_property_prediction_V3.py`, `fire_inverse_design_v3.py` |




## Requirements

A Python package requirements file (`requirements.txt`) is provided.

```bash
pip install -r requirements.txt
```

Key dependencies:

- Python 3.10.18
- torch==2.8.0+cu128, from the PyTorch wheel index named at the top of `requirements.txt`
- numpy==2.2.6
- transformers==4.53.2
- tokenizers==0.21.2
- rdkit==2025.3.3
- pandas==2.3.3

A CUDA GPU is recommended: about 4 GB of memory for inference, about 16 GB for training.

> **Reconstruction accuracy depends on the exact** `transformers` **version**, because the SMILES tokenizer subclasses `BertTokenizer`. If reconstruction comes out near 0% in an environment of your own, re-create the environment from `requirements.txt` before investigating anything else.



## Pretrained models

The SMI-TED Python package is bundled with this repository under `smi_ted_light/`; you only need to download the model weights.

Download the following from our [HuggingFace mirror](https://huggingface.co/fishmoon1234/poly4mer) and place them at the paths shown:

- `poly4mer_v3.ckpt` (3.7 GB) → `checkpoint/`
  - This is the only v3 checkpoint you need. It carries the encoder, the decoder and
  **both** property head sets, so the same file serves property prediction, inverse
  design and the cross-validation reproduction. Every script loads it by default and
  picks the right head set for the task; see *Two head sets* below.
- Pretrained SMI-TED encoder (1.1 GB) → `smi_ted_light/`
  - `smi-ted-Light_40.pt`
- *Optional*, the per-fold property heads, `property_heads_10folds/<p>/` on the hub
  (40 files, 126 MB) → `checkpoint/fire_property_heads/<p>/`, for `tig`, `pkhrr`, `sea`, `co`
  - needed only to compare your own re-training against ours; prediction and design do not use them
- *Optional*, the monotone heads, `monotone_heads_10folds/<p>/` on the hub
  (20 files, 63 MB) → `poly4mer_monotone/checkpoint/<p>/last_two_linear/`, for `tig` and `phrr`
  - see [poly4mer_monotone/README.md](poly4mer_monotone/README.md)

All weights are ignored by git. After downloading, the layout should look like:

```
poly4mer/
|-- checkpoint/
    |-- poly4mer_v3.ckpt
|-- smi_ted_light/
    |-- bert_vocab_curated.txt
    |-- fast_transformers/
    |-- load.py
    |-- smi-ted-Light_40.pt
    ...
|-- poly4mer_monotone/
    |-- checkpoint/
        |-- tig/last_two_linear/
        |-- phrr/last_two_linear/
    |-- fire_property_prediction_monotone.py
    |-- run_property_prediction_monotone.sh
    |-- README.md
|-- data/
    |-- 10folds_data/
        |-- tig/ pkhrr/ sea/ co/      data_0 .. data_9, each with train.csv + test.csv
|-- training/
    |-- train_poly4mer_v3_property_reg.py
    |-- train_property_predictor_finetune.py
|-- evaluate_fire_prediction_heads.py
|-- fire_property_prediction_V3.py
|-- fire_inverse_design_v3.py
|-- evaluate_reconstruction_v3.py
|-- models.py
|-- utils1.py
|-- example_pSMILES_homo.txt
|-- run_property_prediction.sh
|-- run_design.sh
|-- run_demo_property_heads.sh
|-- example_pSMILES_gavit.txt
|-- requirements.txt
```



## Datasets

The data behind the paper is on the HuggingFace Hub at
[fishmoon1234/Polymer_Fire_Properties](https://huggingface.co/datasets/fishmoon1234/Polymer_Fire_Properties):

| File | Rows | Contents |
|---|---:|---|
| `v1_1.7M_synthetic_structure.smi` | 1,735,755 | Enumerated repeat-unit structures, one pSMILES per line |
| `v2_110k_FDS.smi` | 109,993 | Simulated cone-calorimeter cases; 55,000 repeat units at two exposures each |
| `experimental_flux_thickness.smi` | 203 | Measured cone-calorimeter data, 49 polymers |
| `10folds_data/` | | The ten-fold experimental splits, one folder per property |

Nothing has to be downloaded to run the scripts here: `data/10folds_data/` in this
repository already carries the fold splits, file for file identical to `10folds_data/` on
the hub. Only the folder names differ, `tig/ pkhrr/ sea/ co/` here against
`tig_exp_data/ phrr_exp_data/ SEA_exp_data_avg/ CO_exp_data_avg/` there.

The dataset card documents the column layout and units of each file, and the two caveats
worth reading before training on `v2`: runs that never ignited carry a censored `tig` of
exactly 900.0 s, and `sea` and `co` do not vary with the exposure.

```bash
hf download fishmoon1234/Polymer_Fire_Properties --repo-type dataset --local-dir polymer_fire_properties/
```


## Model components


| Module                  | Parts, in order                                                           | Where the weights are                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Encoder module**      | SMI-TED token embedder, then `star_encoder`, then the autoencoder encoder | SMI-TED: `smi_ted_light/smi-ted-Light_40.pt`, frozen. The rest: `star_encoder` and `autoencoder_encoder` in `poly4mer_v3.ckpt` |
| **Decoder module**      | autoencoder decoder, then the token predictor                             | `decoder` and `predictor` in `poly4mer_v3.ckpt`                                                                                |
| **Property predictors** | four MLP heads on the latent `z`, for Tig, pHRR, SEA, CO                  | `property_regressors` and `property_regressors_selected`, with shared `property_stats`, in `poly4mer_v3.ckpt`                  |




### Two head sets

The checkpoint carries two sets of property heads. They share the same architecture and the
same standardization statistics, and differ only in weights.


| Key                            | What it is                                                                                            | Used by                                                  |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `property_regressors`          | the model's own heads, produced with the rest of the model                                            | the cross-validation reproduction, as its starting point |
| `property_regressors_selected` | per-property heads chosen by 10-fold cross-validation on the experimental data, one fold per property | property prediction and inverse design, by default       |


`--heads selected` (the default) and `--heads joint` switch between them in
`fire_property_prediction_V3.py` and `fire_inverse_design_v3.py`. Each run prints which set
it loaded.

The reproduction in `--cv` starts from `property_regressors`, because the selected heads were
themselves chosen from these same folds; starting from them would be circular. `--cv_init_heads`
can override that, and warns when it does.

The selection criterion behind `property_regressors_selected` scores each head on the whole
property set, most of which it trained on. Those selection errors are optimistic and are not
held-out estimates; see the caveat under the cross-validation section below.

The property predictors consume the **raw** latent `z`. Only thickness and flux are standardized, using the statistics stored alongside the heads:


| Head | Input                            | Dimension |
| ---- | -------------------------------- | --------- |
| Tig  | `[z, thickness_norm, flux_norm]` | 770       |
| pHRR | `[z, thickness_norm, flux_norm]` | 770       |
| SEA  | `z`                              | 768       |
| CO   | `z`                              | 768       |


Each head returns a standardized value; the real value is `output * y_std[k] + y_mean[k]`.

## Input pSMILES format

Inputs must be **RDKit-canonical pSMILES with bare** `*` **attachment points**, for example `*CC(*)c1ccccc1`. Bracketed stars such as `[*]` and non-canonical orderings of the same molecule do not encode faithfully and will not reconstruct.

`fire_property_prediction_V3.py` takes `--canonicalize` to rewrite inputs first, and `evaluate_reconstruction_v3.py` exports `canonical_psmiles()` for use in your own code.

A design is accepted when the decoded unit has **two or more attachment points**: two is a linear repeat unit, three a branch point, four or six a network former. Units with zero or one attachment point cannot propagate a chain and are rejected. The count is reported for every design.

## Running experiments

**Property predictor**:

1. Add the pSMILES of polymers to be predicted in

```
example_pSMILES_homo.txt
```

The file is a CSV with columns `mol_id, mol_name, smiles`.

1. Edit the polymer list and the test conditions in

```
run_property_prediction.sh
```

The test conditions are the heat flux (`FLUX`) and the sample thickness (`THICKNESS`); both are passed to the model so the same polymer can be queried under different fire-test scenarios. Set `POLYMERS="ALL"` to predict for every row of the file.

1. Run the predictor with the shell script

```
./run_property_prediction.sh
```

1. Predictions are printed and written to `predictions.csv`.

A single pSMILES can also be queried directly:

```bash
python fire_property_prediction_V3.py --smiles '*CC(*)C' --thickness 3 --flux 50
```

**Polymer design**:

1. Add the pSMILES of candidate initialization polymers in

```
example_pSMILES_homo.txt
```

1. Edit the polymer list in

```
run_design.sh
```

together with the penalty parameters, the test conditions, and the optimization hyperparameters.

The test conditions are the heat flux and the sample thickness:

```
FLUX=50.0
THICKNESS=3.0
```

For the penalty parameters, one can tune

```
C_TIG=0.2
C_QPUA=0.6
C_SEA=0.1
C_CO=0.1
```

to change weights between different fire properties. Weight by how far each property sits from its bound rather than by importance: a property already near its optimum contributes little gradient, and weighting it heavily starves the search.

For the optimization hyperparameters, one can change the maximum optimization steps (`MAX_EPOCHS`) and the step size, which is set per polymer in the `RUNS` list.

1. Run inverse design with the shell script

```
./run_design.sh
```

1. Candidate polymers are stored in

```
results/optimized_smiles_canonicalized/tig${C_TIG}_phrr${C_QPUA}_sea${C_SEA}_co${C_CO}_flux${FLUX}_thickness${THICKNESS}/optimized_smiles_canonicalized_${name}_${idx}_lr_${lr}.txt
```

The optimization path and estimated fire property values can be found in

```
results/log_optimized_prop/.../log_optimized_props_${name}_${idx}_lr_${lr}.txt
```

and the accepted designs, with their discovery step, attachment-point count and predicted properties, in

```
results/designs/.../designs_${name}_${idx}_lr_${lr}.json
```

> **Report the properties of the design, not of the latent point.** The property values along the optimization path are predictions at a point in latent space, and they do not in general match the predictions for the molecule that point decodes to. Re-encode each candidate and predict again before quoting any number. `--clamp` clips predictions to physical ranges before the loss; note that it does not keep the latent inside the decodable region, so the property log records both the raw and the clipped value.

**Reconstruction check**:

To verify an installation, or to measure reconstruction on your own list of pSMILES:

```bash
python evaluate_reconstruction_v3.py --files homo=example_pSMILES_homo.txt
```

This reports exact match, token accuracy, validity, same-molecule rate, and how many inputs were already in canonical bare-star form.

**Training**:

`training/train_poly4mer_v3_property_reg.py` trains the full model, and
`training/train_property_predictor_finetune.py` fine-tunes only the property heads on labeled
data with the encoder frozen. Both read `checkpoint/poly4mer_v3.ckpt` by default. See the
docstring at the top of each script for the expected data format.

## Reproducing the cross-validation tables

`data/10folds_data/` carries the 10-fold experimental splits for all four properties, the
same ones that selected the heads bundled in `poly4mer_v3.ckpt`, and the same ones
published at [fishmoon1234/Polymer_Fire_Properties](https://huggingface.co/datasets/fishmoon1234/Polymer_Fire_Properties).
tig and pHRR have 228 rows each with thickness and flux; SEA and CO are the per-polymer averaged sets, 44 and 32 rows,
and carry no test conditions because those heads take the latent alone.

Run the whole thing as a demo, which scores the bundled heads, re-trains one property's
10 folds with the same recipe, and prints the two side by side:

```bash
./run_demo_property_heads.sh                       # ~15 min, re-trains CO
DEMO_FOLDS=0,1,2 ./run_demo_property_heads.sh      # ~5 min
SKIP_TRAIN=1 ./run_demo_property_heads.sh          # scoring only, under a minute
```

Score the shipped per-fold heads on their own:

```bash
python evaluate_fire_prediction_heads.py --csv folds.csv
```

This prints, per property, the full per-fold table of train and test relative error, the mean
over the best 5 folds, the mean over all 10, and the best overall relative error with the fold
that achieved it. Where a head ships with its recorded result the script prints both so the
two can be compared.

Re-run the training that produced them:

```bash
python training/train_property_predictor_finetune.py --cv --out_dir runs/cv
python training/train_property_predictor_finetune.py --cv --properties tig --cv_folds 8 --out_dir runs/tig8
```

`--cv` sweeps the per-property learning-rate grid on each fold and keeps the epoch with the
lowest error on that fold's test rows, starting every head from the one already in the
checkpoint. The per-property grids, epoch budgets and batch sizes are in `CV_RECIPE` near the
top of that file.

> **Two caveats when quoting these numbers.** The epoch and learning rate are selected on the
> same test fold the error is then read from, so per-fold test errors are optimistically
> biased. And "best overall relative error" scores each head on the entire property set,
> roughly 90% of which it trained on, so it is optimistic by construction; it is reported
> because it is the criterion that picked the shipped heads, not as an estimate of
> generalization. Prefer the all-10 mean when a single number is needed.



## Monotone property heads (optional)

`[poly4mer_monotone/](poly4mer_monotone/README.md)` holds alternative Tig and pHRR heads that
are hard-monotone in heat flux: Tig can only fall as flux rises, pHRR can only rise. The
constraint is structural rather than a penalty, so it holds for every input. The encoder is
the same file as the main model; only the heads differ, and SEA and CO are not covered.

```bash
cd poly4mer_monotone
./run_property_prediction_monotone.sh
```



## Inverse design examples

`run_design.sh` runs five worked examples and prints every design it finds:


| Seed      | lr   | Weights tig/pHRR/SEA/CO | Designs                  |
| --------- | ---- | ----------------------- | ------------------------ |
| PMMA      | 0.3  | 0.2/0.6/0.1/0.1         | 2 linear                 |
| PP        | 1.0  | 0.2/0.6/0.1/0.1         | 2 linear                 |
| NYLON     | 0.3  | 0.2/0.6/0.1/0.1         | 2 linear                 |
| BPA-ether | 0.1  | 0.2/0.6/0.1/0.1         | 1 linear                 |
| PEI-N     | 0.01 | 1/1/1/1                 | 1 linear, 1 branch point |


```bash
./run_design.sh
ONLY="PMMA PP" ./run_design.sh        # a subset
MAX_EPOCHS=500 ./run_design.sh        # quick smoke test, finds fewer design
```

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

