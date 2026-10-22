# Fire-property prediction from SAE features

`predict_fire_properties.py` predicts four cone-calorimeter properties from a
polymer pSMILES, using the **sparse features of a Top-K SAE** rather than the raw
poly4mer embedding.

## Quick start

```bash
python predict_fire_properties.py --alpha 8 --k 160
python predict_fire_properties.py --alpha 8 --k 160 --solver ridge --ridge_lambda 1
python predict_fire_properties.py --properties sea co --alpha 32 --k 1000
```

First run embeds every pSMILES through the CLLM and caches the result under
`checkpoint/sae_cache/fire_psmiles_embeddings_<hash>.pt`, keyed by the content of
the SMILES list. Later runs with the same molecules reuse it, so only the first
invocation pays the cost of loading the 3.9 GB checkpoint.


### Solvers

`--solver sgd` (default) trains an `nn.Linear` with Adam. Note this is **not**
unregularized: `weight_decay` adds an L2 penalty to the gradient, so it is ridge
reached iteratively and stopped after a fixed epoch budget.

`--solver ridge` solves it exactly, with an unpenalized intercept:

```
w = (Xc' Xc + lambda I)^-1 Xc' yc        b = ybar - xbar' w
```

Because N < D here, the code uses the algebraically identical dual form, which
inverts an N×N matrix instead of D×D (203×203 rather than 24576×24576 at
alpha=32):

```
w = Xc' (Xc Xc' + lambda I_n)^-1 yc
```

## Options

| flag | default | meaning |
|---|---|---|
| `--alpha` | 8 | SAE expansion factor, D = alpha·768. Trained: 4, 8, 16, 32 |
| `--k` | 160 | SAE sparsity. Trained: 80, 100, 160, 200, 250, 500, 1000 |
| `--sae_ckpt` | — | override the `topk_sae_alpha{A}_k{K}.pt` path |
| `--properties` | all four | subset to run |
| `--solver` | `sgd` | `sgd` (Adam) or `ridge` (closed form) |
| `--epochs` / `--lr` / `--weight_decay` | 3000 / 1e-3 / 1e-2 | sgd solver |
| `--ridge_lambda` | 10.0 | ridge solver |
| `--target_transform` | `log` | `log` or `none` |
| `--condition_transform` | `log` | `log` or `none` for thickness/flux; standardized either way |
| `--no_conditions` | — | drop thickness/flux from tig and pkhrr |
| `--aggregate` | `none` | `mean` collapses repeats to one row per pSMILES |
| `--out` | — | write per-row predictions to csv |

`--alpha` and `--k` resolve to `checkpoint/topk_sae/topk_sae_alpha{A}_k{K}.pt`
and are validated against the config stored inside the checkpoint.

## Output

Console summary of MSE, RMSE and relative error per property. With `--out`, a csv
of every input row plus:

| column | meaning |
|---|---|
| `predicted` | prediction in the property's original units |
| `relative_error_pct` | per-row `|pred − y| / |y| × 100` |
| `property` | which property the row belongs to |




# 10-fold evaluation: `predict_fire_properties_10folds.py`

Same pipeline, run over the pre-made splits in `data/10folds/`. For each property
and each fold `data_{0..9}`, a head is trained on `train.csv`, evaluated on
`test.csv` after every epoch, and the epoch with the lowest **test** relative
error is kept. The 5 folds with the lowest test error are then summarized as
mean ± std (ddof=1). Every selected head is saved.

```bash
python predict_fire_properties_10folds.py --alpha 8 --k 160
python predict_fire_properties_10folds.py --alpha 8 --k 160 --out data/fire_10folds_alpha8_k160.csv
```

## Saved heads

`checkpoint/fire_heads_10folds/alpha{A}_k{K}/<property>_fold<i>.pt`, 40 per run,
~1.3 MB total. Each carries:

| key | meaning |
|---|---|
| `state_dict` | the selected head (lowest test rel err) |
| `state_dict_final` | the head at the last epoch |
| `code_mask` | which of the α·768 SAE features survived dead-feature dropping |
| `condition_cols`, `condition_transform` | `["thickness","flux"]` + `log`/`none`, or empty |
| `cond_mu`, `cond_sd` | standardization for the condition columns (identity on code) |
| `mu_y`, `sd_y`, `target_transform` | target log-statistics |
| `alpha`, `k`, `in_features`, `n_code`, `n_cond` | geometry |
| `best_epoch`, `train_rel`, `test_rel`, `test_final` | the fold's numbers |
| `train_csv`, `test_csv` | provenance |

`load_head(path)` in the script reloads one; its docstring gives the exact
inference sequence. Reloaded heads reproduce their recorded test error to
float32 precision. `--no_save` skips saving; `--save_dir` relocates it.


## Hyperparameters and how to tune them

### SAE settings — `--alpha`, `--k`

These select which frozen SAE the codes come from; nothing is retrained. They
resolve to `checkpoint/topk_sae/topk_sae_alpha{A}_k{K}.pt`, and the available
grid is α ∈ {4, 8, 16, 32} × k ∈ {80, 100, 160, 200, 250, 500, 1000}.

- **α** sets the code width D = α·768 and so the number of candidate features.
  After dead-feature dropping on ~200 training rows the head sees ~1000 live
  features at α=8; expect roughly proportional growth with α.
- **k** sets how many features are active per molecule. Small k gives a very
  sparse, low-rank code; large k is closer to a dense embedding.

Note that the design matrix rank is capped by the number of distinct molecules
in the training split (≤ 49, see the single-split section), regardless of α.
Raising α adds columns but not independent directions, so gains from α are
bounded and the main effect is on which features are available to the head.

Sweep recipe:

```bash
for A in 4 8 16 32; do for K in 80 160 500 1000; do
  python predict_fire_properties_10folds.py --alpha $A --k $K --no_save \
    --out data/sweep/fire_10folds_alpha${A}_k${K}.csv \
    | grep -A5 "^property   top folds"
done; done
```

Each run is a couple of minutes once embeddings are cached; the α=32 runs are
slowest because of per-epoch evaluation over ~4000 features.

### Head optimization — `--epochs`, `--lr`, `--weight_decay`

The head is `nn.Linear` trained with Adam, full-batch. `weight_decay` is Adam's
coupled L2 penalty, so the solution is ridge-regularized; `epochs` acts as an
additional early-stopping regularizer because training is not run to
convergence.

| | default | guidance |
|---|---|---|
| `--epochs` | 3000 | tig/pkhrr typically select epochs 2300–3000, so they are still improving at the end; try 5000–10000. sea/co select 1–100 — more epochs will not help them. |
| `--lr` | 1e-3 | lower (1e-4) with more epochs is the first thing to try for sea/co divergence. |
| `--weight_decay` | 1e-2 | the main capacity control. sea/co have ~30–40 training rows against ~750–970 features and want more; tig/pkhrr tolerate less. Sweep 1e-3, 1e-2, 1e-1. |
| `--eval_every` | 1 | evaluate every N epochs. Raise it to speed up large-α runs; the selected epoch gets coarser. |

The sea/co regime is the hard one. With rows ≈ rank the fit can interpolate the
training data, and a fixed-step optimizer on unscaled sparse inputs can
overshoot. Treat `(lr, weight_decay)` jointly for those two properties rather
than carrying over the tig/pkhrr settings.

### Input design — `--condition_transform`, `--no_conditions`

`--condition_transform` chooses how thickness and flux enter the tig/pkhrr head.
Both options are standardized afterwards; only the transform differs:


### Target — `--target_transform`

`log` (default) fits in log space and exponentiates predictions back. All four
properties are positive and span one to two orders of magnitude, so `none` will
let the largest values dominate the loss. Keep `log` unless there is a specific
reason not to.

### Selection — `--top`, `--folds`

`--top 5` of `--folds 10` is the specification. Reporting all 10 as well
(printed above the top-5 line) shows how much the selection is doing: for co it
halves the mean (42.75 → 20.59) and the two numbers should not be quoted
interchangeably.

