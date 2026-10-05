# Probing_FL

Probing how small BERT-like transformer encoders represent formulas of propositional logic (PL) and first-order logic (FOL), and whether **reporting bias** (obvious truths being mentioned less often) affects the model's internal picture of the world.

The pipeline is:

1. **Generate** a synthetic corpus of formulas, true or false in a randomly sampled "actual world".
2. **Train** a small encoder with masked language modelling (MLM) on the true formulas.
3. **Extract** representations (mean-pooled or `[CLS]`).
4. **Probe / plot** them (online-codelength probing, t-SNE, similarity analyses).

> Internship project. Part 1 covers PL. Part 2 covers FOL.

---

## Requirements

Python 3.9+ and:

```bash
pip install torch numpy matplotlib scikit-learn
```

All scripts must be run from the repository root (they resolve paths relative to their own location).

## Folder layout (created automatically)

```
dataset/<dataset_folder>/     # corpora          (tf_generation*.py)
model/<model_name>/           # checkpoints      (training.py)
reps/<model>/<epoch>/<type>/<dataset_folder>/   # representations (reps.py)
probe_results/                # probing JSONs    (probe.py)
plots/                        # t-SNE plots      (plot.py, plot_letters.py)
```

---

## Files

### Core library

| File | What it does |
|---|---|
| `classes.py` | PL classes: `PLetter`, `Neg`, `Conj`, `InterpretationFunc`, with `check` (truth evaluation). Adapted from course material. |
| `classes_fol.py` | FOL classes: `V` (variable), `P` (predicate with arity), `PredApp`, `Neg`, `Conj`, `Ex`, `Model`, `InterpretationFunc`, `VarAssignment`, with `check` / `check_closed`. Adapted from course material. |

These are imported by the other scripts and are not run directly.

### Corpus generation

Formulas are generated deterministically from an integer index, organised in depth "blocks", so formulas can be sampled without enumerating the whole set. Formulas use only `¬`, `∧` (and `∃` in FOL). Depth is defined recursively (atom = 1, `¬φ` = 1 + d(φ), `φ∧ψ` = 1 + max, `∃xφ` = 1 + d(φ)).

**`tf_generation.py`: propositional corpus.**
Because false formulas vastly outnumber true ones as depth grows, true and false formulas have separate, mutually recursive index tables, so true formulas are sampled directly.

```bash
python tf_generation.py \
  --number_pl 20 --min_depth 2 --max_depth 3 \
  --corpus_size 6000 --n_worlds 1000 --prop_tf 0.5 \
  --folder_name pl_example
```

- `--number_pl`: number of propositional letters
- `--prop_tf`: proportion of letters that are true in each world
- `--n_worlds`: number of worlds sampled; one is the actual world, the rest are alternative worlds
- `--corpus_size`: size of train (true only), `dev_t` (true) and `dev_f` (false)

**`tf_generation_fol.py`: first-order corpus.**
Closed formulas only (closedness is enforced during generation by tracking bound variables). No individual constants; all predicate arguments are quantified variables. Truth is checked by rejection sampling.

```bash
python tf_generation_fol.py \
  --number_id 10 --number_vr 8 --number_pr 20 \
  --min_depth 2 --max_depth 5 --min_arity 1 --max_arity 3 \
  --corpus_size 6000 --n_worlds 1000 --prop_tf 0.5 \
  --folder_name fol_example
```

- `--number_id`: domain size (individuals `0..n-1`)
- `--number_vr`: number of variables. Keep this larger than the maximum number of nested `∃`, otherwise `filter_uw_fol.py` cannot rename shadowed variables.
- `--prop_spread`: each predicate gets a random number of true tuples (from 2 up to `prop_tf` of all tuples) instead of a fixed proportion
- Predicates are named `A, B, C, …` and sorted by arity; variables are `a, b, c, …`

Both generators write to `dataset/<folder_name>/`:
`train.pkl`, `dev_t.pkl`, `dev_f.pkl` (formula indices), `probs.pkl` (reporting-bias weights, see below), `act_world.pkl`, `alt_worlds.pkl`, `params.pkl`.

**Reporting bias weights** (`probs.pkl`): for each true training formula, the fraction of alternative worlds in which it is *false*, i.e. `(N-n)/N`. Formulas true in many alternative worlds ("obvious" ones) get lower weight. These are only used by `training.py --r_bias`.

**`alt_truths.py`: PL only.** Computes, for every `dev_t`/`dev_f` formula, its truth value in each alternative world and saves it to `dataset/<folder>/alt_truths.pkl`.

```bash
python alt_truths.py --folder pl_example
```

### Training

**`training.py`** trains the MLM encoder (learned token and positional embeddings, 15% masking, AdamW, patience-based early stopping with LR halving on each bad epoch). Dev loss is measured on `dev_t` with a fixed masking pattern.

```bash
# PL, with [CLS], boosted reporting bias (weights ** 8)
python training.py --logic pl --dataset_folder pl_example \
  --cls --r_bias --bias_power 8 \
  --hidden 16 --heads 4 --layers 4 --batch_size 64 --epochs 100 --seed 0

# FOL
python training.py --logic fol --dataset_folder fol_example \
  --hidden 128 --heads 4 --layers 2 --epochs 100 --seed 0
```

| Flag | Meaning |
|---|---|
| `--logic {pl,fol}` | which generator/alphabet to use |
| `--dataset_folder` | corpus folder under `dataset/` |
| `--cls` | prepend a `[CLS]` token |
| `--r_bias` | sample training batches with weights from `probs.pkl` (reporting bias); without it, sampling is uniform |
| `--bias_power` | exponent applied to the weights (`8` = "boosted" version) |
| `--tokenization {char,bigram,bigram2}` | character-level, overlapping bigrams, or non-overlapping bigrams |
| `--hidden/--heads/--layers/--batch_size/--epochs` | model and training size |
| `--seed` | model init and sampling; use the same seed across conditions for fair comparison |
| `--patience` | epochs without dev improvement before stopping |

Output goes to `model/<bs>bs_<epochs>e_<hidden>hl_<heads>h_<layers>l<suffix>_seed<seed>_<dataset_folder>/` with one `epoch_N.pt` per epoch, plus `params.pkl`, `vocab.pkl` and `loss_plot.png`. That folder name is what the `--model` / `--model_dir` arguments below expect.

### Extracting representations

**`reps.py`** loads a checkpoint and encodes a set of formulas, saving mean-pooled representations (and `[CLS]` if `--cls`). With a `[CLS]` model, `[CLS]` is excluded from the mean.

```bash
python reps.py --model <model_name> --epoch 10 --dataset_name dev_t --cls
python reps.py --model <model_name> --epoch 10 --dataset_name dev_f --cls
```

- `--dataset_name`: `dev_t`, `dev_f` or `train`
- `--dataset_folder`: represent a different corpus than the one the model was trained on (must share domain/variables/predicates, or number of letters for PL)
- `--alt_world k`: re-instantiate the same formula indices as if alternative world `k` were the actual world, saved with suffix `_k` (needed for probing Task 2)

Output: `reps/<model>/<epoch>/{mean,cls}/<dataset_folder>/<dataset_name>[_k]`, a pickle with `{"indexes", "type", "reps"}`.

### Probing

**`probe.py`** runs the **online-codelength (MDL) probing** of Voita & Titov (2020). An MLP probe (2 × 1000 hidden units by default) is trained on growing prefixes of the training data; each next block is encoded with the current probe, and the sum gives the online codelength. The **compression** is the uniform codelength divided by the online codelength. Higher means the information is more easily extractable. It also reports the dev accuracy of a standard probe trained on all data.

```bash
# Task 1: true/false in the actual world
python probe.py --task 1 --model <model_name> --epoch 10 --rep_type mean

# Task 2: true/false in alternative world 0 (needs reps.py --alt_world 0 for dev_t and dev_f)
python probe.py --task 2 --model <model_name> --epoch 10 --rep_type cls --w_idx 0
```

Useful options: `--dataset_folder` (defaults to the one the model was trained on), `--hidden`, `--hidden_layers`, `--lr`, `--batch_size`, `--max_epochs`, `--patience`, `--seed`, `--dev_frac`, `--min_first_block`.

Result: `probe_results/<model>-<epoch>-<dataset>-<rep_type>-<task>[-w<idx>].json`, with codelength, compression, per-block bits/accuracy and the config used.

### Visualisation (PL)

**`plot.py`** t-SNE of formula representations, coloured true vs. false.

```bash
python plot.py \
  --true-reps  reps/<model>/10/mean/<dataset>/dev_t \
  --false-reps reps/<model>/10/mean/<dataset>/dev_f \
  --output my_run --n-sample 10000
```

Saved to `plots/<output>/`. Both inputs must be the same representation type.

**`plot_letters.py`** t-SNE of the non-contextual (static) embeddings of the tokens containing each propositional letter, coloured by letter. Written for **bigram / bigram2** models; by default it only plots tokens that actually occur in the corpus (`--all_letters` overrides).

```bash
python plot_letters.py --model <bigram_model_name> --epoch 19
```

Saved to `plots/<model>_epoch<N>_tsne_letters.png`.
