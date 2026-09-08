# NegationLLM

**The Geometry of 'not': Does Negation Transfer Between Encoders?**

Does negation occupy a stable, low-dimensional subspace of a model's representation space — and if so, can
that subspace be **transferred between independently trained encoders** using an alignment learned on
affirmative sentences alone?

---

## Setup

One command builds the environment:

```bash
./setup/setup.sh
```

Run it from the repository root. The leading `./` matters: it tells the shell to execute *this* file rather
than look for a command called `setup.sh` on your `PATH`. The environment is created at the repository
root regardless of the directory you launch it from.

Then activate it and open the notebooks:

```bash
source .venv/bin/activate
jupyter lab code/
```

`source` is what switches your shell to the project's Python — without it, `python` and `jupyter` remain
the system ones. You will know it worked when your prompt gains a `(negllm)` prefix. To leave, type
`deactivate`. You need to re-activate in every new terminal window; the notebooks themselves just need the
**Python (NegationLLM)** kernel, which the script registers for you.

Budget a few minutes on the first run — almost all of it downloading torch (~2.5 GB).

### Options

| Flag | Effect |
|---|---|
| `--force` | delete and rebuild `.venv` from scratch |
| `--no-jupyter` | skip Jupyter kernel registration |
| `--requirements F` | install from `F` instead of `setup/requirements.txt` |
| `--ascii` | plain ASCII output for terminals that mangle unicode |
| `-h`, `--help` | usage |

The script has no dependencies of its own — it has to run before an environment exists — so it is plain
bash 3.2, the version macOS ships. It installs packages one at a time so a failure names the culprit,
verifies every import afterwards (installed ≠ importable), detects the torch device, and writes a full log
to `.setup.log`. Exit codes: `0` success, `1` failure, `2` bad usage.

---

## Layout

```
├── setup/
│   ├── setup.sh                        environment bootstrapper
│   └── requirements.txt                dependency stack (tested versions in brackets)
├── code/
│   ├── 00_load_nevir.ipynb             NevIR → oriented, categorised minimal pairs → parquet
│   └── 01_negation_subspace_rank.ipynb the rank experiment ("manip 0")
└── data/                               created on first run; not tracked
    ├── nevir_pairs.parquet             produced by notebook 00
    └── cache/                          embedding cache + saved results (safe to delete)
```

Run `00_load_nevir.ipynb` first: it creates `data/` and the parquet that notebook 01 consumes. Without it,
notebook 01 falls back to a small built-in demo dataset (it says so in a banner) whose numbers are
meaningless — N is far too small relative to the embedding dimension.

---

## The experiment

### Why rank is the question that comes first

Three papers make three **mutually incompatible** implicit assumptions about the shape of negation, and
none of them tests it:

| Source | Implicit assumption | Predicted geometry |
|---|---|---|
| Sammani et al., *negation steering* | one linear probe weight vector | rank 1 — a single universal direction |
| Aggarwal et al., *Seeing What's Not There* | `e_neg` recomputed per sentence | effectively unbounded rank |
| Petcu et al., *negation taxonomy* | negation types differ sharply in difficulty | one sub-axis per type |

The answer decides the rest of the project: rank 1 means a single Householder reflection and a classic
orthogonal Procrustes transfer; a low-rank subspace means a soft mixture of type subspaces and
CCA + Procrustes; no linear structure means a non-linear mapper. So it is measured first.

### Notebook 00 — building the dataset

Pulls NevIR from the HuggingFace `datasets-server` REST API. It harvests the **queries**, not the corpus:
NevIR passages run to several hundred tokens while CLIP truncates at 77, and the negation edit usually sits
past the cut — Δ would then be measured on text that no longer contains the negation.

Two problems it solves:

* **Orientation is not given.** `q1`/`q2` does not consistently run affirmative → negated (`1000-2` goes
  positive → negative, `1000-3` goes `unoccupied` → `occupied`). A flipped sign leaves the spectrum intact
  but destroys the mean displacement — the entire rank-1 signal. Every pair is oriented explicitly and the
  undecidable ones are flagged.
* **No category labels.** Pairs are classified with the project taxonomy using the *diff* between the two
  sentences, which localises the edit far better than a lexicon scan (otherwise `international` and
  `district` both fire the `in-` prefix rule).

Output: 1383 pairs — `particle_not` 622, `other_substitution` 394, `affixal_prefix` 177, `implicit_verb` 68,
`quantifier_no` 52, `adverb_never` 35, `prep_without` 18, `exclusion` 14.

### Notebook 01 — the rank experiment

Consumes any parquet with this schema (`affirmative`, `negative`, `category` required):

| Column | Description |
|---|---|
| `affirmative` / `negative` | the minimal pair |
| `category` | negation mechanism |
| `subclass` | finer subclass (`im_allomorph`, `modal`, …) |
| `object` | content control — enables the `category × object` variance decomposition |
| `is_distractor` | `True` for `mis-` / `anti-`, which look negative but are not negation |

Pipeline: CLIP text embeddings → Δ = e(neg) − e(aff) → spectrum → information-theoretic dimension
estimators → null models → cross-validated rank → type structure → verdict (H1–H4).

Two design points worth knowing before reading the output:

* **The mean of Δ *is* the translation vector of H1**, so running PCA only on centred Δ would delete the H1
  signal before measuring it. The notebook reports the translation share ρ *and* both spectra.
* **A raw scree plot proves nothing** when N ≲ d: structureless vectors also produce a decaying spectrum
  (Marchenko–Pastur). Significance is established against four nulls — shuffled pairs, unrelated affirmative
  pairs, **content-matched** affirmative pairs (same source passage), and isotropic Gaussian.

---

## Results so far — NevIR × CLIP ViT-B/32

986 pairs after dropping `other_substitution` (not negation) and categories with fewer than 5 pairs.

| | final projected space | layer 4 (peak probe) |
|---|---|---|
| translation share ρ | 0.035 | 0.071 |
| top-1 variance share | 0.062 | 0.121 |
| Shannon effective rank / participation ratio | 177.8 / 79.5 | 102.6 / 34.7 |
| significant rank vs shuffled / affirmative / **matched** | 0 / 0 / **0** | 0 / 0 / **0** |
| significant rank vs gaussian | 3 | 6 |
| linear probe accuracy | 0.686 | **0.753** |

**Variance decomposition: source passage 64%, negation category 2.8%, residual 33%.**

Reading: in CLIP's text space, on naturalistic NevIR queries, the negation displacement is **not**
low-dimensional — its spectrum is no more concentrated than that of a content-matched, negation-free
difference. Negation remains linearly *detectable* (probe 0.75 at layer 4), which is not a contradiction:
linear separability does not imply a low-dimensional displacement.

Two causes this experiment cannot separate: NevIR pairs are not strict minimal edits (rewritten questions,
multi-token substitutions), and subtraction does not cancel content in CLIP.

**The hypothesis this generates is more interesting than the measurement.** Sammani et al. obtain their
single direction from 4000 *templated* COCO caption pairs — far cleaner minimal pairs than NevIR. If a
purpose-built controlled dataset yields rank ≈ 1 in the same model where NevIR yields ≈ 100, then the
"negation direction" reported in the literature is an artefact of the template rather than a property of the
model. Half of that experiment is already built.

---

## Next

1. **Generate the controlled dataset** — factorial `category × object`, balanced N per category, imposed
   lexicon per category, `mis-`/`anti-` distractors included. Drop it in `data/` and point
   `CONFIG["parquet_path"]` at it; nothing else changes.
2. **Templated vs naturalistic comparison** — the experiment above.
3. **Involution test** — build the operator the verdict implies (translation / projection / reflection) and
   check which satisfies `f(f(x)) ≈ x` on real double negations. None of the nine reviewed papers tests it.
4. **Second encoder** — if the effective rank matches before any alignment, the dimensionality of negation is
   a property of the task, not of one architecture.
5. **Cross-model transfer** — only then, with the machinery the verdict selected.

---

## Troubleshooting

**`transformers` v5 changed the API.** `get_text_features` now returns an output object rather than a
tensor; its `pooler_output` holds the projected joint-space embedding (verified identical to
`text_projection(final_layer_norm(EOS))`). The notebooks handle both v4 and v5.

**Apple Silicon.** The device is auto-detected (`mps` → `cuda` → `cpu`); nothing to configure.

**HuggingFace rate limits.** Unauthenticated requests are throttled. `export HF_TOKEN=...` if model
downloads or datasets-server calls start failing.

**Re-running is cheap.** Embeddings are cached under `data/cache/`, keyed by model, layer and dataset
fingerprint. Delete that directory to force recomputation.

**`permission denied` when running the script.** `chmod +x setup/setup.sh`.
