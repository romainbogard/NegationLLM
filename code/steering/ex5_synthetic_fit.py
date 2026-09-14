"""The honest version: fit on a dataset we build, evaluate on the whole benchmark, tune nothing on it.

Two rules this file exists to respect.

1. NOTHING IS FITTED ON THE BENCHMARK. Every scalar is fitted on synthetic questions generated here
   from the project's own word bank, with a surrogate "image" (see below). NegBench is then scored
   once, on all 5914 questions — no split, because no split is needed when nothing was tuned on it.

2. NO BENCHMARK-SPECIFIC TAXONOMY. The earlier per-class version conditioned its scalars on
   positive/negative/hybrid, which is NegBench COCO's own annotation scheme — a method that needs to
   know a benchmark's caption templates cannot transfer. Here every coefficient is instead an affine
   function of (n_aff, n_neg), the number of asserted and negated spans the parser finds in the
   sentence. That is a property of any sentence in any dataset, and it *contains* the old taxonomy as
   a special case (n_neg = 0 is "positive", n_aff = 0 is "negative", both > 0 is "hybrid") without
   naming it.

The surrogate image: CLIP is a joint space, so the embedding of a plain description of the intended
scene ("a photo of a cat") stands in for the image of that scene while fitting. It is a stand-in, and
whether it transfers to real images is exactly what the benchmark then measures.
"""
import sys
from pathlib import Path

import numpy as np

import common
import ex2_compositional_spans as ex2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from make_conjunction_dataset import WORDS, phrase          # noqa: E402

RNG = np.random.default_rng(0)
NOUNS = [w for c in WORDS.values() for w in c]

# Our own phrasings. They have to be English sentences about presence and absence, so some overlap
# with any such benchmark is unavoidable, but nothing here is copied from the NegBench file.
AFF = ["a photo of {x}", "this picture shows {x}", "there is {x} in this picture",
       "an image containing {x}"]
NEG = ["this picture does not show {y}", "there is no {y} in this picture",
       "a photo without {y}", "{y} is absent from this picture"]
HYB = ["this picture shows {x} but there is no {y}", "a photo of {x} without {y}",
       "there is {x} in this picture, but no {y}", "this picture shows {x} and does not show {y}"]


def make_synthetic(n=1500):
    """n questions, each: a true scene (x present, y absent) and four options, one correct."""
    caps, gold, scene = [], [], []
    for _ in range(n):
        x, y = RNG.choice(NOUNS, size=2, replace=False)
        px, py = phrase(x), phrase(y)
        kind = RNG.choice(["aff", "neg", "hyb"])
        if kind == "aff":
            right = RNG.choice(AFF).format(x=px)
            wrong = [RNG.choice(AFF).format(x=py), RNG.choice(NEG).format(y=px),
                     RNG.choice(HYB).format(x=py, y=px)]
        elif kind == "neg":
            right = RNG.choice(NEG).format(y=py)
            wrong = [RNG.choice(NEG).format(y=px), RNG.choice(AFF).format(x=py),
                     RNG.choice(HYB).format(x=py, y=px)]
        else:
            right = RNG.choice(HYB).format(x=px, y=py)
            wrong = [RNG.choice(HYB).format(x=py, y=px), RNG.choice(AFF).format(x=py),
                     RNG.choice(NEG).format(y=px)]
        opts = [right] + wrong
        order = RNG.permutation(4)
        caps += [opts[i] for i in order]
        gold.append(int(np.where(order == 0)[0][0]))
        scene.append(f"a photo of {px}")                     # the surrogate image
    return caps, np.array(gold), scene


def count_basis(D_like, flat):
    """Basis matrices, each replicated with weights 1, n_aff and n_neg -> coefficients that are
    affine in the span counts. 6 base terms x 3 = 18 scalars, none of them benchmark-specific."""
    parsed = [ex2.parse_spans(c) for c in flat]
    A, N_, na, nn = ex2.span_embeddings(parsed, "ens")
    ANC = np.tile(common.anchor(), (len(flat), 1))
    base = {"aff": A, "anc": ANC, "anc_na": na[:, None] * ANC, "anc_nn": nn[:, None] * ANC,
            "neg": N_, "cap": D_like["CAP"]}
    mats = {}
    for k, v in base.items():
        mats[k] = v
        mats[f"{k}*na"] = na[:, None] * v
        mats[f"{k}*nn"] = nn[:, None] * v
    return mats


# ---------------------------------------------------------------- fit on synthetic only
caps, gold_s, scene = make_synthetic()
CAP_s = common.embed_texts(caps)
IMG_s = common.embed_texts(scene)
n_s = len(gold_s)
D_s = {"flat": caps, "CAP": CAP_s, "IMG": IMG_s, "img_idx": np.arange(n_s), "gold": gold_s,
       "tmpl": np.array(["synthetic"] * n_s), "tr_rows": np.arange(n_s), "te_rows": np.arange(n_s),
       "n": n_s}
mats_s = count_basis(D_s, caps)

In_s = common.l2n(IMG_s)
best = (-1, None, None)
for c in np.round(np.arange(0.0, 1.001, 0.1), 2):
    b = ex2.Basis(D_s, mats_s, IMG=In_s - c * In_s.mean(0))
    coef, a = ex2.multistart(D_s, b, [np.eye(len(mats_s))[0]], n_random=4, rounds=12)
    print(f"  c={c:.1f}  synthetic fit {100 * a:.2f}%")
    if a > best[0]:
        best = (a, c, coef)
fit_acc, c_star, coef = best
print(f"\nfitted on {n_s} synthetic questions only: {100 * fit_acc:.2f}%   c* = {c_star}")

# ---------------------------------------------------------------- score the benchmark, untouched
D = common.load_mcq()
mats_b = count_basis(D, D["flat"])
In = common.l2n(D["IMG"])
IMG_b = In - c_star * In.mean(0)
b_b = ex2.Basis(D, mats_b, IMG=IMG_b)
CAPmat = b_b.vector(dict(zip(mats_b, coef)) if isinstance(coef, dict) else coef)

allrows = np.arange(D["n"])
print("\nNegBench COCO-MCQ, ALL 5914 questions, nothing tuned on it:")
for name, C, I in (("plain CLIP", D["CAP"], None), ("synthetic-fitted geometry", CAPmat, IMG_b)):
    a = common.accuracy(D, C, allrows, I)
    t = common.tie_rate(D, C, allrows, I)
    pt = common.per_template(D, C, allrows, I)
    print(f"  {name:<28} {100 * a:6.2f}%   ties {100 * t:.2f}%   "
          f"(pos {100 * pt['positive']:.1f} / neg {100 * pt['negative']:.1f} / "
          f"hyb {100 * pt['hybrid']:.1f})")

common.report("ex5_synthetic_fit_wholebench", dict(D, tr_rows=allrows, te_rows=allrows),
              CAPmat, IMGmat=IMG_b,
              extra={"fitted_on": "1500 synthetic questions, surrogate image = text of the scene",
                     "tuned_on_benchmark": "nothing", "c": float(c_star),
                     "synthetic_fit_acc": float(fit_acc), "n_scalars": len(mats_b)})
