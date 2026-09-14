"""How much of the 79.5% is geometry, and how much is supervision on the benchmark?

The winning configuration fits 18 scalars by coordinate ascent on the MCQ answers of the training
half. That is far more supervision than the published method (one scalar, lambda), so three audits:

1. IMAGE-DISJOINT SPLIT. 5914 questions share only 3926 images, so the default row split puts the
   same image on both sides. Re-split by image so no image is ever seen at fit time.
2. LEARNING CURVE. Fit on N labelled questions, N from 25 up. If a hundred questions are enough,
   the scalars are a calibration; if it needs thousands, it is training.
3. PARAMETER COUNT. One scalar (the published correction), six global scalars, eighteen per-class.
"""
import numpy as np

import common
import ex2_compositional_spans as ex2

D = common.load_mcq()
flat = D["flat"]
parsed = [ex2.parse_spans(c) for c in flat]
A, N_, na, nn = ex2.span_embeddings(parsed, "ens")
ANCM = np.tile(common.anchor(), (len(flat), 1))
ctype = np.array([common.caption_type(c) for c in flat])
BASE = ex2._base_mats(D, A, N_, na, nn, ANCM)
PERC = ex2._per_class(BASE, ctype)
IN = common.l2n(D["IMG"])


def fit_eval(tr, te, per_class=True, c=0.85, rounds=10):
    """Fit the scalars on `tr` only (image centring included) and score on `te`."""
    tr, te = np.asarray(tr), np.asarray(te)
    D2 = dict(D, tr_rows=tr, te_rows=te)
    IMG = IN - c * IN[np.unique(D["img_idx"][tr])].mean(0)

    bg = ex2.Basis(D2, BASE, IMG=IMG)
    cg, ag = ex2.multistart(D2, bg, [np.eye(len(BASE))[0]], n_random=2, rounds=rounds)
    if not per_class:
        return ag, ex2.acc(D2, bg.scores(cg), te), len(BASE)

    bc = ex2.Basis(D2, PERC, IMG=IMG)
    cc, ac = ex2.multistart(D2, bc, [np.concatenate([cg] * 3)], n_random=1, rounds=rounds + 2)
    return ac, ex2.acc(D2, bc.scores(cc), te), len(PERC)


print("=" * 92)
print("1. IMAGE-DISJOINT SPLIT — no image seen at fit time appears at test time")
print("=" * 92)
imgs = np.unique(D["img_idx"])
rng = np.random.default_rng(0)
half = set(rng.permutation(imgs)[:len(imgs) // 2].tolist())
tr_i = np.array([r for r in range(D["n"]) if D["img_idx"][r] in half])
te_i = np.array([r for r in range(D["n"]) if D["img_idx"][r] not in half])
shared = len(set(D["img_idx"][D["tr_rows"]]) & set(D["img_idx"][D["te_rows"]]))
print(f"default split shares {shared} images between the halves "
      f"({100 * shared / len(imgs):.1f}% of all images)")
print(f"image-disjoint split: {len(tr_i)} fit rows / {len(te_i)} test rows, 0 shared images")
a_tr, a_te, k = fit_eval(tr_i, te_i)
print(f"  -> fit {100 * a_tr:.2f}%   TEST {100 * a_te:.2f}%   ({k} scalars)")
a_tr0, a_te0, _ = fit_eval(D["tr_rows"], D["te_rows"])
print(f"  default row split, same code: fit {100 * a_tr0:.2f}%   TEST {100 * a_te0:.2f}%")

print()
print("=" * 92)
print("2. LEARNING CURVE — how many labelled questions the scalars actually need")
print("=" * 92)
tr_all, te = D["tr_rows"], D["te_rows"]
for n in (25, 50, 100, 250, 500, 1000, len(tr_all)):
    sub = tr_all[:n]
    a_tr, a_te, k = fit_eval(sub, te)
    print(f"  N = {n:>4} labelled questions   fit {100 * a_tr:6.2f}%   TEST {100 * a_te:6.2f}%")

print()
print("=" * 92)
print("3. PARAMETER COUNT — what the extra scalars buy")
print("=" * 92)
a_tr, a_te, k = fit_eval(tr_all, te, per_class=False)
print(f"  {k:>2} global scalars      fit {100 * a_tr:6.2f}%   TEST {100 * a_te:6.2f}%")
a_tr, a_te, k = fit_eval(tr_all, te, per_class=True)
print(f"  {k:>2} per-class scalars   fit {100 * a_tr:6.2f}%   TEST {100 * a_te:6.2f}%")
print("   1 scalar (published Eq. 2, lambda only)              TEST  50.19%")
print("   0 scalars (plain CLIP)                               TEST  38.25%")
