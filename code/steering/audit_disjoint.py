"""The honest number: the full fitting procedure, on an IMAGE-DISJOINT split.

5914 questions share only 3926 images, so the default row split lets 20% of images appear on both
sides — the scalars can then be fitted against images they will be scored on. This refits everything
(the same c-sweep and multistart as the winning run) with no image ever crossing the split.
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

imgs = np.unique(D["img_idx"])
rng = np.random.default_rng(0)
half = set(rng.permutation(imgs)[:len(imgs) // 2].tolist())
tr = np.array([r for r in range(D["n"]) if D["img_idx"][r] in half])
te = np.array([r for r in range(D["n"]) if D["img_idx"][r] not in half])
D2 = dict(D, tr_rows=tr, te_rows=te)
print(f"image-disjoint: {len(tr)} fit rows / {len(te)} test rows, "
      f"{len(set(D['img_idx'][tr]) & set(D['img_idx'][te]))} shared images")

ex2.BEST.clear()
b_cls = ex2.Basis(D2, PERC)
ex2._fit_best(D2, b_cls, BASE, ctype)          # same c-sweep and multistart as the winning run
IMG = None
In = common.l2n(D["IMG"])
IMG = In - ex2.BEST["c"] * In[np.unique(D["img_idx"][tr])].mean(0)
b_cls.set_images(IMG)
S = b_cls.scores(ex2.BEST["coef"])
print(f"\nc* = {ex2.BEST['c']}")
print(f"fit  {100 * ex2.acc(D2, S, tr):.2f}%     TEST(image-disjoint) {100 * ex2.acc(D2, S, te):.2f}%")
common.report("ex2_recon_perclass_imagedisjoint", D2, b_cls.vector(ex2.BEST["coef"]), IMGmat=IMG,
              extra={"c": ex2.BEST["c"], "split": "image-disjoint", "n_scalars": len(PERC)})
