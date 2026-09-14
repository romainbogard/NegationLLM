"""ex1_global_geometry.py — global geometric corrections of the CLIP embedding space.

No per-caption parsing and no negation-specific logic in sections 0-8: every technique there is a
*global* linear (or per-question linear) re-shaping of the shared space, applied identically to
every caption.  Section 9 composes the same geometry on top of a stronger caption base (the
span-projection operator) at the coordinator's request; section 10 fits a supervised global metric
purely as an upper bound on what a single global linear map can do.

Techniques (one knob each, all composable through `apply_geometry`)
------------------------------------------------------------------
  ci   image-side centering        I <- I - ci * mean(I)      (raw or l2-normalised images)
  ct   text-side centering         T <- T - ct * mean(T)
       CSLS / hubness correction, score = cos(i,t) - r * mean_{i' in bg} cos(i',t), is *exactly*
       `ci` on l2-normalised images: mean_{i'}<i'^,t^> = <mean(i'^), t^>, and rescaling an image
       vector rescales all four caption scores of a question by the same positive factor, so the
       argmax cannot see the image norm.  CSLS === normalised image centering, r === ci.
  q    per-question centering      t_j <- t_j - q * mean_{j'} t_j'   (removes the content the four
       options share).  Commutes with any right-multiplication, so it is applied last.
  k*   all-but-the-top             zero the top-k principal components (basis fit on train).
       A projection P is symmetric and idempotent, so <i,Pt> = <i,PPt> = <Pi,Pt>: removing PCs on
       the text side only gives the same *ranking* as removing them on both sides.  We therefore
       always apply the map to both sides.
  a*   partial PCA whitening       component d scaled by var_d^(-a/2): a=0 nothing, a=1 full
       whitening, a<0 emphasises the dominant directions.  Applied to both sides, because a
       one-sided non-projective map compares vectors living in two different metrics.
  ad   diagonal whitening          per-coordinate z-scoring (train fit), both sides.
  `t` = caption basis, `i` = image basis, `j` = joint (captions + images) basis.

Protocol
--------
Every hyper-parameter is chosen by accuracy on D["tr_rows"]; every reported number is on
D["te_rows"].  Means / PCA bases are fit on the training half only, except the explicitly labelled
`*_allfit` control (label-free but transductive; reported, never used to select).  The supervised
metrics of section 10 split the *train* half again into fit/val.  Test is never used to select.

Degenerate corner
-----------------
Whitening, PC removal and per-question centering can all collapse a question's four options onto
each other; the argmax then breaks ties by index rather than by the image.  Any candidate whose
*train* tie rate exceeds TIE_MAX is rejected during the search, and every reported line carries
common.report's tie rate.

Run:  .venv/bin/python ex1_global_geometry.py
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np

import common
from common import l2n

t_start = time.time()
TIE_MAX = 0.005                 # a candidate with more ties than this on train is not a result

# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
D = common.load_mcq()
n = D["n"]
CAP = np.ascontiguousarray(D["CAP"], dtype=np.float64)        # (4n, d) raw text embeddings
IMG = np.ascontiguousarray(D["IMG"], dtype=np.float64)        # (n_img, d) raw image embeddings
d = CAP.shape[1]
gold, img_idx = D["gold"], D["img_idx"]
tr, te = D["tr_rows"], D["te_rows"]

cap_rows = lambda rows: (np.asarray(rows)[:, None] * 4 + np.arange(4)).ravel()
TR_CAP, TR_IMG = cap_rows(tr), np.unique(img_idx[tr])
ALL_CAP, ALL_IMG = np.arange(4 * n), np.arange(len(IMG))
FIT_ROWS = {"train": (TR_CAP, TR_IMG), "all": (ALL_CAP, ALL_IMG)}

print(f"[ex1] n={n} rows, {len(IMG)} images, d={d}; train {len(tr)} / test {len(te)}; "
      f"train images {len(TR_IMG)}")


def score_mat(T, I):
    """(n, 4) cosine scores — identical to common.scores, minus the repeated allocation."""
    return np.einsum("nd,nkd->nk", l2n(I)[img_idx], l2n(T).reshape(n, 4, -1))


def acc(S, rows):
    return float((S[rows].argmax(1) == gold[rows]).mean())


def ties(S, rows):
    P = np.sort(S[rows], axis=1)
    return float(((P[:, -1] - P[:, -2]) < 1e-12).mean())


def _pca(X):
    """Principal directions of X, descending. eigh on the 512x512 covariance: fast enough to
    refit the basis for every new caption base."""
    Xc = X - X.mean(0)
    w, V = np.linalg.eigh(Xc.T @ Xc / max(len(X) - 1, 1))
    o = np.argsort(w)[::-1]
    return np.ascontiguousarray(V[:, o].T), np.clip(w[o], 1e-12, None)


# ---------------------------------------------------------------------------------------------
# a Space = one (caption, image) base + the statistics fitted on it
# ---------------------------------------------------------------------------------------------
DEFAULTS = dict(ci=0.0, ct=0.0, q=0.0, kt=0, at=0.0, ki=0, ai=0.0, kj=0, aj=0.0, ad=0.0,
                raw=False, fit="train")


class Space:
    def __init__(self, CAPb, IMGb, name="space"):
        self.name, self.CAP, self.IMG = name, CAPb, IMGb
        self.TN, self.IN = l2n(CAPb), l2n(IMGb)
        self._st, self._maps = {}, {}

    def stats(self, fit="train"):
        if fit not in self._st:
            crows, irows = FIT_ROWS[fit]
            Vt_t, var_t = _pca(self.TN[crows])
            Vt_i, var_i = _pca(self.IN[irows])
            Vt_j, var_j = _pca(np.vstack([self.TN[crows], self.IN[irows]]))
            self._st[fit] = dict(
                mu_t=self.TN[crows].mean(0), mu_i=self.IN[irows].mean(0),
                mu_t_raw=self.CAP[crows].mean(0), mu_i_raw=self.IMG[irows].mean(0),
                Vt_t=Vt_t, var_t=var_t, Vt_i=Vt_i, var_i=var_i, Vt_j=Vt_j, var_j=var_j,
                sd=np.vstack([self.TN[crows], self.IN[irows]]).std(0) + 1e-12)
        return self._st[fit]

    def spec(self, fit, which, k, a):
        """V^T diag(var^(-a/2), top-k zeroed) V — a symmetric map in the original coordinates."""
        key = (fit, which, k, round(float(a), 6))
        if key not in self._maps:
            st = self.stats(fit)
            Vt, var = st[f"Vt_{which}"], st[f"var_{which}"]
            sc = (var / var.max()) ** (-a / 2.0) if a else np.ones_like(var)
            if k:
                sc = sc.copy()
                sc[:k] = 0.0
            self._maps[key] = (Vt.T * sc) @ Vt
        return self._maps[key]

    def geom(self, **p):
        """Apply the requested global corrections; returns (CAPmat, IMGmat)."""
        p = {**DEFAULTS, **p}
        st = self.stats(p["fit"])
        if p["raw"]:                                          # centre the *unnormalised* vectors
            T = self.CAP - p["ct"] * st["mu_t_raw"] if p["ct"] else self.CAP
            I = self.IMG - p["ci"] * st["mu_i_raw"] if p["ci"] else self.IMG
        else:
            T = self.TN - p["ct"] * st["mu_t"] if p["ct"] else self.TN
            I = self.IN - p["ci"] * st["mu_i"] if p["ci"] else self.IN

        M = None                                              # fold the spectral maps together
        for w, k, a in (("t", p["kt"], p["at"]), ("i", p["ki"], p["ai"]), ("j", p["kj"], p["aj"])):
            if k or a:
                Mi = self.spec(p["fit"], w, k, a)
                M = Mi if M is None else M @ Mi
        if p["ad"]:
            sc = st["sd"] ** (-p["ad"])
            M = np.diag(sc) if M is None else M * sc
        if M is not None:
            T, I = T @ M, I @ M

        if p["q"]:                                            # per-question centering, applied last
            T3 = T.reshape(n, 4, -1)
            T = (T3 - p["q"] * T3.mean(1, keepdims=True)).reshape(4 * n, -1)
        return np.ascontiguousarray(T), np.ascontiguousarray(I)

    def train_acc(self, **p):
        """Train accuracy, with degenerate (tie-collapsing) candidates rejected outright."""
        S = score_mat(*self.geom(**p))
        return -1.0 if ties(S, tr) > TIE_MAX else acc(S, tr)


BASE = Space(CAP, IMG, "raw-clip")

_SPACES: dict = {}


def apply_geometry(CAPmat, IMGmat, params=None, key=None):
    """Public entry point so other experiments can stack this geometry on their own caption base.

    params: any subset of DEFAULTS (ci, ct, q, kt, at, ki, ai, kj, aj, ad, raw, fit).
    Statistics are re-fit on the training half of whatever base is handed in, so a caller must
    re-tune the scalars for their base rather than reusing ours.  Returns (CAP2, IMG2), both
    consumable by common.report / common.scores.
    """
    k = key or (id(CAPmat), id(IMGmat))
    if k not in _SPACES:
        _SPACES[k] = Space(np.ascontiguousarray(CAPmat, dtype=np.float64),
                           np.ascontiguousarray(IMGmat, dtype=np.float64), str(k))
    return _SPACES[k].geom(**(params or {}))


# ---------------------------------------------------------------------------------------------
# reporting / scoreboard
# ---------------------------------------------------------------------------------------------
BOARD = []


def finalize(name, params=None, extra=None, mats=None, space=None):
    space = space or BASE
    T, I = mats if mats is not None else space.geom(**(params or {}))
    identity = bool(np.allclose(T, space.CAP) and np.allclose(I, space.IMG))
    tr_s, te_s = common.per_template(D, T, tr, I), common.per_template(D, T, te, I)
    clean = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
             for k, v in (params or {}).items()}
    common.report(name, D, T, IMGmat=I,
                  extra={"params": clean, "base": space.name, "identity_transform": identity,
                         "tie_rate_train": ties(score_mat(T, I), tr), **(extra or {})})
    if identity:
        print("      (note: the train sweep selected this technique's no-op value — "
              "the transform is the identity here, see the printed grid)")
    BOARD.append(dict(name=name, params=clean, base=space.name, train=tr_s["all"],
                      test=te_s["all"], pos=te_s["positive"], neg=te_s["negative"],
                      hyb=te_s["hybrid"], tie=ties(score_mat(T, I), te), ident=identity))
    return te_s


def sweep(knob, values, base=None, space=None, verbose=True):
    space = space or BASE
    b = dict(base or {})
    curve = [(v, space.train_acc(**{**b, knob: v})) for v in values]
    best = max(curve, key=lambda kv: kv[1])
    if verbose:
        step = max(1, len(curve) // 16)
        print("  [%s] %s" % (knob, " ".join(f"{v}:{100 * a:.2f}" for v, a in curve[::step])))
        print(f"  -> best {knob}={best[0]} train {100 * best[1]:.2f}%")
    return best[0], best[1], curve


CG = [round(x, 2) for x in np.arange(0.0, 3.01, 0.1)]          # centering coefficients
QG = [round(x, 2) for x in np.arange(0.0, 2.01, 0.1)]          # per-question coefficient
KG = list(range(0, 21))                                        # principal components removed
AG = [round(x, 2) for x in np.arange(-1.0, 1.01, 0.1)]         # whitening exponent
KNOBS = [("ci", CG), ("q", QG), ("kt", KG), ("at", AG), ("ct", CG),
         ("kj", KG), ("aj", AG), ("ki", KG), ("ai", AG), ("ad", AG)]


def coord_ascent(start, space=None, rounds=4, knobs=KNOBS, tag=""):
    space = space or BASE
    cur, best = dict(start), space.train_acc(**start)
    for r in range(rounds):
        moved = False
        for knob, grid in knobs:
            v, a, _ = sweep(knob, grid, cur, space, verbose=False)
            if a > best + 1e-9:
                cur[knob], best, moved = v, a, True
        print(f"  {tag}round {r + 1}: train {100 * best:.2f}%  "
              f"{ {k: (round(v, 2) if isinstance(v, float) else v) for k, v in cur.items() if v} }")
        if not moved:
            break
    return {k: v for k, v in cur.items() if v}, best


# =============================================================================================
# 0. baseline
# =============================================================================================
print("\n=== 0. baseline (plain CLIP) ===")
finalize("ex1_baseline", {})

# =============================================================================================
# 1-6. single techniques; every hyper-parameter picked on the train half only
# =============================================================================================
print("\n=== 1a. image centering, raw image mean ===")
v, _, c = sweep("ci", CG, {"raw": True}); finalize("ex1_img_center_raw", {"ci": v, "raw": True}, {"grid": c})

print("\n=== 1b. image centering on normalised images  ==  CSLS with r = ci ===")
v_csls, _, c = sweep("ci", CG)
finalize("ex1_csls_img_center", {"ci": v_csls}, {"grid": c,
         "note": "identical to CSLS with the training-half images as background"})

print("\n=== 1c. same, statistics fit on ALL images (transductive control) ===")
v, _, c = sweep("ci", CG, {"fit": "all"}); finalize("ex1_csls_allfit", {"ci": v, "fit": "all"}, {"grid": c})

print("\n=== 2a. text centering, raw caption mean ===")
v, _, c = sweep("ct", CG, {"raw": True}); finalize("ex1_txt_center_raw", {"ct": v, "raw": True}, {"grid": c})

print("\n=== 2b. text centering on normalised captions ===")
v_ct, _, c = sweep("ct", CG); finalize("ex1_txt_center", {"ct": v_ct}, {"grid": c})

print("\n=== 3. both sides centered (2-D train grid) ===")
G2 = CG[::2]
b = max(((ci, ct, BASE.train_acc(ci=ci, ct=ct)) for ci in G2 for ct in G2), key=lambda t: t[2])
print(f"  -> best ci={b[0]} ct={b[1]} train {100 * b[2]:.2f}%")
finalize("ex1_both_center", {"ci": b[0], "ct": b[1]})

print("\n=== 5. per-question centering ===")
v_q, _, c = sweep("q", QG); finalize("ex1_perq_center", {"q": v_q}, {"grid": c})
print("      (sanity: q=1.0 changes every caption vector — "
      f"train {100 * BASE.train_acc(q=1.0):.2f}%, "
      f"train tie rate {100 * ties(score_mat(*BASE.geom(q=1.0)), tr):.2f}%)")

print("\n=== 6a. all-but-the-top, caption basis ===")
v_kt, _, c = sweep("kt", KG); finalize("ex1_abtt_text", {"kt": v_kt}, {"grid": c})

print("\n=== 6b. all-but-the-top, image basis ===")
v_ki, _, c = sweep("ki", KG); finalize("ex1_abtt_image", {"ki": v_ki}, {"grid": c})

print("\n=== 6c. all-but-the-top, joint basis ===")
v_kj, _, c = sweep("kj", KG); finalize("ex1_abtt_joint", {"kj": v_kj}, {"grid": c})

print("\n=== 6d. partial PCA whitening, caption basis ===")
v_at, _, c = sweep("at", AG); finalize("ex1_whiten_text", {"at": v_at}, {"grid": c})

print("\n=== 6e. partial PCA whitening, joint basis ===")
v_aj, _, c = sweep("aj", AG); finalize("ex1_whiten_joint", {"aj": v_aj}, {"grid": c})

print("\n=== 6f. diagonal (per-coordinate) whitening ===")
v_ad, _, c = sweep("ad", AG); finalize("ex1_whiten_diag", {"ad": v_ad}, {"grid": c})

# =============================================================================================
# 7. combinations on raw CLIP
# =============================================================================================
print("\n=== 7a. coordinate ascent from the best single techniques ===")
combo_a, acc_a = coord_ascent({"ci": v_csls, "q": v_q, "ct": v_ct}, tag="a ")

print("\n=== 7b. product grid over (ci, q, kt), then re-ascent ===")
gi, gq = [round(x, 2) for x in np.arange(0, 2.61, 0.2)], [round(x, 2) for x in np.arange(0, 2.01, 0.2)]
gk = [0, 1, 2, 3, 5, 8, 12, 20]
bg = max(((ci, q, kt, BASE.train_acc(ci=ci, q=q, kt=kt))
          for ci, q, kt in itertools.product(gi, gq, gk)), key=lambda t: t[3])
print(f"  grid best ci={bg[0]} q={bg[1]} kt={bg[2]} train {100 * bg[3]:.2f}%")
combo_b, acc_b = coord_ascent({"ci": bg[0], "q": bg[1], "kt": bg[2]}, tag="b ")

finalize("ex1_combo_fromsingles", combo_a, {"train_selected": acc_a})
finalize("ex1_combo_grid", combo_b, {"train_selected": acc_b})
best_combo, best_train = max([(combo_a, acc_a), (combo_b, acc_b)], key=lambda t: t[1])
print(f"\n=== 7c. best unsupervised combination on train: {best_combo} ({100 * best_train:.2f}%) ===")
finalize("ex1_combo_best", best_combo, {"train_selected": best_train})
finalize("ex1_combo_best_allfit", {**best_combo, "fit": "all"})

# =============================================================================================
# 8. diagnostics: what centering actually does, and why the negative subset resists it
# =============================================================================================
print("\n=== 8. diagnostics ===")
st = BASE.stats("train")
print(f"  modality gap ||mean(img^) - mean(cap^)|| = "
      f"{np.linalg.norm(st['mu_i'] - st['mu_t']):.3f}   "
      f"||mean(img^)|| = {np.linalg.norm(st['mu_i']):.3f}  (image cone) "
      f"||mean(cap^)|| = {np.linalg.norm(st['mu_t']):.3f}  (caption cone)")


def margin(S, sel):
    """mean(gold score - best distractor score); negative means the gold option loses."""
    Z, ar = S[sel].copy(), np.arange(len(sel))
    g = Z[ar, gold[sel]].copy()
    Z[ar, gold[sel]] = -np.inf
    return float(np.mean(g - Z.max(1)))


# image centering adds the caption-dependent term  -ci * <mu_img, t^>  to every score: a pure
# "how image-typical is this caption" penalty.  It can only re-rank a question if that term
# *varies across the four options of that question*.
TNn = l2n(CAP).reshape(n, 4, d)
hub = TNn @ st["mu_i"]                                        # (n,4) hub term per option
base_cos = score_mat(CAP, IMG)
S0, S1 = base_cos, score_mat(*BASE.geom(**best_combo))
print(f"  {'subset':<9}{'base':>7}{'combo':>8}{'margin0':>9}{'margin1':>9}"
      f"{'std(hub)':>10}{'std(cos)':>10}{'ratio':>8}")
for t in ("positive", "negative", "hybrid"):
    sel = te[D["tmpl"][te] == t]
    sh, sc_ = hub[sel].std(1).mean(), base_cos[sel].std(1).mean()
    print(f"  {t:<9}{100 * acc(S0, sel):6.1f}%{100 * acc(S1, sel):7.1f}%"
          f"{margin(S0, sel):+9.4f}{margin(S1, sel):+9.4f}{sh:10.4f}{sc_:10.4f}{sh / sc_:8.2f}")

# =============================================================================================
# 9. composition: the same geometry on top of the span-projection caption base
#    (this is the ONE place where negation parsing enters, at the coordinator's request)
#    e* = e^c - lam * (proj_{e^neg}(e^c) - a),  a = common.anchor()
# =============================================================================================
print("\n=== 9. span-projection base + geometry ===")
NEGSPAN = [common.extract_negated(t) for t in D["flat"]]
ENEG = l2n(common.embed_texts([s or "" for s in NEGSPAN]))
HAS = np.array([s is not None for s in NEGSPAN], dtype=float)[:, None]
ANCH = common.anchor()


def span_base(lam, masked=True):
    proj = np.einsum("ij,ij->i", CAP, ENEG)[:, None] * ENEG
    corr = lam * (proj - ANCH)
    return CAP - (corr * HAS if masked else corr)


for masked in (True, False):
    curve = [(lam, acc(score_mat(span_base(lam, masked), IMG), tr))
             for lam in np.arange(0.0, 2.01, 0.1)]
    bl = max(curve, key=lambda kv: kv[1])
    print(f"  masked={masked}: " + " ".join(f"{l:.1f}:{100 * a:.2f}" for l, a in curve[::2]))
    print(f"  -> best lam={bl[0]:.1f} train {100 * bl[1]:.2f}%")
    if masked:
        lam_m, best_m = bl
    else:
        lam_u, best_u = bl
MASKED, LAM = (True, lam_m) if best_m >= best_u else (False, lam_u)
finalize("ex1_span_base", {"lam": float(LAM), "masked": MASKED},
         mats=(span_base(LAM, MASKED), IMG), space=Space(span_base(LAM, MASKED), IMG, "span"))

# re-tune the geometry scalars on top of the span base (never reuse the raw-base scalars), and
# re-tune lam jointly by rebuilding the space for a few values of lam
best_span = (None, -1.0, None)
for lam in sorted({round(float(LAM), 1), round(float(LAM) - 0.2, 1), round(float(LAM) + 0.2, 1)}):
    sp = Space(span_base(lam, MASKED), IMG, f"span(lam={lam})")
    cmb, a = coord_ascent({"ci": v_csls}, space=sp, rounds=3, tag=f"lam={lam} ")
    if a > best_span[1]:
        best_span = (cmb, a, sp, lam)
print(f"  best span+geometry on train: lam={best_span[3]} {best_span[0]} "
      f"({100 * best_span[1]:.2f}%)")
finalize("ex1_span_plus_geometry", {**best_span[0], "lam": float(best_span[3]), "masked": MASKED},
         {"train_selected": best_span[1]}, space=best_span[2])

# =============================================================================================
# 10. upper bound: a supervised global metric (still one global linear map, no parsing)
#     fit on half the train rows, early-stopped on the other half, read once on test
# =============================================================================================
print("\n=== 10. supervised global metric (train-fit / train-val; test untouched) ===")
try:
    import torch

    rng = np.random.default_rng(1)
    pm = rng.permutation(len(tr))
    fit_rows, val_rows = tr[pm[: len(tr) // 2]], tr[pm[len(tr) // 2:]]
    tg = torch.tensor(gold.astype(np.int64))

    def metric_on(space, params, tag):
        T0, I0 = space.geom(**params)
        Vt = _pca(np.vstack([l2n(T0)[TR_CAP], l2n(I0)[TR_IMG]]))[0]     # train-only basis
        tT = torch.tensor((l2n(T0) @ Vt.T).reshape(n, 4, d), dtype=torch.float32)
        tI = torch.tensor((l2n(I0) @ Vt.T)[img_idx], dtype=torch.float32)

        def run(mode, wd, steps, lr=0.05, tau=30.0):
            torch.manual_seed(0)
            P = (torch.zeros(d) if mode == "diag" else torch.zeros(d, d)).requires_grad_(True)
            opt, best = torch.optim.Adam([P], lr=lr), (-1.0, None)
            for s in range(steps + 1):
                A = torch.exp(P) if mode == "diag" else torch.eye(d) + P
                def sc(rows, A=A):
                    i = torch.nn.functional.normalize(
                        tI[rows] * A if mode == "diag" else tI[rows] @ A, dim=-1)
                    t = torch.nn.functional.normalize(
                        tT[rows] * A if mode == "diag" else tT[rows] @ A, dim=-1)
                    return tau * torch.einsum("nd,nkd->nk", i, t)
                if s % 20 == 0:
                    with torch.no_grad():
                        va = (sc(val_rows).argmax(1) == tg[val_rows]).float().mean().item()
                    if va > best[0]:
                        best = (va, P.detach().clone())
                opt.zero_grad()
                (torch.nn.functional.cross_entropy(sc(fit_rows), tg[fit_rows])
                 + wd * (P ** 2).sum()).backward()
                opt.step()
            return best

        for mode, wds, steps in (("diag", [1e-4, 1e-3, 1e-2], 600),
                                 ("full", [1e-3, 1e-2, 1e-1], 150)):
            cands = [(run(mode, wd, steps), wd) for wd in wds]
            (va, P), wd = max(cands, key=lambda c: c[0][0])
            print(f"  {tag}/{mode}: wd={wd} train-val {100 * va:.2f}%")
            with torch.no_grad():
                A = (torch.exp(P) if mode == "diag" else torch.eye(d) + P).numpy().astype(np.float64)
            Tr_, Ir_ = l2n(T0) @ Vt.T, l2n(I0) @ Vt.T
            mats = (Tr_ * A, Ir_ * A) if mode == "diag" else (Tr_ @ A, Ir_ @ A)
            finalize(f"ex1_metric_{mode}_{tag}", {"init": tag, "wd": wd, "supervised": True},
                     {"train_val": va}, mats=mats, space=space)

    metric_on(BASE, best_combo, "geom")
    metric_on(best_span[2], best_span[0], "span")
except ImportError:
    print("  torch unavailable — skipped")

# =============================================================================================
# 11. scoreboard
# =============================================================================================
print("\n" + "=" * 122)
print(f"{'variant':<30}{'base':<12}{'train':>8}{'TEST':>9}{'pos':>7}{'neg':>7}{'hyb':>7}"
      f"{'ties':>7}   params")
print("-" * 122)
for r in sorted(BOARD, key=lambda r: -r["test"]):
    pr = ", ".join(f"{k}={v}" for k, v in r["params"].items() if v not in (0, 0.0, False, "train"))
    print(f"{r['name']:<30}{r['base'][:11]:<12}{100 * r['train']:>7.2f}%{100 * r['test']:>8.2f}%"
          f"{100 * r['pos']:>7.1f}{100 * r['neg']:>7.1f}{100 * r['hyb']:>7.1f}"
          f"{100 * r['tie']:>6.2f}%   {pr}{'   [IDENTITY]' if r['ident'] else ''}")
print("=" * 122)
pure = [r for r in BOARD if "metric" not in r["name"] and "span" not in r["name"]]
sel = max(pure, key=lambda r: r["train"])
print(f"pure-geometry winner (train-selected): {sel['name']}  TEST {100 * sel['test']:.2f}% "
      f"(pos {100 * sel['pos']:.1f} / neg {100 * sel['neg']:.1f} / hyb {100 * sel['hyb']:.1f})")
for grp, lbl in ((["span"], "geometry on the span base"), (["metric"], "supervised metric bound")):
    rs = [r for r in BOARD if any(g in r["name"] for g in grp)]
    if rs:
        s = max(rs, key=lambda r: r["train"])
        print(f"{lbl:<38}: {s['name']}  TEST {100 * s['test']:.2f}% "
              f"(pos {100 * s['pos']:.1f} / neg {100 * s['neg']:.1f} / hyb {100 * s['hyb']:.1f})")

common.save_result("ex1_best_params", {
    "note": "feed these to ex1_global_geometry.apply_geometry(CAP, IMG, params); "
            "scalars are base-specific, re-tune on your own caption base",
    "pure_geometry": {"params": sel["params"], "train": sel["train"], "test": sel["test"]},
    "span_plus_geometry": next((r for r in BOARD if r["name"] == "ex1_span_plus_geometry"), None)})
print(f"[ex1] done in {time.time() - t_start:.0f}s")
