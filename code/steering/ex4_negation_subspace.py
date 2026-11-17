"""ex4 — subspace geometry of negation on NegBench COCO-MCQ.

Steering by a *linear subspace operation* rather than by adding one fixed vector.

The four parts follow the brief:

  1. THE NEGATION SUBSPACE.  10 000 minimal pairs (data/negation_dataset.csv) give
     displacements  Δ_i = e(negated_i) − e(affirmative_i).  We report the PCA spectrum and
     how much of Δ is a pure translation (the mean) versus a genuine subspace.
  2. SUBSPACE OPERATORS on the captions the rule flags as non-positive: amplify the
     component in the subspace, project it out, reflect across it, translate by the mean
     displacement (the known baseline), and an input-dependent version whose strength is
     read off the caption's own position along the subspace.
  3. IN-DOMAIN vs OUT-OF-DOMAIN geometry.  The same operators with the subspace re-estimated
     from the NegBench *training-half captions only* (class-conditional means + PCA of the
     within-question positive→negative displacements).
  4. COMPOSITION with the published parsed correction (Aggarwal et al., Eq. 2), and the
     per-input, span-conditional subspace operator that comes out of it.

PROTOCOL.  Every scalar — r, alpha, the gate, the per-type weights — is chosen by maximising
accuracy on D["tr_rows"].  The reported number is D["te_rows"], via common.report().
common.py is never modified; the only thing that changes is the geometry of the captions.

TIE HYGIENE.  argmax breaks ties by slot, so an operator that collapses the four options
"wins" on the tie-break instead of on the image.  Every candidate in every sweep is rejected
outright if it produces any tied question on the training half; common.report() prints the
test tie rate as a second check.

Run:  .venv/bin/python code/steering/ex4_negation_subspace.py
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

import common as C

EPS = 1e-12
RANKS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]


# ============================================================================================
# 0.  Text side: caption type, negated span, and the affirmative remainder
# ============================================================================================
# common.extract_negated() returns the words *inside* the negation scope.  Its complement —
# everything outside the scope and outside the negator itself — is the caption's affirmative
# assertion ("This image features a knife, but no car is present." -> "this image features a
# knife").  It is built from common.py's own tokenizer and negator tables, so the two parses
# always agree; common.py itself is untouched.

_DANGLING = {"there", "is", "are", "was", "were", "and", "with", "that", "it", "this",
             "but", "however", "while"}


def _first_negator(caption: str):
    """(tokens, (index, length, kind)) for the first negation cue — mirrors extract_negated."""
    toks = C._TOK.findall(str(caption).lower())
    best = None
    for i in range(len(toks)):
        for kind, table in (("post", C.POST_NEGATORS), ("pre", C.PRE_NEGATORS)):
            for phrase in table:
                w = phrase.split()
                if toks[i:i + len(w)] == w and (best is None or len(w) > best[1]):
                    best = (i, len(w), kind)
            if best and best[0] == i:
                break
        if best:
            break
    return toks, best


def affirmative_remainder(caption: str):
    """C^aff: what the caption asserts *outside* the negation scope, or None."""
    toks, best = _first_negator(caption)
    if best is None:
        return None
    i, n, kind = best
    if kind == "pre":                      # scope runs forward, remainder is what precedes
        out = toks[:i]
        while out and (C._is_boundary(out[-1]) or out[-1] in _DANGLING):
            out.pop()
    else:                                  # scope runs backward, remainder is what follows
        out = toks[i + n:]
        while out and (C._is_boundary(out[0]) or out[0] in _DANGLING):
            out.pop(0)
        while out and C._is_boundary(out[-1]):
            out.pop()
    return " ".join(out) or None


# ============================================================================================
# 1.  Geometry estimation
# ============================================================================================
def spectrum(X, label):
    """PCA spectrum of a displacement cloud + the share of its energy that is pure translation."""
    mu = X.mean(0)
    energy = float((X ** 2).sum(1).mean())
    trans = float(mu @ mu) / max(energy, EPS)          # ||mean||^2 / E||Δ||^2
    sv_c = np.linalg.svd(X - mu, compute_uv=False) ** 2
    sv_u = np.linalg.svd(X, compute_uv=False) ** 2
    evr_c, evr_u = sv_c / sv_c.sum(), sv_u / sv_u.sum()
    part = float(np.exp(-(evr_c * np.log(evr_c + EPS)).sum()))   # participation / effective dim
    print(f"\n[{label}]  n={len(X)}  mean||Δ||={np.linalg.norm(X, axis=1).mean():.4f}  "
          f"||mean Δ||={np.linalg.norm(mu):.4f}")
    print(f"  pure-translation share of the energy : {100 * trans:.1f}%")
    print(f"  centred EVR   top20 : {np.round(evr_c[:20], 4)}")
    print(f"  centred cumEVR      : r=1 {evr_c[0]:.3f}  r=5 {evr_c[:5].sum():.3f}  "
          f"r=10 {evr_c[:10].sum():.3f}  r=20 {evr_c[:20].sum():.3f}")
    print(f"  uncentred cumEVR    : r=1 {evr_u[0]:.3f}  r=5 {evr_u[:5].sum():.3f}  "
          f"r=20 {evr_u[:20].sum():.3f}")
    print(f"  effective dimension (exp. entropy of the centred spectrum) : {part:.0f}")
    return {"n": len(X), "translation_share": trans, "effective_dim": part,
            "evr_centred_top20": evr_c[:20].tolist(), "evr_uncentred_top20": evr_u[:20].tolist(),
            "cum_centred": {str(r): float(evr_c[:r].sum()) for r in (1, 5, 10, 20)}}


def pca_basis(X, rank, centred=True):
    """Top-`rank` right singular vectors of the displacement cloud (rows are orthonormal)."""
    M = X - X.mean(0) if centred else X
    return np.linalg.svd(M, full_matrices=False)[2][:rank]


# ============================================================================================
# 2.  The subspace operators.  All of them are  e -> e + β(e) · P_r e  for some β.
# ============================================================================================
def op_scale(E, gate, V, beta):
    """amplify (β=+α) / project out (β=−α) / reflect across the subspace (β=−2α)."""
    out = E.copy()
    out[gate] = out[gate] + beta * ((out[gate] @ V.T) @ V)
    return out


def op_translate(E, gate, direction, alpha):
    """The known baseline: one fixed vector added to every flagged caption."""
    out = E.copy()
    out[gate] = out[gate] + alpha * direction
    return out


def op_adaptive(E, gate, V, target):
    """Input-dependent strength: rescale each caption's own subspace component to `target`.

    r = 1  -> the signed coordinate <e,v> is *set* to `target` (so the sign of the correction
              flips depending on which side of the hyperplane the caption already sits on);
    r > 1  -> the norm of P_r e is set to |target| along its own direction.
    Either way α is a function of the input rather than a constant.
    """
    out = E.copy()
    Cf = out[gate] @ V.T                                   # (m, r) coordinates
    if V.shape[0] == 1:
        out[gate] = out[gate] + (target - Cf[:, 0])[:, None] * V[0]
    else:
        nrm = np.linalg.norm(Cf, axis=1, keepdims=True)
        out[gate] = out[gate] + ((target - nrm) / np.clip(nrm, EPS, None)) * (Cf @ V)
    return out


# ============================================================================================
# 3.  Train-half model selection, with the tie guard
# ============================================================================================
class Selector:
    """Evaluates candidates on tr_rows only and refuses anything that wins on tie-breaks."""

    def __init__(self, D):
        self.D = D

    def train_score(self, CAPmat, IMGmat=None):
        S = C.scores(self.D, CAPmat, IMGmat)
        tr = self.D["tr_rows"]
        srt = np.sort(S[tr], axis=1)
        if ((srt[:, -1] - srt[:, -2]) < EPS).any():
            return -1.0, None                              # degenerate: collapsed captions
        acc = C.per_template(self.D, CAPmat, tr, IMGmat)
        return 100 * acc["all"], {k: 100 * v for k, v in acc.items()}

    def sweep(self, make, grid, IMGmat=None):
        """`make(params) -> CAPmat`; returns (best_params, train_all, train_breakdown)."""
        best = (None, -1.0, None)
        for p in grid:
            s, d = self.train_score(make(p), IMGmat)
            if s > best[1]:
                best = (p, s, d)
        return best


def main():
    D = C.load_mcq()
    tr, te = D["tr_rows"], D["te_rows"]
    sel = Selector(D)
    tr_cap = np.concatenate([4 * tr + j for j in range(4)])   # caption slots of the train rows
    summary = {}

    # ---- text-side features ----------------------------------------------------------------
    typ = np.array([C.caption_type(t) for t in D["flat"]])
    span = [C.extract_negated(t) for t in D["flat"]]
    rem = [affirmative_remainder(t) for t in D["flat"]]
    gate = np.array([s is not None for s in span])            # rule-flagged non-positive
    print(f"\n[gate] {100 * gate.mean():.1f}% of captions carry a negation scope; "
          f"types: {dict(zip(*np.unique(typ, return_counts=True)))}")

    U = C.l2n(D["CAP"])                                       # captions on the unit sphere
    SPN = C.l2n(C.embed_texts([s if s else "" for s in span]))
    REM = C.l2n(C.embed_texts([r if r else "" for r in rem]))

    # class-conditional means, train half only
    m_pos = U[tr_cap[typ[tr_cap] == "positive"]].mean(0)
    m_pos_h = m_pos / np.linalg.norm(m_pos)
    m_neg = U[tr_cap[typ[tr_cap] != "positive"]].mean(0)
    m_span = SPN[np.intersect1d(tr_cap, np.where(gate)[0])].mean(0)
    REM[np.array([r is None for r in rem])] = m_pos_h         # fallback for bare negations

    # ========================================================================================
    # PART 1 — the spectrum of the negation displacement
    # ========================================================================================
    print("\n" + "=" * 92)
    print("PART 1 — how many dimensions does negation actually occupy?")
    print("=" * 92)

    pairs = pd.read_csv(C.ROOT / "data" / "negation_dataset.csv")
    A = C.l2n(C.embed_texts(pairs["affirmative"].tolist()))
    N = C.l2n(C.embed_texts(pairs["negated"].tolist()))
    DELTA_OOD = N - A                                         # out-of-domain displacements
    spec_ood = spectrum(DELTA_OOD, "out-of-domain: 10k generic minimal pairs")

    # in-domain: within a question, every (positive-flagged -> non-positive-flagged) pair is a
    # displacement between two descriptions of the *same image*, i.e. a minimal pair by
    # construction.  Train rows only.
    pair_idx = [(4 * r + b, 4 * r + a)
                for r in tr
                for a in range(4) if typ[4 * r + a] == "positive"
                for b in range(4) if typ[4 * r + b] != "positive"]
    pair_idx = np.array(pair_idx)
    DELTA_IN = U[pair_idx[:, 0]] - U[pair_idx[:, 1]]
    spec_in = spectrum(DELTA_IN, "in-domain: within-question caption pairs (train half)")

    mu_ood = DELTA_OOD.mean(0)
    mu_in = DELTA_IN.mean(0)
    d_cls = m_neg - m_pos                                     # class-conditional mean difference
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"\n  cos(mean OOD displacement, mean in-domain displacement) = {cos(mu_ood, mu_in):.3f}")
    print(f"  cos(mean in-domain displacement, class-conditional mean difference) = "
          f"{cos(mu_in, d_cls):.3f}")
    print(f"  principal-angle cosines, top-5 OOD subspace vs top-5 in-domain subspace: "
          f"{np.round(np.linalg.svd(pca_basis(DELTA_OOD, 5) @ pca_basis(DELTA_IN, 5).T, compute_uv=False), 3)}")
    summary["spectra"] = {"out_of_domain": spec_ood, "in_domain": spec_in,
                          "cos_mean_ood_vs_indomain": cos(mu_ood, mu_in)}

    # ========================================================================================
    # baseline
    # ========================================================================================
    print("\n" + "=" * 92)
    print("PART 2 — subspace operators (r and alpha selected on tr_rows)")
    print("=" * 92)
    C.report("ex4_plain_clip", D, D["CAP"], extra={"note": "untouched CLIP"})

    # ---- the operator battery, run once per geometry ---------------------------------------
    ALPHA = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    def battery(tag, DELTA, mu):
        """amplify / project-out / reflect / translate / adaptive on one displacement cloud."""
        bases = {r: pca_basis(DELTA, r) for r in RANKS}
        mu_h = mu / np.linalg.norm(mu)
        out = {}

        for opname, betas in [("amplify", [+a for a in ALPHA]),
                              ("projectout", [-a for a in ALPHA]),
                              ("reflect", [-2 * a for a in ALPHA])]:
            grid = [(r, b) for r in RANKS for b in betas]
            (r, b), s, d = sel.sweep(lambda p: op_scale(U, gate, bases[p[0]], p[1]), grid)
            f = C.report(f"ex4_{opname}_{tag}", D, op_scale(U, gate, bases[r], b),
                         extra={"geometry": tag, "operator": opname, "rank": r, "beta": b,
                                "train_breakdown": d, "selected_on": "tr_rows"})
            out[f"{opname}_{tag}"] = (r, b, s)

        # the published-style translation baseline: one fixed vector, magnitude swept both ways
        tgrid = [t for t in (-1.0, -0.6, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05,
                             0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0)]
        t, s, d = sel.sweep(lambda a: op_translate(U, gate, mu_h, a), tgrid)
        C.report(f"ex4_translate_{tag}", D, op_translate(U, gate, mu_h, t),
                 extra={"geometry": tag, "operator": "mean translation", "alpha": t,
                        "train_breakdown": d, "selected_on": "tr_rows"})
        out[f"translate_{tag}"] = (None, t, s)

        # input-dependent strength
        tgrid = [(r, v) for r in RANKS
                 for v in (-0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5)]
        (r, v), s, d = sel.sweep(lambda p: op_adaptive(U, gate, bases[p[0]], p[1]), tgrid)
        C.report(f"ex4_adaptive_{tag}", D, op_adaptive(U, gate, bases[r], v),
                 extra={"geometry": tag, "operator": "per-input alpha", "rank": r, "target": v,
                        "train_breakdown": d, "selected_on": "tr_rows"})
        out[f"adaptive_{tag}"] = (r, v, s)
        return out

    picks = {}
    picks.update(battery("ood", DELTA_OOD, mu_ood))

    print("\n" + "=" * 92)
    print("PART 3 — the same operators with the subspace re-estimated IN DOMAIN")
    print("=" * 92)
    picks.update(battery("indomain", DELTA_IN, mu_in))

    # rank-1 in-domain special case: steer along the class-conditional mean difference
    d_h = d_cls / np.linalg.norm(d_cls)
    t, s, d = sel.sweep(lambda a: op_translate(U, gate, d_h, a),
                        [-0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3])
    C.report("ex4_classmean_indomain", D, op_translate(U, gate, d_h, t),
             extra={"geometry": "indomain", "operator": "class-conditional mean difference",
                    "alpha": t, "train_breakdown": d, "selected_on": "tr_rows"})
    summary["subspace_picks"] = {k: {"rank": v[0], "alpha": v[1], "train": v[2]}
                                 for k, v in picks.items()}

    # ========================================================================================
    # PART 4 — composition with the published correction, and the span-conditional operator
    # ========================================================================================
    print("\n" + "=" * 92)
    print("PART 4 — composition with Aggarwal et al. Eq. 2, and per-input span geometry")
    print("=" * 92)

    # Eq. 2 as published: subtract the negated span, anchored on "neutral/balanced/unbiased/fair".
    a_h = C.anchor(); a_h = a_h / np.linalg.norm(a_h)
    lam_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]

    def eq2(lam, centre):
        out = U.copy()
        out[gate] = out[gate] - lam * (SPN[gate] - centre)
        return out

    lam_pub, s, d = sel.sweep(lambda l: eq2(l, a_h), lam_grid)
    C.report("ex4_eq2_published", D, eq2(lam_pub, a_h),
             extra={"operator": "Eq.2, published anchor", "lambda": lam_pub,
                    "train_breakdown": d, "selected_on": "tr_rows"})

    # Same operator, in-domain centre: the mean *span* embedding over the training half.
    # The published anchor is a generic word bundle; the in-domain centre is where the span
    # cloud actually sits, and swapping it is worth several points on its own.
    lam_id, s_id, d_id = sel.sweep(lambda l: eq2(l, m_span), lam_grid)
    C.report("ex4_eq2_indomain_centre", D, eq2(lam_id, m_span),
             extra={"operator": "Eq.2 with the in-domain span mean as the centre",
                    "lambda": lam_id, "train_breakdown": d_id, "selected_on": "tr_rows"})

    # ---- the per-input, span-conditional subspace operator ---------------------------------
    # Parts 2-3 show that a caption-global operator cannot work: the correction has to depend on
    # *which* concept is negated.  So take the steering direction from the caption itself,
    #     v_i = l2n(e(span_i)) - m_span,
    # a one-dimensional subspace that is different for every input, and rebuild the caption as
    #     e*_i = wa·e(remainder_i) + wm·m_pos + wc·e_i - lam·v_i
    # with (wa, wm, wc, lam) chosen per caption type.  wc is pinned away from 0 so the four
    # options can never collapse onto each other.
    Vspan = SPN - m_span
    masks = {"negative": gate & (typ == "negative"),
             "hybrid": gate & (typ == "hybrid"),
             "positive": ~gate}

    def compose(P):
        out = U.copy()
        for t, m in masks.items():
            wa, wm, wc, lam = P[t]
            out[m] = wa * REM[m] + wm * m_pos_h + wc * out[m] - lam * Vspan[m]
        return out

    GRID = {0: [0.0, 0.15, 0.3, 0.5, 0.8, 1.2],          # weight on the affirmative remainder
            1: [0.0, 0.3, 0.6, 1.0, 1.4, 1.8],           # weight on the positive-caption anchor
            2: [0.05, 0.15, 0.3, 0.5, 0.8, 1.2],         # weight kept on the caption itself
            3: [0.0, 0.15, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0]}   # strength of the span direction

    def coordinate_descent(IMGmat=None, rounds=8):
        """Greedy coordinate ascent on tr_rows; also collects every tie-free point visited."""
        P = {"negative": [0.15, 1.0, 0.15, 0.4],
             "hybrid": [0.5, 1.0, 0.30, 0.4],
             "positive": [0.0, 0.0, 1.00, 0.0]}
        best, _ = sel.train_score(compose(P), IMGmat)
        visited = []
        for _ in range(rounds):
            improved = False
            for t in ("negative", "hybrid", "positive"):
                for k in range(4):
                    if t == "positive" and k in (0, 3):   # positives have no span to steer with
                        continue
                    for v in GRID[k]:
                        old, P[t][k] = P[t][k], v
                        s, d = sel.train_score(compose(P), IMGmat)
                        if d is not None:
                            visited.append((s, d, copy.deepcopy(P)))
                        if s > best + 1e-9:
                            best, improved = s, True
                        else:
                            P[t][k] = old
            if not improved:
                break
        return P, best, visited

    P_raw, s_raw, visited_raw = coordinate_descent()
    C.report("ex4_span_subspace", D, compose(P_raw),
             extra={"operator": "per-input span-conditional subspace operator",
                    "weights": P_raw, "train_all": s_raw, "selected_on": "tr_rows"})

    # Centring the image cloud is orthogonal to all of this and composes with it, so the
    # weights are re-selected with it on rather than stacked on top of a pre-tuned operator.
    IMGc = D["IMG"] - D["IMG"][np.unique(D["img_idx"][tr])].mean(0)
    P_ic, s_ic, visited_ic = coordinate_descent(IMGc)
    C.report("ex4_span_subspace_imgcentred", D, compose(P_ic), IMGmat=IMGc,
             extra={"operator": "span-conditional subspace operator + centred image cloud",
                    "weights": P_ic, "train_all": s_ic, "selected_on": "tr_rows"})

    # An operating point aimed at the subset everything else collapses on: maximise the
    # *negative* template on the training half subject to keeping positives above 80%.
    pool = [p for p in (visited_raw + visited_ic) if p[1]["positive"] >= 80.0]
    if pool:
        pool.sort(key=lambda p: -p[1]["negative"])
        _, d_mn, P_mn = pool[0]
        use_ic = any(P_mn == p[2] for p in visited_ic)
        C.report("ex4_span_subspace_maxneg", D, compose(P_mn),
                 IMGmat=IMGc if use_ic else None,
                 extra={"operator": "span-conditional operator, tuned for the negative subset",
                        "weights": P_mn, "train_breakdown": d_mn,
                        "img_centred": bool(use_ic),
                        "criterion": "argmax train negative s.t. train positive >= 80%",
                        "selected_on": "tr_rows"})

    # ---- Eq. 2 first, then the subspace operator on top of the corrected embeddings ---------
    base = eq2(lam_id, m_span)
    b_ood = {r: pca_basis(DELTA_OOD, r) for r in RANKS}
    b_in = {r: pca_basis(DELTA_IN, r) for r in RANKS}
    for tag, bs in (("ood", b_ood), ("indomain", b_in)):
        grid = [(r, b) for r in RANKS
                for b in (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, 1.5)]
        (r, b), s, d = sel.sweep(lambda p: op_scale(base, gate, bs[p[0]], p[1]), grid)
        C.report(f"ex4_eq2_then_subspace_{tag}", D, op_scale(base, gate, bs[r], b),
                 extra={"operator": f"Eq.2 (in-domain centre) then rank-r scaling, {tag} basis",
                        "lambda": lam_id, "rank": r, "beta": b, "train_breakdown": d,
                        "selected_on": "tr_rows"})

    (C.RESULTS / "ex4_geometry_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ========================================================================================
    # scoreboard
    # ========================================================================================
    print("\n" + "=" * 92)
    print("SCOREBOARD  (sorted by TEST accuracy; selection was done on tr_rows only)")
    print("=" * 92)
    rows = []
    for f in sorted(C.RESULTS.glob("ex4_*.json")):
        if f.name == "ex4_geometry_summary.json":
            continue
        j = json.loads(f.read_text())
        rows.append((j["test"]["all"], j["train"]["all"], j["test"], j.get("tie_rate_test", 0.0),
                     j["technique"]))
    rows.sort(reverse=True)
    print(f"{'technique':44s} {'train':>7s} {'TEST':>7s} {'pos':>7s} {'neg':>7s} {'hyb':>7s} {'ties':>6s}")
    for a, trn, t, ties, name in rows:
        print(f"{name:44s} {100 * trn:7.2f} {100 * a:7.2f} {100 * t['positive']:7.2f} "
              f"{100 * t['negative']:7.2f} {100 * t['hybrid']:7.2f} {100 * ties:6.2f}")


if __name__ == "__main__":
    main()
