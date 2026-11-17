"""ex2 — compositional reconstruction of a caption embedding from its spans.

Motivation
----------
CLIP text embeddings compose additively:  e("A and B") ~= e(A) + e(B) + a, for a constant
bias `a` (the anchor).  A NegBench caption is a *signed* conjunction: some objects are
asserted, some are denied.  So instead of patching the caption embedding with a single
projection (Aggarwal et al., Eq. 2), rebuild it from its parts:

        e* = sum_i e(aff_i) + k * anchor - beta * sum_j e(neg_j)

beta > 0 does something plain removal cannot: it makes the text vector actively *repel*
images that contain the denied object, rather than merely being agnostic about it.

Three geometry upgrades on top of that skeleton, each swept on train only:
  * prompt ensembling of each span (average of unit embeddings over 8 carrier templates);
  * image centring  I <- I - c * mean(I over TRAIN images), c swept;
  * a count-dependent anchor  (k0 + k1*n_aff + k2*n_neg) * anchor, which is exactly a
    per-span decision threshold tau written in embedding space;
  * per-caption-class scalars: common.caption_type() labels each caption from its text
    alone, so conditioning on it is legal, and a pure-absence caption plainly needs a
    different beta from a hybrid one.

Protocol
--------
Every scalar (k, beta, w, tau, c, lambda) is selected on D["tr_rows"]; every number
reported is on D["te_rows"].  Nothing is ever selected on test.

Run:  python ex2_compositional_spans.py
"""

from __future__ import annotations

import itertools
import re

import numpy as np

import common

RNG = np.random.default_rng(0)

# ============================================================================================
# 1. PARSER — all affirmative spans + all negated spans
# ============================================================================================
# common.extract_negated returns only the *first* negation scope and throws the rest of the
# caption away.  Here we (a) cut the caption into clauses, (b) find the negator inside each
# clause, (c) assign the in-scope tokens to `neg` and the out-of-scope tokens to `aff`, and
# (d) split each side into individual object phrases.

TOK = re.compile(r"[a-z']+|[.,;]")

# Aggarwal's tables, plus the cues this corpus actually uses that they miss.
PRE_NEGATORS = common.PRE_NEGATORS + [
    "won't", "wont", "will not", "would not", "wouldn't", "sans", "minus",
    "excluding", "excludes", "exclude", "void of", "bereft of", "short of",
]
POST_NEGATORS = common.POST_NEGATORS + [
    # "<X> is noticeably absent", "<X> is nowhere to be seen", ... — scope is what precedes.
    "noticeably absent", "notably absent", "conspicuously absent", "strikingly absent",
    "nowhere to be seen", "nowhere to be found", "cannot be seen", "can't be seen",
    "absent", "missing", "excluded", "omitted",
]
PRE_NEGATORS = sorted(set(PRE_NEGATORS), key=lambda s: -len(s.split()))
POST_NEGATORS = sorted(set(POST_NEGATORS), key=lambda s: -len(s.split()))

BOUNDARY_WORDS = common.BOUNDARY_WORDS | {"despite", "unlike", "besides", "meanwhile"}
BOUNDARY_PUNCT = {".", ";", ","}

# scaffolding: template words that carry no object identity.  Removing them leaves the noun.
SCAFF = set("""
this that these those it its the a an image images photo photos photograph picture pic
scene view frame shot snapshot composition
features feature featuring featured shows show showing shown displays display displayed
depicts depict depicting depicted contains contain containing includes include including
included has have had is are was were be been being am
there here in on at of to into within inside among amongst upon over under
present presence visible seen sight appear appears appearing appeared exist exists existed
found notice noticed noticeable observable captured capture portrays portray portrayed
prominently notably noticeably conspicuously clearly apparently evidently indeed truly really
any some anyone anybody anything one ones thing things other others else
what where which who whom whose when
we you i they he she our your their his her my me us them
find finds finding see sees seeing look looks looking spot spots spotted
part element item object subject piece bit kind sort type
rest nearby around anywhere somewhere
""".split())
# every negation cue is scaffolding *inside* a span (handles doubled cues: "no book and no pen")
SCAFF |= {w for ph in PRE_NEGATORS + POST_NEGATORS for w in ph.split()}

# words that separate two distinct objects inside one span
SPLIT_WORDS = {",", "and", "or", "nor", "plus", "alongside", "accompanied", "along",
               "together", "addition", "additionally", "well", "aside", "next", "beside",
               "near", "beyond", "wearing", "seated", "sitting", "holding", "carrying",
               "showcases", "showcase", "than", "except", "versus"}

_S_SINGULAR = {"skis", "scissors", "bus", "glass", "wine glass", "grass"}


def _clauses(caption: str) -> list[list[str]]:
    """Cut a caption into clauses on contrastive conjunctions and hard punctuation."""
    t = re.sub(r"\([^)]*\)", " ", str(caption).lower())   # drop LLM meta-commentary
    t = t.replace('"', " ")
    out, cur = [], []
    for tk in TOK.findall(t):
        if tk in BOUNDARY_PUNCT or tk in BOUNDARY_WORDS:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(tk)
    if cur:
        out.append(cur)
    return out


def _find_negator(toks: list[str]):
    """(index, n_words, kind) of the first negator in a clause, longest match wins."""
    for i in range(len(toks)):
        best = None
        for kind, table in (("post", POST_NEGATORS), ("pre", PRE_NEGATORS)):
            for ph in table:
                w = ph.split()
                if toks[i:i + len(w)] == w and (best is None or len(w) > best[1]):
                    best = (i, len(w), kind)
            if best:                       # a post-match at i wins over a pre-match at i
                break
        if best:
            return best
    return None


def _phrases(toks: list[str]) -> list[str]:
    """Content tokens -> individual object phrases (scaffolding stripped, list split)."""
    groups, cur = [], []
    for t in toks:
        if t in SPLIT_WORDS:
            if cur:
                groups.append(cur)
            cur = []
        elif t in SCAFF or t.endswith("'s"):
            continue
        else:
            cur.append(t)
    if cur:
        groups.append(cur)
    # <=3 tokens keeps "traffic light"/"baseball glove" and drops LLM prose leftovers
    return [" ".join(g) for g in groups if 0 < len(g) <= 3]


def parse_spans(caption: str) -> tuple[list[str], list[str]]:
    """caption -> (affirmative phrases, negated phrases), order-preserving and deduped."""
    aff, neg = [], []
    for cl in _clauses(caption):
        m = _find_negator(cl)
        if m is None:
            aff += _phrases(cl)
        else:
            i, n, kind = m
            if kind == "pre":              # scope follows the cue
                neg += _phrases(cl[i + n:])
                aff += _phrases(cl[:i])
            else:                          # scope precedes the cue
                neg += _phrases(cl[:i])
                aff += _phrases(cl[i + n:])
    ded = lambda xs: list(dict.fromkeys(xs))
    aff, neg = ded(aff), ded(neg)
    # an object cannot be both asserted and denied in one caption; the denial wins
    aff = [p for p in aff if p not in set(neg)]
    return aff, neg


# ---- how a bare object phrase becomes text we can embed -------------------------------
CARRIERS = ["a photo of {A}.",
            "a photo of {A} in the scene.",
            "this image features {A}.",
            "an image containing {A}.",
            "there is {A} in this photo.",
            "a cropped photo of {A}.",
            "a blurry photo of {A}.",
            "a close-up photo of {A}."]
_SOLO = 2                                  # "this image features a chair." — the dataset's voice


def _np(phrase: str) -> str:
    """'chair' -> 'a chair';  'cars' -> 'cars';  'orange' -> 'an orange'."""
    if phrase.endswith("s") and phrase not in _S_SINGULAR:
        return phrase
    return ("an " if phrase[:1] in "aeiou" else "a ") + phrase


def render(phrase: str, mode: str = "tmpl") -> list[str]:
    """Texts to embed for one span. 'raw' = the bare phrase, 'tmpl' = one carrier,
    'ens' = the full carrier ensemble (averaged as unit vectors by span_embeddings)."""
    if mode == "raw":
        return [phrase]
    if mode == "tmpl":
        return [CARRIERS[_SOLO].format(A=_np(phrase))]
    return [c.format(A=_np(phrase)) for c in CARRIERS]


def validate_parser(D, n=20):
    """Print n parsed examples per template type so parse quality is visible."""
    df, tmpl = D["df"], D["tmpl"]
    print("\n" + "=" * 96)
    print("PARSER VALIDATION — 20 captions per template type (gold caption of the first 20 rows)")
    print("=" * 96)
    for t in ("positive", "negative", "hybrid"):
        print(f"\n--- {t.upper()} ---")
        for r in np.where(tmpl == t)[0][:n]:
            cap = str(df.loc[r, f"caption_{D['gold'][r]}"])
            a, g = parse_spans(cap)
            print(f"  {cap[:76]:<76}\n      AFF={a}  NEG={g}")

    from collections import Counter
    pa = [parse_spans(c) for c in D["flat"]]
    tot = len(pa)
    empty = sum(1 for a, g in pa if not a and not g)
    withneg = sum(1 for a, g in pa if g)
    ca = Counter(p for a, _ in pa for p in a)
    cg = Counter(p for _, g in pa for p in g)
    noise = sum(c for p, c in ca.items() if c < 15)
    print(f"\n  corpus: {tot} captions | empty parses {empty} ({100*empty/tot:.2f}%) | "
          f"with >=1 negated span {withneg} ({100*withneg/tot:.1f}%)")
    print(f"  distinct phrases: aff {len(ca)}, neg {len(cg)} "
          f"(COCO has 80 object classes, so the tail is parse noise)")
    print(f"  top aff {[p for p, _ in ca.most_common(6)]}")
    print(f"  top neg {[p for p, _ in cg.most_common(6)]}")
    print(f"  aff phrases seen <15x = the noise floor: {noise} tokens "
          f"({100*noise/max(sum(ca.values()),1):.2f}% of all affirmative spans)")


# ============================================================================================
# 2. FAST SCORER — every variant is a linear combination of a few per-caption basis vectors
# ============================================================================================
# score(caption) = <img_hat, v> / ||v||   with   v = sum_k c_k * B_k[caption].
# Precomputing <img_hat, B_k> and <B_k, B_l> per caption makes one coefficient evaluation
# O(n * m^2) instead of O(n * d), so a dense sweep or a coordinate ascent is nearly free.
# The Gram G does not depend on the images, so swapping the image matrix only redoes `d`.

class Basis:
    def __init__(self, D, mats: dict[str, np.ndarray], IMG=None):
        self.names = list(mats)
        self.mats = [np.asarray(mats[k], dtype=np.float64) for k in self.names]
        self.img_idx = D["img_idx"]
        m = len(self.mats)
        self.G = np.empty((m, m, self.mats[0].shape[0]))
        for i in range(m):
            for j in range(i, m):
                g = np.einsum("ij,ij->i", self.mats[i], self.mats[j])
                self.G[i, j] = self.G[j, i] = g
        self.set_images(D["IMG"] if IMG is None else IMG)

    def set_images(self, IMG):
        I = np.repeat(common.l2n(IMG)[self.img_idx], 4, axis=0)       # (4n, d)
        self.d = np.stack([np.einsum("ij,ij->i", I, b) for b in self.mats])
        return self

    def scores(self, coef, normalise=True):
        c = np.asarray(coef, dtype=np.float64)
        num = c @ self.d
        if not normalise:
            return num.reshape(-1, 4)
        den = np.sqrt(np.clip(np.einsum("k,l,kln->n", c, c, self.G), 1e-24, None))
        return (num / den).reshape(-1, 4)

    def vector(self, coef):
        """The explicit steered caption matrix, for common.report."""
        return sum(c * b for c, b in zip(coef, self.mats))


def acc(D, S, rows):
    return float((S[rows].argmax(1) == D["gold"][rows]).mean())


def sweep(D, basis, coef_fn, grids, normalise=True):
    """Exhaustive grid search of coef_fn(*params) on tr_rows -> (best_params, tr_acc)."""
    tr = D["tr_rows"]
    best, best_a = None, -1.0
    for params in itertools.product(*grids):
        a = acc(D, basis.scores(coef_fn(*params), normalise), tr)
        if a > best_a:
            best_a, best = a, params
    return best, best_a


def polish(D, basis, coef, normalise=True, rounds=10, step=0.8):
    """Coordinate ascent on TRAIN accuracy from a seed — still train-only."""
    tr = D["tr_rows"]
    c = np.asarray(coef, dtype=np.float64).copy()
    best = acc(D, basis.scores(c, normalise), tr)
    s = step
    for _ in range(rounds):
        improved = False
        for i in range(len(c)):
            for delta in (s, -s, s / 3, -s / 3):
                t = c.copy()
                t[i] += delta
                a = acc(D, basis.scores(t, normalise), tr)
                if a > best + 1e-12:
                    best, c, improved = a, t, True
        if not improved:
            s *= 0.4
    return c, best


def multistart(D, basis, seeds, n_random=4, **kw):
    m = len(basis.names)
    best_c, best_a = None, -1.0
    for s in list(seeds) + [RNG.normal(0, 0.6, m) for _ in range(n_random)]:
        c, a = polish(D, basis, np.asarray(s, float), **kw)
        if a > best_a:
            best_c, best_a = c, a
    return best_c, best_a


BOARD = []


def record(name, D, basis, coef, normalise=True, IMG=None, extra=None):
    """Score, print, and persist one final variant (test numbers are never tuned on)."""
    S = basis.scores(coef, normalise)
    tr, te = {}, {}
    tr["all"], te["all"] = acc(D, S, D["tr_rows"]), acc(D, S, D["te_rows"])
    for t in ("positive", "negative", "hybrid"):
        for dst, rows in ((tr, D["tr_rows"]), (te, D["te_rows"])):
            sel = rows[D["tmpl"][rows] == t]
            dst[t] = acc(D, S, sel) if len(sel) else float("nan")
    ex = {"coef": {n: float(c) for n, c in zip(basis.names, coef)},
          "normalise": bool(normalise), **(extra or {})}
    if normalise:
        # cosine scoring is exactly common.scores, so persist through the harness
        common.report(name, D, basis.vector(coef), IMGmat=IMG, extra=ex)
    else:
        print(f"[{name}]  train {100*tr['all']:.2f}%   TEST {100*te['all']:.2f}%   "
              f"(pos {100*te['positive']:.1f} / neg {100*te['negative']:.1f} / "
              f"hyb {100*te['hybrid']:.1f})   [dot-product scorer, not a cosine]")
        common.save_result(name, {"train": tr, "test": te, **ex})
    BOARD.append((name, tr["all"], te["all"], te["positive"], te["negative"], te["hybrid"]))
    return te["all"]


# ============================================================================================
# 3. SPAN EMBEDDINGS
# ============================================================================================
def span_embeddings(parsed, mode):
    """Per-caption sums of affirmative / negated span embeddings.

    Spans are averaged over their carrier prompts as unit vectors, then re-normalised, so a
    span contributes a unit vector and the sum is a genuine composition of equal parts.
    Captions with no affirmative span get the zero vector — deliberately: a pure denial
    "no chair" must be represented as k*anchor - beta*e(chair), i.e. a vector that points
    away from chairs, not as a fallback to the caption embedding.
    """
    E = lambda t: common.embed_texts([t])[0]
    cache: dict[str, np.ndarray] = {}

    def emb(p):
        if p not in cache:
            V = np.stack([common.l2n(E(t)) for t in render(p, mode)])
            cache[p] = common.l2n(V.mean(0))
        return cache[p]

    n4 = len(parsed)
    d = len(emb(parsed[0][0][0] if parsed[0][0] else parsed[0][1][0]))
    A, N = np.zeros((n4, d)), np.zeros((n4, d))
    na, nn = np.zeros(n4), np.zeros(n4)
    for i, (a, g) in enumerate(parsed):
        for p in a:
            A[i] += emb(p)
        for p in g:
            N[i] += emb(p)
        na[i], nn[i] = len(a), len(g)
    return A, N, na, nn


# ============================================================================================
# 4. THE WINNING CONFIGURATION, exposed for cross-experiment fusion
# ============================================================================================
BEST = {}          # filled by main(); {"coef":…, "c":…, "names":…}


def build_best(D):
    """Return (CAPmat, IMGmat) for the best train-selected configuration.

    CAPmat is (4n, d) steered caption embeddings; IMGmat is (n_img, d) centred image
    embeddings.  Score with common.scores(D, CAPmat, IMGmat) / common.report(...).
    """
    flat = D["flat"]
    parsed = [parse_spans(c) for c in flat]
    A, N, na, nn = span_embeddings(parsed, "ens")
    ANCM = np.tile(common.anchor(), (len(flat), 1))
    ctype = np.array([common.caption_type(c) for c in flat])
    base = _base_mats(D, A, N, na, nn, ANCM)
    mats = _per_class(base, ctype)
    b = Basis(D, mats)
    if not BEST:
        _fit_best(D, b, base, ctype)
    IMG = _centred(D, BEST["c"])
    return b.vector(BEST["coef"]), IMG


def _centred(D, c):
    """I - c * mean(I over TRAIN images).  The mean uses train images only, no test leakage."""
    In = common.l2n(D["IMG"])
    mu = In[np.unique(D["img_idx"][D["tr_rows"]])].mean(0)
    return In - c * mu


def _base_mats(D, A, N, na, nn, ANCM):
    """The reconstruction basis.  anc / anc_na / anc_nn together give the anchor a
    count-dependent weight (k0 + k1*n_aff + k2*n_neg) — which is exactly a per-span
    decision threshold tau written in embedding space."""
    return {"aff": A, "anc": ANCM, "anc_na": na[:, None] * ANCM,
            "anc_nn": nn[:, None] * ANCM, "neg": N, "cap": D["CAP"]}


def _per_class(base, ctype):
    """Mask every basis vector by the caption's class -> independent scalars per class."""
    mats = {}
    for cl in ("positive", "negative", "hybrid"):
        m = (ctype == cl).astype(float)[:, None]
        for k, v in base.items():
            mats[f"{k}|{cl}"] = v * m
    return mats


def _fit_best(D, b_cls, base, ctype, c_grid=np.round(np.arange(0.55, 1.001, 0.05), 3)):
    """Sweep the centring coefficient c and fit per-class scalars, all on TRAIN."""
    b_glb = Basis(D, base)
    best = (-1.0, None, None, None)
    for c in c_grid:
        IMG = _centred(D, c)
        b_glb.set_images(IMG)
        cg, ag = multistart(D, b_glb, [np.eye(len(base))[0]], n_random=4, rounds=10)
        b_cls.set_images(IMG)
        cc, ac = multistart(D, b_cls, [np.concatenate([cg] * 3)], n_random=2, rounds=12)
        print(f"    c={c:.2f}   global {100*ag:5.2f}%   per-class {100*ac:5.2f}%  (train)")
        if ac > best[0]:
            best = (ac, c, cc, cg)
    BEST.update(train=best[0], c=float(best[1]), coef=best[2], coef_global=best[3],
                names=list(b_cls.names), names_global=list(base))
    return BEST


# ============================================================================================
# 5. MAIN
# ============================================================================================
def main():
    D = common.load_mcq()
    flat, n4 = D["flat"], len(D["flat"])
    print(f"[ex2] {D['n']} questions, {n4} captions, "
          f"train {len(D['tr_rows'])} / test {len(D['te_rows'])} rows")

    validate_parser(D)

    # ---- parse everything, then embed every distinct span text in ONE batched call ----
    parsed = [parse_spans(c) for c in flat]
    eq2_neg = [common.extract_negated(c) for c in flat]      # published baseline's scope
    need = set(common.ANCHOR_WORDS) | {s for s in eq2_neg if s}
    for a, g in parsed:
        for p in a + g:
            for mode in ("raw", "tmpl", "ens"):
                need |= set(render(p, mode))
    print(f"\n[ex2] embedding {len(need)} distinct span texts (one batched call)")
    common.embed_texts(sorted(need))

    E = lambda t: common.embed_texts([t])[0]
    C, d = D["CAP"], D["CAP"].shape[1]
    ANC = common.anchor()
    ANCM = np.tile(ANC, (n4, 1))
    ctype = np.array([common.caption_type(c) for c in flat])

    A_e, N_e, na, nn = span_embeddings(parsed, "ens")        # prompt-ensembled spans
    A_t, N_t, _, _ = span_embeddings(parsed, "tmpl")         # single carrier
    A_r, N_r, _, _ = span_embeddings(parsed, "raw")          # bare phrase
    print(f"[ex2] spans/caption: aff mean {na.mean():.2f}, neg mean {nn.mean():.2f}; "
          f"{int((nn > 0).sum())} captions carry a negated span")

    KS = np.round(np.arange(-4.0, 4.01, 0.25), 3)
    BS = np.round(np.arange(0.0, 2.51, 0.05), 3)
    WS = np.round(np.arange(0.0, 1.001, 0.05), 3)

    # ---------------------------------------------------------------- baselines ----
    print("\n" + "=" * 96); print("BASELINES"); print("=" * 96)
    record("ex2_baseline_plain_clip", D, Basis(D, {"cap": C}), [1.0])

    # published Eq. 2:  e* = e^c - lambda * (proj_{e^neg}(e^c) - a)
    P, has = np.zeros((n4, d)), np.zeros(n4, bool)
    for i, s in enumerate(eq2_neg):
        if s:
            u = E(s)
            P[i] = (C[i] @ u) / max(u @ u, 1e-12) * u
            has[i] = True
    b_eq2 = Basis(D, {"cap": C, "proj": P, "anc": np.where(has[:, None], ANCM, 0.0)})
    (lam,), tra = sweep(D, b_eq2, lambda l: [1.0, -l, l],
                        [np.round(np.arange(0.0, 2.01, 0.02), 3)])
    print(f"  Eq.2 lambda* = {lam} (train {100*tra:.2f}%)")
    record("ex2_baseline_eq2_published", D, b_eq2, [1.0, -lam, lam], extra={"lambda": float(lam)})

    # ------------------------------------- the brief's formula, swept as written ----
    print("\n" + "=" * 96)
    print("COMPOSITIONAL RECONSTRUCTION   e* = sum(aff) + k*anchor - beta*sum(neg)")
    print("=" * 96)
    recons = {"raw": (A_r, N_r), "tmpl": (A_t, N_t), "ens": (A_e, N_e)}
    bases = {}
    for tag, (A, N) in recons.items():
        b = Basis(D, {"aff": A, "anc": ANCM, "neg": N})
        bases[tag] = b
        (k, beta), tr_ = sweep(D, b, lambda k, be: [1.0, k, -be], [KS, BS])
        print(f"  [{tag:<4}] k*={k:<6} beta*={beta:<5} (train {100*tr_:.2f}%)")
        record(f"ex2_recon_{tag}", D, b, [1.0, k, -beta],
               extra={"k": float(k), "beta": float(beta), "render": tag})
    k_e, beta_e = sweep(D, bases["ens"], lambda k, be: [1.0, k, -be], [KS, BS])[0]

    # beta = 0 ablation: what does the repulsion term actually buy?
    (k0,), tr0 = sweep(D, bases["ens"], lambda k: [1.0, k, 0.0], [KS])
    print(f"  [ens, beta=0 ablation] k*={k0} (train {100*tr0:.2f}%)")
    record("ex2_recon_ens_beta0", D, bases["ens"], [1.0, k0, 0.0],
           extra={"k": float(k0), "beta": 0.0, "note": "ablation: no repulsion term"})

    # caption embedding in place of the affirmative sum
    b_cap = Basis(D, {"cap": C, "anc": ANCM, "neg": N_e})
    (kc, bc), trc = sweep(D, b_cap, lambda k, be: [1.0, k, -be], [KS, BS])
    print(f"  [cap-as-aff] k*={kc} beta*={bc} (train {100*trc:.2f}%)")
    record("ex2_recon_capaff", D, b_cap, [1.0, kc, -bc], extra={"k": float(kc), "beta": float(bc)})

    # blend  w*e^c + (1-w)*reconstruction
    b_bl = Basis(D, {"cap": C, "aff": A_e, "anc": ANCM, "neg": N_e})
    (w, kb, bb), trb = sweep(D, b_bl, lambda w, k, be: [w, 1 - w, (1 - w) * k, -(1 - w) * be],
                             [WS, KS, BS])
    print(f"  [blend] w*={w} k*={kb} beta*={bb} (train {100*trb:.2f}%)")
    record("ex2_recon_blend", D, b_bl, [w, 1 - w, (1 - w) * kb, -(1 - w) * bb],
           extra={"w": float(w), "k": float(kb), "beta": float(bb)})

    # ------------------------------------------------- the difference scorer -------
    print("\n" + "=" * 96)
    print("DIFFERENCE SCORER   score = cos(img, aff) - beta*cos(img, neg)")
    print("=" * 96)
    A_h, N_h = common.l2n(A_e), np.where(nn[:, None] > 0, common.l2n(N_e), 0.0)
    b_df = Basis(D, {"affhat": A_h, "neghat": N_h})
    (bd,), trd = sweep(D, b_df, lambda be: [1.0, -be], [BS], normalise=False)
    Ihat = np.repeat(common.l2n(D["IMG"])[D["img_idx"]], 4, axis=0)
    direct = (np.einsum("ij,ij->i", Ihat, A_h)
              - bd * np.einsum("ij,ij->i", Ihat, N_h)).reshape(-1, 4)
    dot_s = b_df.scores([1.0, -bd], normalise=False)
    cos_s = b_df.scores([1.0, -bd], normalise=True)
    nv = np.linalg.norm(b_df.vector([1.0, -bd]), axis=1)
    print(f"  equivalence: max|cos(img,aff) - beta*cos(img,neg)  -  <img_hat, v>| = "
          f"{np.abs(direct - dot_s).max():.2e}")
    print(f"  -> the difference scorer IS the single steered vector v = aff_hat - beta*neg_hat,")
    print(f"     scored by DOT product.  Cosine scoring divides by ||v||, which is NOT constant")
    print(f"     across captions (||v|| in [{nv.min():.3f}, {nv.max():.3f}], because a caption "
          f"with no\n     negated span has ||v||=1 while a hybrid one does not), so the two "
          f"disagree on\n     {100*(dot_s.argmax(1) != cos_s.argmax(1)).mean():.2f}% of rows.")
    record("ex2_diffscore_dot", D, b_df, [1.0, -bd], normalise=False,
           extra={"beta": float(bd), "note": "score = cos(img,aff) - beta*cos(img,neg)"})
    (bdc,), _ = sweep(D, b_df, lambda be: [1.0, -be], [BS], normalise=True)
    record("ex2_diffscore_cos", D, b_df, [1.0, -bdc], normalise=True,
           extra={"beta": float(bdc), "note": "same vector, cosine-scored"})

    # -------------------------------------------------- geometry upgrades ----------
    print("\n" + "=" * 96)
    print("GEOMETRY UPGRADES  (image centring, count-dependent anchor, per-class scalars)")
    print("=" * 96)
    base = _base_mats(D, A_e, N_e, na, nn, ANCM)
    b_glb = Basis(D, base)
    b_cls = Basis(D, _per_class(base, ctype))

    # (a) count-dependent anchor, uncentred images — isolates that one term
    cg0, ag0 = multistart(D, b_glb, [np.eye(len(base))[0]], n_random=4)
    print(f"  count-aware anchor, uncentred images: train {100*ag0:.2f}%")
    record("ex2_recon_countanchor", D, b_glb, cg0,
           extra={"note": "anchor weight = k0 + k1*n_aff + k2*n_neg"})

    # (b) + centring sweep, global and per-class scalars
    print("  sweeping the image-centring coefficient c (train only):")
    _fit_best(D, b_cls, base, ctype)
    c_star = BEST["c"]
    IMGc = _centred(D, c_star)
    b_glb.set_images(IMGc)
    record("ex2_recon_centred_global", D, b_glb, BEST["coef_global"], IMG=IMGc,
           extra={"c": c_star, "note": "one (k, beta, tau) shared by all caption classes"})
    b_cls.set_images(IMGc)
    print(f"\n  >>> WINNER: c*={c_star}, per-class scalars")
    record("ex2_recon_perclass", D, b_cls, BEST["coef"], IMG=IMGc,
           extra={"c": c_star, "note": "independent (k, beta, tau) per common.caption_type"})

    # (c) ablation of the repulsion term inside the winner
    kill = np.array([0.0 if n.startswith("neg") else 1.0 for n in b_cls.names])
    c_ab, a_ab = multistart(D, b_cls, [BEST["coef"] * kill], n_random=0, rounds=12)
    c_ab = c_ab * kill
    record("ex2_recon_perclass_beta0", D, b_cls, c_ab, IMG=IMGc,
           extra={"c": c_star, "note": "WINNER with beta forced to 0 (no repulsion)"})

    # ---- what beta buys, printed per caption class ----
    print("\n  fitted per-class scalars of the winner "
          "(v = aff + (k0+k1*n_aff+k2*n_neg)*anchor - beta*neg + g*cap):")
    cf = dict(zip(b_cls.names, BEST["coef"]))
    print(f"    {'class':<10} {'aff':>7} {'k0':>7} {'k1':>7} {'k2':>7} {'beta':>7} {'cap':>7}")
    for cl in ("positive", "negative", "hybrid"):
        s = cf[f"aff|{cl}"] or 1.0
        print(f"    {cl:<10} {cf[f'aff|{cl}']:7.3f} {cf[f'anc|{cl}']:7.3f} "
              f"{cf[f'anc_na|{cl}']:7.3f} {cf[f'anc_nn|{cl}']:7.3f} "
              f"{-cf[f'neg|{cl}']:7.3f} {cf[f'cap|{cl}']:7.3f}")

    # -------------------------------------------------- error analysis -------------
    print("\n" + "=" * 96); print("ERROR ANALYSIS on the held-out half"); print("=" * 96)
    S = b_cls.scores(BEST["coef"])
    te = D["te_rows"]
    pred = S.argmax(1)
    sig = [(tuple(sorted(a)), tuple(sorted(g))) for a, g in parsed]
    n_par = n_amb = n_geo = 0
    for r in te:
        if pred[r] == D["gold"][r]:
            continue
        gs = sig[4 * r + D["gold"][r]]
        others = [sig[4 * r + j] for j in range(4) if j != D["gold"][r]]
        if not gs[0] and not gs[1]:
            n_par += 1
        elif gs in others:
            n_amb += 1
        else:
            n_geo += 1
    tot = n_par + n_amb + n_geo
    print(f"  {tot} errors on {len(te)} test rows")
    print(f"    gold caption parsed to nothing (parser failure) : {n_par:4d} ({100*n_par/tot:.1f}%)")
    print(f"    gold's span set identical to a distractor's     : {n_amb:4d} ({100*n_amb/tot:.1f}%)")
    print(f"    parse fine and unique, geometry chose wrong     : {n_geo:4d} ({100*n_geo/tot:.1f}%)")

    # per-object presence AUC: the ceiling this whole family is bounded by
    vocab = sorted({p for a, g in parsed for p in a + g})
    EVm = np.stack([common.l2n(np.stack([common.l2n(E(t)) for t in render(p, "ens")]).mean(0))
                    for p in vocab])
    vi = {p: i for i, p in enumerate(vocab)}
    Z = common.l2n(IMGc) @ EVm.T
    pres, absn = {}, {}
    for r in D["tr_rows"]:
        a, g = parsed[4 * r + D["gold"][r]]
        pres.setdefault(D["img_idx"][r], set()).update(a)
        absn.setdefault(D["img_idx"][r], set()).update(g)
    zp = np.array([Z[im, vi[p]] for im, ps in pres.items() for p in ps])
    zn = np.array([Z[im, vi[p]] for im, ps in absn.items() for p in ps])
    print(f"  per-object presence AUC of the span embeddings (train images, labels from the\n"
          f"  train gold captions only): {(zp[:, None] > zn[None, :]).mean():.4f}  "
          f"— this is the ceiling on any object-level scorer.")

    # -------------------------------------------------- scoreboard -----------------
    print("\n" + "=" * 96)
    print("SCOREBOARD (sorted by TEST accuracy; every scalar fitted on train only)")
    print("=" * 96)
    print(f"  {'technique':<34} {'train':>7} {'TEST':>7} | {'pos':>6} {'neg':>6} {'hyb':>6}")
    print("  " + "-" * 76)
    for nm, tr_, te_, p, g, h in sorted(BOARD, key=lambda r: -r[2]):
        print(f"  {nm:<34} {100*tr_:7.2f} {100*te_:7.2f} | "
              f"{100*p:6.1f} {100*g:6.1f} {100*h:6.1f}")

    # sanity: build_best reproduces the winner through the public entry point
    CAPmat, IMGmat = build_best(D)
    print(f"\n[ex2] build_best() check: test {100*common.accuracy(D, CAPmat, D['te_rows'], IMGmat):.2f}%")


if __name__ == "__main__":
    main()
