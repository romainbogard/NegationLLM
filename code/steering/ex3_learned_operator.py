"""ex3 — a *learned linear operator* instead of a single steering direction.

Experiments 1/2 in this folder steer a caption embedding by a translation:  e* = e - alpha * d.
That is the rank-0 case of a much larger family.  Here we replace the direction by a full
linear (optionally affine) map

        e*  =  l2n( l2n(e) @ W  +  b )                       W in R^{512x512}

and ask which member of that family is actually needed.  Three operator families are compared:

  ridge       W = argmin ||X W - Y||^2 + lam||W||^2                     (shrinks toward 0)
  procrustes  W = argmin ||X W - Y||^2  s.t. W'W = I                    (a rotation only)
  lowrank     W = I + A_r,  A_r the rank-r reduced-rank-regression fit of the residual Y - X
              (r = 1 is literally "steering with one direction, made input-dependent";
               r = d recovers the ridge-toward-identity solution, so the family interpolates
               between the identity and a full affine map)

THE OPERATOR NEVER SEES THE BENCHMARK.
-------------------------------------
W is fitted purely on *synthetic* caption/target pairs that we build ourselves:

  * the caption **templates** are mined from the NegBench captions of the TRAIN half only
    (`D["tr_rows"]`) — text only, no gold answers, no images, no test rows;
  * the **objects** that fill the templates come from an external word bank
    (`code/make_conjunction_dataset.py`, 240 concrete nouns / 12 categories) — deliberately
    *not* the COCO class names, so W has to generalise across the noun vocabulary;
  * an optional auxiliary block comes from `data/negation_dataset.csv` (10k affirmative /
    negated minimal pairs), which is a generic English negation corpus, not image captions.

For each synthetic source caption we know, by construction, which object is asserted present
(A) and which is asserted absent (N), so we can write down the *ideal* target embedding:

    target  =  l2n( base  -  beta * l2n e("a photo of a N") )
    base    =  l2n e("a photo of a A")            (or the conjunction, for two-object captions)
            =  the vocabulary-mean "generic photo" anchor when the caption asserts no object

i.e. "the picture this sentence is actually describing".  beta = 0 makes the target the pure
affirmative content ("drop the negated clause"); beta > 0 additionally pushes *away* from the
negated object.  beta is a hyper-parameter like any other.

PROTOCOL.  Every choice — beta, auxiliary-corpus weight, ridge strength, rank, intercept,
blend t, and which captions the operator is applied to — is selected by accuracy on
`D["tr_rows"]`; `D["te_rows"]` is read exactly once per reported variant, at the end.

Run:  python ex3_learned_operator.py            (add --quick for a small grid)
"""

from __future__ import annotations

import argparse
import collections
import random
import re
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # common.py
sys.path.insert(0, str(HERE.parent))   # make_conjunction_dataset.py

import common as C                                              # noqa: E402
from make_conjunction_dataset import WORDS, article             # noqa: E402

VOCAB = sorted({w for ws in WORDS.values() for w in ws})        # 240 external nouns

# COCO's 80 public class names.  Used ONLY as a masker, to turn real training-half captions
# into slotted templates.  They are dataset metadata, never labels, and they are not used to
# fill the templates.
COCO_CLASSES = (
    "person bicycle car motorcycle airplane bus train truck boat bench bird cat dog horse "
    "sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard kite skateboard surfboard bottle cup fork knife spoon bowl banana apple "
    "sandwich orange broccoli carrot pizza donut cake chair couch bed toilet tv TV laptop "
    "mouse remote keyboard microwave oven toaster sink refrigerator book clock vase scissors "
    "hairdrier toothbrush"
).split()
COCO_PHRASES = ["traffic light", "fire hydrant", "stop sign", "parking meter", "sports ball",
                "baseball bat", "baseball glove", "tennis racket", "wine glass", "hot dog",
                "potted plant", "dining table", "cell phone", "teddy bear", "hair drier"]
MASK_VOCAB = sorted(set(COCO_CLASSES) | set(COCO_PHRASES), key=len, reverse=True)
MASK_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in MASK_VOCAB) + r")\b")

TARGET_PROMPT = "a photo of {}"          # {} <- "a knife" / "a knife and a car"


# ==============================================================================================
# 1.  Template mining  (TRAIN-HALF CAPTION TEXT ONLY)
# ==============================================================================================
def _slot_roles(template: str) -> tuple[list[int], list[int]]:
    """Which slots of a template are asserted present, and which are negated.

    Reuses `common.extract_negated` (the published negation-scope parser): substitute readable
    placeholders for the slots, ask for the negation scope, and see which placeholders fall
    inside it.  "This image features a {1}, but no {2} is present."  ->  pos=[1], neg=[2].
    """
    probe = template.replace("{1}", " slotone ").replace("{2}", " slottwo ")
    scope = C.extract_negated(probe) or ""
    slots = [i for i in (1, 2) if ("{%d}" % i) in template]
    names = {1: "slotone", 2: "slottwo"}
    neg = [i for i in slots if names[i] in scope.split()]
    pos = [i for i in slots if i not in neg]
    return pos, neg


def mine_templates(D, min_count=8, verbose=True):
    """Turn the training-half captions into (template, pos_slots, neg_slots, weight) records."""
    tr = set(int(r) for r in D["tr_rows"])
    caps = [D["flat"][4 * r + k] for r in tr for k in range(4)]

    counter = collections.Counter()
    for cap in caps:
        n = [0]

        def rep(_m):
            n[0] += 1
            return "{%d}" % n[0]

        t = MASK_RE.sub(rep, cap)
        if 1 <= n[0] <= 2:
            counter[t] += 1

    tmpls = []
    for t, k in counter.most_common():
        if k < min_count:
            break
        pos, neg = _slot_roles(t)
        if len(neg) > 1:                       # captions negating two objects are vanishingly rare
            continue
        tmpls.append({"tmpl": t, "pos": pos, "neg": neg, "w": k})

    if verbose:
        tot = sum(t["w"] for t in tmpls)
        cov = tot / max(sum(counter.values()), 1)
        print(f"[ex3] mined {len(tmpls)} templates from the TRAIN half "
              f"({tot} caption instances, {100 * cov:.1f}% of maskable train captions)")
        for t in tmpls[:12]:
            print(f"       {t['w']:5d}  pos={t['pos']} neg={t['neg']}  |  {t['tmpl']}")
    return tmpls


# ==============================================================================================
# 2.  Synthetic caption generation
# ==============================================================================================
_ART = re.compile(r"\b(a|an)\s+([a-z]+)")


def fix_articles(s: str) -> str:
    """Templates carry 'a {1}'; the filler noun decides whether it should be 'a' or 'an'."""
    return _ART.sub(lambda m: f"{article(m.group(2))} {m.group(2)}", s)


def make_synthetic(tmpls, n=9000, seed=0):
    """Sample synthetic NegBench-style captions with their (A present, N absent) annotation."""
    rng = random.Random(seed)
    weights = [t["w"] for t in tmpls]
    rows = []
    for _ in range(n):
        t = rng.choices(tmpls, weights=weights, k=1)[0]
        words = rng.sample(VOCAB, k=len(t["pos"]) + len(t["neg"]))
        fill = dict(zip(t["pos"] + t["neg"], words))
        src = t["tmpl"]
        for slot, w in fill.items():
            src = src.replace("{%d}" % slot, w)
        rows.append({
            "src": fix_articles(src),
            "pos": [fill[s] for s in t["pos"]],
            "neg": [fill[s] for s in t["neg"]],
        })
    return rows


def aux_pairs(n=2000, seed=0):
    """Generic-English auxiliary block: negated sentence -> (affirmative, negated scope)."""
    import pandas as pd
    f = C.ROOT / "data" / "negation_dataset.csv"
    if not f.exists():
        return []
    df = pd.read_csv(f)
    df = df.sample(n=min(n, len(df)), random_state=seed)
    out = []
    for _, r in df.iterrows():
        scope = C.extract_negated(str(r["negated"]))
        if not scope:
            continue
        out.append({"src": str(r["negated"]), "aff": str(r["affirmative"]), "scope": scope})
    return out


# ==============================================================================================
# 3.  Targets
# ==============================================================================================
def build_target_bank(rows, aux):
    """Embed everything the targets are made of, once."""
    singles = sorted({w for r in rows for w in r["pos"] + r["neg"]})
    single_txt = [TARGET_PROMPT.format(f"{article(w)} {w}") for w in singles]
    conj = sorted({tuple(r["pos"]) for r in rows if len(r["pos"]) == 2})
    conj_txt = [TARGET_PROMPT.format(f"{article(a)} {a} and {article(b)} {b}") for a, b in conj]
    vocab_txt = [TARGET_PROMPT.format(f"{article(w)} {w}") for w in VOCAB]

    E = {}
    E["single"] = dict(zip(singles, C.l2n(C.embed_texts(single_txt))))
    E["conj"] = dict(zip(conj, C.l2n(C.embed_texts(conj_txt)))) if conj else {}
    E["neutral"] = C.l2n(C.l2n(C.embed_texts(vocab_txt)).mean(0))   # "a generic photo"
    if aux:
        E["aff"] = dict(zip([a["aff"] for a in aux],
                            C.l2n(C.embed_texts([a["aff"] for a in aux]))))
        E["scope"] = dict(zip([a["scope"] for a in aux],
                              C.l2n(C.embed_texts([a["scope"] for a in aux]))))
    return E


def targets_for(rows, aux, E, beta):
    """The ideal embedding of each source caption's *true meaning*, for a given beta."""
    Y = np.zeros((len(rows) + len(aux), E["neutral"].shape[0]))
    for i, r in enumerate(rows):
        if len(r["pos"]) == 2:
            base = E["conj"][tuple(r["pos"])]
        elif len(r["pos"]) == 1:
            base = E["single"][r["pos"][0]]
        else:
            base = E["neutral"]
        v = base.copy()
        for w in r["neg"]:
            v = v - beta * E["single"][w]
        Y[i] = v
    for j, a in enumerate(aux):
        Y[len(rows) + j] = E["aff"][a["aff"]] - beta * E["scope"][a["scope"]]
    return C.l2n(Y)


# ==============================================================================================
# 4.  Operator families
# ==============================================================================================
class Fitter:
    """Caches the Gram matrices so the (lam, rank, beta, intercept) grid is nearly free."""

    def __init__(self, X, w):
        self.X = X
        self.sw = np.sqrt(w)[:, None]
        Xw = X * self.sw
        self.n = float(w.sum())
        self.G = Xw.T @ Xw / self.n
        self.xbar = (X * w[:, None]).sum(0) / self.n
        Xc = (X - self.xbar) * self.sw
        self.Gc = Xc.T @ Xc / self.n
        self.Xw, self.Xcw, self.w = Xw, Xc, w
        self.d = X.shape[1]
        self.I = np.eye(self.d)

    def prep_Y(self, Y):
        Yw = Y * self.sw
        ybar = (Y * self.w[:, None]).sum(0) / self.n
        Ycw = (Y - ybar) * self.sw
        return {"XtY": self.Xw.T @ Yw / self.n, "XtYc": self.Xcw.T @ Ycw / self.n,
                "ybar": ybar, "Yw": Yw, "Ycw": Ycw}

    # ---- ridge (shrinkage toward the zero map) -----------------------------------------
    def ridge(self, P, lam, intercept):
        G, XtY = (self.Gc, P["XtYc"]) if intercept else (self.G, P["XtY"])
        W = np.linalg.solve(G + lam * self.I, XtY)
        b = P["ybar"] - self.xbar @ W if intercept else np.zeros(self.d)
        return W, b

    # ---- identity + low rank  (reduced-rank regression on the residual Y - X) ----------
    def lowrank(self, P, lam, ranks, intercept):
        G, XtY = (self.Gc, P["XtYc"]) if intercept else (self.G, P["XtY"])
        A = np.linalg.solve(G + lam * self.I, XtY - G)      # X^T(Y-X)/n = XtY - G
        M = A.T @ G @ A                                     # covariance of the fitted residual
        M = (M + M.T) / 2
        ev, V = np.linalg.eigh(M)
        order = np.argsort(ev)[::-1]
        V = V[:, order]
        out = {}
        for r in ranks:
            Vr = V[:, :min(r, self.d)]
            Ar = A @ (Vr @ Vr.T)
            W = self.I + Ar
            b = P["ybar"] - self.xbar @ W if intercept else np.zeros(self.d)
            out[r] = (W, b)
        return out

    # ---- orthogonal Procrustes ---------------------------------------------------------
    def procrustes(self, P, center):
        A, B = (self.Xcw, P["Ycw"]) if center else (self.Xw, P["Yw"])
        R, _ = orthogonal_procrustes(A, B)
        b = P["ybar"] - self.xbar @ R if center else np.zeros(self.d)
        return R, b


# ==============================================================================================
# 5.  Applying the operator to the benchmark
# ==============================================================================================
def apply_op(Xn, Y_op, mask, t):
    """e* = (1-t) e + t (W e) on the gated captions, untouched elsewhere.  Xn/Y_op normalised."""
    out = Xn.copy()
    out[mask] = (1.0 - t) * Xn[mask] + t * Y_op[mask]
    return out


def train_acc(D, CAPmat):
    return C.accuracy(D, CAPmat, D["tr_rows"])


# ==============================================================================================
# 6.  Reference baselines (no operator)
# ==============================================================================================
def baseline_plain(D):
    C.report("ex3_baseline_plain_clip", D, D["CAP"], extra={"family": "none"})


def baseline_parsed(D, verbose=True):
    """Aggarwal et al. Eq. 2, the published parsed correction — our reference point.

        e* = e_c - lam * ( <e_c, e_neg>/<e_neg, e_neg> * e_neg  -  a )

    on RAW (unnormalised) embeddings, applied only to captions where the scope parser actually
    found a negation; `a` is the neutral anchor of their Table 12.  lam is swept on tr_rows.
    """
    scopes = [C.extract_negated(c) for c in D["flat"]]
    uniq = sorted({s for s in scopes if s})
    Es = dict(zip(uniq, C.embed_texts(uniq)))            # raw, not normalised
    E = D["CAP"].copy()
    a = C.anchor()
    mask = np.array([s is not None for s in scopes])
    Corr = np.zeros_like(E)                              # the (proj - anchor) correction term
    for i, s in enumerate(scopes):
        if s:
            v = Es[s]
            Corr[i] = (E[i] @ v) / (v @ v) * v - a
    best = None
    for lam in np.arange(0.0, 2.01, 0.05):
        acc = train_acc(D, E - lam * Corr)
        if best is None or acc > best[0]:
            best = (acc, float(lam))
    lam = best[1]
    if verbose:
        print(f"[ex3] parsed-correction lam={lam:.2f} (train {100 * best[0]:.2f}%, "
              f"{int(mask.sum())}/{len(mask)} captions corrected)")
    C.report("ex3_baseline_parsed_eq2", D, E - lam * Corr,
             extra={"family": "parsed", "lam": lam, "n_parsed": int(mask.sum())})


# ==============================================================================================
# 7.  Main sweep
# ==============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grid, for a smoke run")
    ap.add_argument("--n-synth", type=int, default=9000)
    ap.add_argument("--n-aux", type=int, default=2000)
    args = ap.parse_args()

    D = C.load_mcq()
    print(f"[ex3] {D['n']} questions, {len(D['tr_rows'])} train / {len(D['te_rows'])} test rows")

    # -- 0. look at the real captions we have to mimic -------------------------------------
    rng = random.Random(0)
    print("[ex3] real captions (train half):")
    tr = [int(r) for r in D["tr_rows"]]
    for r in rng.sample(tr, 8):
        for k in range(4):
            print(f"       [{C.caption_type(D['flat'][4 * r + k])[:3]}] {D['flat'][4 * r + k]}")
        break
    for r in rng.sample(tr, 10):
        print(f"       {D['flat'][4 * r + rng.randrange(4)]}")

    baseline_plain(D)
    baseline_parsed(D)

    # -- 1. synthetic supervision -----------------------------------------------------------
    tmpls = mine_templates(D)
    rows = make_synthetic(tmpls, n=args.n_synth)
    aux = aux_pairs(args.n_aux)
    print(f"[ex3] {len(rows)} synthetic captions, {len(aux)} auxiliary negation pairs")
    for r in rows[:6]:
        print(f"       SRC {r['src']!r}  ->  pos={r['pos']} neg={r['neg']}")

    Xsrc = C.l2n(C.embed_texts([r["src"] for r in rows] + [a["src"] for a in aux]))
    E = build_target_bank(rows, aux)

    # -- 2. benchmark side ------------------------------------------------------------------
    Xn = C.l2n(D["CAP"]).astype(np.float32)
    types = np.array([C.caption_type(c) for c in D["flat"]])
    GATES = {"nonpos": types != "positive", "all": np.ones(len(types), bool)}

    # -- 3. grid ---------------------------------------------------------------------------
    if args.quick:
        BETAS, AUXW = [0.0, 0.5], [0.0]
        LAMS, LAMS_LR = [1e-2, 1e-0], [1e-2]
        RANKS, TS = [1, 8, 512], [0.5, 1.0]
        INTS = [True]
    else:
        BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
        AUXW = [0.0, 0.5]
        LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
        LAMS_LR = [1e-3, 1e-1, 1e1]
        RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 512]
        TS = [0.25, 0.5, 0.75, 1.0, 1.25]
        INTS = [False, True]

    n_main, n_aux = len(rows), len(aux)
    fitters, Pcache = {}, {}
    for aw in AUXW:
        if n_aux == 0 and aw > 0:
            continue
        w = np.concatenate([np.ones(n_main), np.full(n_aux, aw)])
        fitters[aw] = Fitter(Xsrc, w)
        for beta in BETAS:
            Pcache[(aw, beta)] = fitters[aw].prep_Y(targets_for(rows, aux, E, beta))

    def build_op(meta):
        """(Re)construct the operator described by a config dict.  Fits are ~free from the
        cached Gram matrices, so we never have to keep 500 dense 512x512 matrices around."""
        F, P = fitters[meta["aux_w"]], Pcache[(meta["aux_w"], meta["beta"])]
        if meta["family"] == "ridge":
            return F.ridge(P, meta["lam"], meta["intercept"])
        if meta["family"] == "procrustes":
            return F.procrustes(P, meta["center"])
        return F.lowrank(P, meta["lam"], [meta["rank"]], meta["intercept"])[meta["rank"]]

    def apply_meta(W, b, meta):
        Yop = C.l2n(Xn @ W.astype(np.float32) + b.astype(np.float32))
        return apply_op(Xn, Yop, GATES[meta["gate"]], meta["t"])

    results = []          # (train_acc, meta)   -- metadata only, operators are rebuilt on demand

    def evaluate(W, b, meta):
        """One matmul per operator; the gate x blend sweep on top of it is then free."""
        Yop = C.l2n(Xn @ W.astype(np.float32) + b.astype(np.float32))
        for gname, gmask in GATES.items():
            for t in TS:
                a = train_acc(D, apply_op(Xn, Yop, gmask, t))
                results.append((a, {**meta, "gate": gname, "t": float(t)}))

    for aw, F in fitters.items():
        for beta in BETAS:
            P = Pcache[(aw, beta)]
            base = {"aux_w": aw, "beta": beta}
            for it in INTS:
                for lam in LAMS:
                    W, b = F.ridge(P, lam, it)
                    evaluate(W, b, {**base, "family": "ridge", "lam": lam, "intercept": it})
                for lam in LAMS_LR:
                    for r, (W, b) in F.lowrank(P, lam, RANKS, it).items():
                        evaluate(W, b, {**base, "family": "lowrank", "lam": lam,
                                        "rank": r, "intercept": it})
            for ctr in ([False, True] if not args.quick else [True]):
                W, b = F.procrustes(P, ctr)
                evaluate(W, b, {**base, "family": "procrustes", "center": ctr})
            print(f"[ex3]   aux_w={aw} beta={beta}: {len(results)} configs", flush=True)

    # -- 4. selection: best of each family ON TRAIN, then read the test half once -----------
    print("\n[ex3] ===== selection on tr_rows =====")
    winners = {}
    for fam in ("ridge", "procrustes", "lowrank"):
        cand = sorted([x for x in results if x[1]["family"] == fam], key=lambda x: -x[0])
        winners[fam] = cand[0]
        print(f"       best {fam:11s} train {100 * cand[0][0]:.2f}%  {cand[0][1]}")

    a, meta = max(results, key=lambda x: x[0])
    print(f"       best OVERALL   train {100 * a:.2f}%  {meta}")

    for fam, (fa, fmeta) in winners.items():
        W, b = build_op(fmeta)
        C.report(f"ex3_{fam}", D, apply_meta(W, b, fmeta),
                 extra={"selected_on": "tr_rows", "train_acc": fa, **fmeta})

    # rank sensitivity of the winning low-rank family, at its own selected settings
    lr = winners["lowrank"][1]
    print("       low-rank family, train accuracy vs rank (other settings fixed):")
    for r in RANKS:
        m = {**lr, "rank": r}
        W_, b_ = build_op(m)
        print(f"         r={r:4d}  train {100 * train_acc(D, apply_meta(W_, b_, m)):.2f}%")

    W, b = build_op(meta)
    Mbest = apply_meta(W, b, meta)

    # -- 5. is W just a translation in disguise? -------------------------------------------
    d = W.shape[0]
    sv_W = np.linalg.svd(W, compute_uv=False)
    R = W - np.eye(d)
    sv_R = np.linalg.svd(R, compute_uv=False)
    energy = sv_R ** 2
    frac1 = float(energy[0] / energy.sum())
    frac8 = float(energy[:8].sum() / energy.sum())
    nrm = {"||W-I||_F": float(np.linalg.norm(R)), "||W||_F": float(np.linalg.norm(W)),
           "||b||": float(np.linalg.norm(b)),
           "eff_rank_W-I": float(energy.sum() ** 2 / (energy ** 2).sum())}
    print("\n[ex3] ===== operator geometry (overall winner) =====")
    print(f"       sv(W)   top10  {np.round(sv_W[:10], 3).tolist()}   min {sv_W[-1]:.3f}")
    print(f"       sv(W-I) top10  {np.round(sv_R[:10], 3).tolist()}")
    print(f"       residual energy in top-1 direction: {100 * frac1:.1f}%   top-8: {100 * frac8:.1f}%")
    print(f"       {nrm}")

    # explicit rank-1 control: the same operator truncated to I + (rank-1), same gate/blend
    U, S, Vt = np.linalg.svd(R)
    W1 = np.eye(d) + (U[:, :1] * S[:1]) @ Vt[:1]
    M1 = apply_meta(W1, b, meta)
    print(f"       train acc  full W {100 * a:.2f}%   I+rank1 truncation "
          f"{100 * train_acc(D, M1):.2f}%")

    C.report("ex3_best", D, Mbest, extra={
        "selected_on": "tr_rows", "train_acc": a, **meta,
        "sv_W_top10": sv_W[:10].tolist(), "sv_W_min": float(sv_W[-1]),
        "sv_W_minus_I_top10": sv_R[:10].tolist(),
        "residual_energy_top1": frac1, "residual_energy_top8": frac8,
        "rank1_truncation_train_acc": float(train_acc(D, M1)), **nrm})
    C.report("ex3_best_rank1_truncation", D, M1, extra={
        "note": "overall winner truncated to I + rank-1, same gate/blend", **meta})

    # -- 6. scoreboard ---------------------------------------------------------------------
    import json
    print("\n[ex3] ===== scoreboard (results/*.json, sorted by TEST) =====")
    board = []
    for f in sorted(C.RESULTS.glob("*.json")):
        j = json.loads(f.read_text())
        board.append((j["test"]["all"], j["train"]["all"], j["technique"], j["test"]))
    board.sort(reverse=True)
    print(f"{'technique':34s} {'train':>7s} {'TEST':>7s} {'pos':>6s} {'neg':>6s} {'hyb':>6s}")
    for te, tra, name, t in board:
        print(f"{name:34s} {100 * tra:7.2f} {100 * te:7.2f} "
              f"{100 * t['positive']:6.1f} {100 * t['negative']:6.1f} {100 * t['hybrid']:6.1f}")


if __name__ == "__main__":
    main()
