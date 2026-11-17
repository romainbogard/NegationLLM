"""Shared harness for the NegBench COCO-MCQ steering experiments.

Every experiment in this folder imports this module so that the model, the embeddings, the
train/test split and the scoring are byte-identical across techniques — the only thing an
experiment file is allowed to change is the geometry applied to the caption embeddings.

Protocol
--------
* Benchmark: NegBench COCO-MCQ, one image and four captions, prediction = argmax cosine.
* **The released file puts the correct caption at index 0 on all 5914 rows.** `np.argmax` breaks ties
  toward the lowest index, so on the raw file any transform that collapses the four captions onto one
  vector scores 100% for free. `load_mcq` therefore SHUFFLES the four options per row with a fixed
  seed and remaps `gold`; a collapsed transform then scores 25%, which is the truth. Every report also
  carries a tie count — a variant with a non-negligible tie rate is reading the tie-break, not the image.
* Split: rows are split 50/50 with a fixed seed. **Tune on `tr_rows`, report on `te_rows`.**
  Anything selected by looking at `te_rows` is not a result.
* Embeddings are raw (unnormalised) CLIP outputs; normalise at scoring time. Caches live in
  data/cache and are keyed by model name + the exact text/image list, so they are shared.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "data" / "cache"
RESULTS = Path(__file__).resolve().parent / "results"
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

NEGBENCH = ROOT.parent / "negbench" / "COCO"
MCQ_CSV = NEGBENCH / "COCO_val_mcq_llama3.1_rephrased.csv"
IMAGE_DIR = NEGBENCH / "val2017"

MODEL_NAME = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
SEED = 0

_MODEL = None


def l2n(x, axis=-1):
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


def _load_model():
    """Lazy: an experiment that only touches cached embeddings never pays for this."""
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoProcessor, CLIPModel

        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if getattr(torch.backends, "mps", None) is not None
               and torch.backends.mps.is_available() else "cpu")
        proc = AutoProcessor.from_pretrained(MODEL_NAME)
        model = CLIPModel.from_pretrained(MODEL_NAME).to(dev).eval()
        _MODEL = (proc, model, dev, torch)
        print(f"[common] {MODEL_NAME} on {dev}", flush=True)
    return _MODEL


def _as_tensor(out):
    """transformers>=5 returns an output object from get_*_features."""
    import torch
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    return out[0]


_MEM: dict[str, np.ndarray] = {}


def embed_texts(texts) -> np.ndarray:
    """Raw text embeddings in the order given; deduplicated in memory, cached on disk."""
    texts = [str(t) for t in texts]
    todo = sorted({t for t in texts if t not in _MEM})
    if todo:
        key = hashlib.sha1(("steer|" + MODEL_NAME + "|" + "\x00".join(todo)).encode()).hexdigest()[:12]
        f = CACHE / f"steer_txt_{key}.npz"
        if f.exists():
            E = np.load(f)["e"]
        else:
            proc, model, dev, torch = _load_model()
            out, bs, t0 = [], 256, time.time()
            with torch.no_grad():
                for s in range(0, len(todo), bs):
                    inp = proc(text=todo[s:s + bs], padding=True, truncation=True,
                               max_length=77, return_tensors="pt").to(dev)
                    out.append(_as_tensor(model.get_text_features(**inp)).float().cpu().numpy())
                    d = min(s + bs, len(todo))
                    print(f"\r[common] texts {d}/{len(todo)} "
                          f"({d / max(time.time() - t0, 1e-9):.0f}/s)", end="", flush=True)
            print()
            E = np.vstack(out).astype(np.float32)
            np.savez_compressed(f, e=E)
        _MEM.update(dict(zip(todo, E)))
    return np.stack([_MEM[t] for t in texts]).astype(np.float64)


def embed_images(names) -> np.ndarray:
    names = list(names)
    key = hashlib.sha1(("steer|" + MODEL_NAME + "|" + "\x00".join(names)).encode()).hexdigest()[:12]
    f = CACHE / f"steer_img_{key}.npz"
    if f.exists():
        return np.load(f)["e"].astype(np.float64)
    from PIL import Image
    proc, model, dev, torch = _load_model()
    out, bs, t0 = [], 64, time.time()
    with torch.no_grad():
        for s in range(0, len(names), bs):
            ims = [Image.open(IMAGE_DIR / n).convert("RGB") for n in names[s:s + bs]]
            inp = proc(images=ims, return_tensors="pt").to(dev)
            out.append(_as_tensor(model.get_image_features(**inp)).float().cpu().numpy())
            d = min(s + bs, len(names))
            print(f"\r[common] images {d}/{len(names)} "
                  f"({d / max(time.time() - t0, 1e-9):.0f}/s)", end="", flush=True)
    print()
    E = np.vstack(out).astype(np.float32)
    np.savez_compressed(f, e=E)
    return E.astype(np.float64)


# --------------------------------------------------------------------------------------------
# negation scope extraction — Aggarwal et al., appendix A.1 (13/13 on their Table 9)
# --------------------------------------------------------------------------------------------
PRE_NEGATORS = ["does not", "doesn't", "do not", "don't", "did not", "didn't",
                "has not", "hasn't", "have not", "haven't", "cannot", "can't",
                "absence of", "devoid of", "empty of", "free of",
                "lacks", "lack", "without", "misses", "no one", "nothing", "nowhere",
                "none", "neither", "nor", "never", "no", "not"]
POST_NEGATORS = [f"{c} {n}" for c in ("is", "are", "was", "were")
                 for n in ("nowhere", "not", "absent", "never", "lacking", "missing", "neither", "nor")]
BOUNDARY_WORDS = {"but", "although", "though", "however", "nevertheless", "yet", "while", "whereas"}
BOUNDARY_PUNCT = {".", ",", ";"}
_TOK = re.compile(r"[a-z']+|[.,;]")


def _is_boundary(t):
    return t in BOUNDARY_PUNCT or t in BOUNDARY_WORDS


def extract_negated(caption: str):
    """C^neg: the words inside the first negation scope, or None."""
    toks = _TOK.findall(str(caption).lower())
    best = None
    for i in range(len(toks)):
        for kind, table in (("post", POST_NEGATORS), ("pre", PRE_NEGATORS)):
            for phrase in table:
                w = phrase.split()
                if toks[i:i + len(w)] == w and (best is None or len(w) > best[1]):
                    best = (i, len(w), kind)
            if best and best[0] == i:
                break
        if best:
            break
    if best is None:
        return None
    i, n, kind = best
    if kind == "pre":
        out = []
        for t in toks[i + n:]:
            if _is_boundary(t):
                break
            out.append(t)
    else:
        out = []
        for t in reversed(toks[:i]):
            if _is_boundary(t):
                break
            out.append(t)
        out.reverse()
    return " ".join(out) or None


NEG_CUE = re.compile(
    r"\b(no|not|n't|without|absent|absence|lacks?|lacking|missing|neither|nor|none|"
    r"devoid|excluded|free of|nowhere|isn't|aren't|doesn't|don't)\b", re.I)
CONTRAST = re.compile(r"\b(but|while|however|whereas|although|though|yet)\b|;", re.I)


def caption_type(text: str) -> str:
    """positive = asserts only presence | negative = only absence | hybrid = both."""
    t = str(text)
    if not NEG_CUE.search(t):
        return "positive"
    return "hybrid" if CONTRAST.search(t) else "negative"


ANCHOR_WORDS = ["neutral", "balanced", "unbiased", "fair"]     # their Table 12


def anchor() -> np.ndarray:
    return embed_texts(ANCHOR_WORDS).mean(0)


# --------------------------------------------------------------------------------------------
# the benchmark
# --------------------------------------------------------------------------------------------
CAPS = [f"caption_{i}" for i in range(4)]


def load_mcq(subsample=None, seed=SEED):
    """Everything an experiment needs, with the split already made."""
    df = pd.read_csv(MCQ_CSV)
    if subsample:
        df = df.sample(n=min(subsample, len(df)), random_state=seed)
    df = df.reset_index(drop=True)

    names = sorted({Path(p).name for p in df.image_path})
    IMG = embed_images(names)
    pos = {n: i for i, n in enumerate(names)}
    img_idx = np.array([pos[Path(p).name] for p in df.image_path])

    flat = [str(df.loc[i, c]) for i in range(len(df)) for c in CAPS]

    # --- de-bias the option order (see the module docstring) ---------------------------------
    rng = np.random.default_rng(seed)
    gold_raw = df["correct_answer"].values
    order = np.stack([rng.permutation(4) for _ in range(len(df))])      # new slot j <- old option
    flat = [flat[4 * i + order[i, j]] for i in range(len(df)) for j in range(4)]
    gold = np.array([int(np.where(order[i] == gold_raw[i])[0][0]) for i in range(len(df))])

    CAP = embed_texts(flat)

    perm = rng.permutation(len(df))
    te_rows, tr_rows = perm[:len(perm) // 2], perm[len(perm) // 2:]

    return {"df": df, "flat": flat, "CAP": CAP, "IMG": IMG, "img_idx": img_idx,
            "gold": gold, "order": order, "tmpl": df["correct_answer_template"].values,
            "tr_rows": tr_rows, "te_rows": te_rows, "n": len(df)}


def scores(D, CAPmat, IMGmat=None):
    """(n, 4) cosine scores. CAPmat is (4n, d); it is normalised here."""
    I = l2n(D["IMG"] if IMGmat is None else IMGmat)[D["img_idx"]]
    return np.einsum("ij,ij->i", np.repeat(I, 4, axis=0), l2n(CAPmat)).reshape(-1, 4)


def accuracy(D, CAPmat, rows=None, IMGmat=None, S=None):
    S = scores(D, CAPmat, IMGmat) if S is None else S
    rows = np.arange(len(S)) if rows is None else rows
    return float((S[rows].argmax(1) == D["gold"][rows]).mean())


def tie_rate(D, CAPmat=None, rows=None, IMGmat=None, S=None, tol=1e-12):
    """Share of questions whose top two options are numerically indistinguishable.

    A transform that collapses captions wins those on the argmax tie-break rather than on the image,
    so a variant with a visible tie rate is not measuring what it claims to measure.
    """
    S = scores(D, CAPmat, IMGmat) if S is None else S
    rows = np.arange(len(S)) if rows is None else rows
    P = np.sort(S[rows], axis=1)
    return float(((P[:, -1] - P[:, -2]) < tol).mean())


def per_template(D, CAPmat, rows, IMGmat=None):
    S = scores(D, CAPmat, IMGmat)
    out = {"all": accuracy(D, None, rows, S=S)}
    for t in ("positive", "negative", "hybrid"):
        sel = rows[D["tmpl"][rows] == t]
        out[t] = accuracy(D, None, sel, S=S) if len(sel) else float("nan")
    return out


def save_result(name: str, payload: dict):
    """One json per technique, so the notebook can pick the winners up later."""
    payload = {"technique": name, "model": MODEL_NAME, **payload}
    f = RESULTS / f"{name}.json"
    f.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[common] wrote {f}")
    return f


def report(name, D, CAPmat, IMGmat=None, extra=None):
    """Standard scoreboard line: tuned nowhere, read on the held-out half."""
    tr = per_template(D, CAPmat, D["tr_rows"], IMGmat)
    te = per_template(D, CAPmat, D["te_rows"], IMGmat)
    S = scores(D, CAPmat, IMGmat)
    ties = tie_rate(D, S=S, rows=D["te_rows"])
    flag = "   *** TIE-INFLATED, not a result ***" if ties > 0.005 else ""
    print(f"[{name}]  train {100 * tr['all']:.2f}%   TEST {100 * te['all']:.2f}%   "
          f"(pos {100 * te['positive']:.1f} / neg {100 * te['negative']:.1f} / "
          f"hyb {100 * te['hybrid']:.1f})   ties {100 * ties:.2f}%{flag}")
    return save_result(name, {"train": tr, "test": te, "tie_rate_test": ties, **(extra or {})})
