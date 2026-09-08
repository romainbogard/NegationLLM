"""Manip 0, run category by category on the controlled dataset.

Pooling all negation types together confounds two very different questions:

  * "is there ONE negation direction?"            -> pooled rank
  * "does EACH negation type have its own?"       -> per-category rank

A pooled rank of ~18 is the *expected* signature of H2 (one sub-axis per type), and is
indistinguishable from H3/H4 unless the categories are also measured separately. So every
statistic below is computed within category first, then pooled for comparison.

Usage:  ../.venv/bin/python run_per_category.py [--layer 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, CLIPModel

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"
N_NULL = 200
NULL_Q = 0.95
SEED = 0

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------- embeddings
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _as_tensor(out):
    """transformers <5 returns a tensor; >=5 returns an output object."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    return out[0]


@torch.no_grad()
def embed(texts, model, tokenizer, device, layer=None, batch_size=256) -> np.ndarray:
    key = hashlib.sha1(
        ("\x00".join(texts) + f"|{MODEL_NAME}|{layer}").encode()
    ).hexdigest()[:12]
    path = CACHE / f"emb_{key}.npz"
    if path.exists():
        return np.load(path)["e"]

    out = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                        max_length=77, return_tensors="pt").to(device)
        if layer is None:
            feats = _as_tensor(model.get_text_features(input_ids=enc["input_ids"],
                                                       attention_mask=enc["attention_mask"]))
        else:
            res = model.text_model(input_ids=enc["input_ids"],
                                   attention_mask=enc["attention_mask"], output_hidden_states=True)
            hs = res.hidden_states[layer]
            eos = tokenizer.eos_token_id
            hit = enc["input_ids"] == eos
            pos = torch.where(hit.any(-1), hit.int().argmax(-1), enc["input_ids"].argmax(-1))
            feats = hs[torch.arange(hs.shape[0], device=hs.device), pos]
        out.append(feats.float().cpu().numpy())
        print(f"\r  embedding {min(start + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    E = np.vstack(out).astype(np.float32)
    np.savez_compressed(path, e=E)
    return E


def l2n(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


# --------------------------------------------------------------------------- estimators
def spectrum(X: np.ndarray) -> np.ndarray:
    """Uncentered second-moment eigenvalues, descending.

    Uncentered on purpose: the mean of the deltas *is* the translation vector of H1, so
    centering would delete the H1 signal before measuring it.
    """
    s = np.linalg.svd(X, compute_uv=False)
    return (s ** 2) / len(X)


def shares(ev: np.ndarray) -> np.ndarray:
    ev = np.clip(ev, 0, None)
    t = ev.sum()
    return ev / t if t > 0 else ev


def shannon_rank(p: np.ndarray) -> float:
    q = p[p > 1e-15]
    return float(np.exp(-(q * np.log(q)).sum()))


def participation_ratio(ev: np.ndarray) -> float:
    ev = np.clip(ev, 0, None)
    d = (ev ** 2).sum()
    return float(ev.sum() ** 2 / d) if d > 0 else 0.0


def translation_share(D: np.ndarray) -> float:
    """rho = ||mean delta||^2 / E||delta||^2. 1 under a pure translation, ~0 under H4."""
    mu = D.mean(axis=0)
    return float(mu @ mu / np.mean(np.sum(D ** 2, axis=1)))


def significant_rank(D: np.ndarray, A: np.ndarray, n_null=N_NULL, q=NULL_Q, seed=SEED) -> int:
    """Leading components whose variance SHARE exceeds the per-index null quantile.

    The null is built from differences between two *affirmative* sentences of the same
    category: same vocabulary, same templates, no negation. Comparing shares rather than raw
    eigenvalues makes the test scale-free, which matters because an affirmative-affirmative
    difference is larger in norm than a minimal negation edit.
    """
    n = len(D)
    if n < 8 or len(A) < 8:
        return -1
    p_real = shares(spectrum(D))
    g = np.random.default_rng(seed)
    k = min(n, len(A), D.shape[1])
    null = np.empty((n_null, k))
    for r in range(n_null):
        i = g.integers(0, len(A), n)
        j = (i + g.integers(1, len(A), n)) % len(A)      # j != i
        null[r] = shares(spectrum(A[i] - A[j]))[:k]
    thr = np.quantile(null, q, axis=0)
    above = p_real[:k] > thr
    return int(np.argmin(above)) if not above.all() else int(k)


# --------------------------------------------------------------------------- run
def analyse(D: np.ndarray, A: np.ndarray) -> dict:
    ev = spectrum(D)
    p = shares(ev)
    return {
        "n": len(D),
        "rho": translation_share(D),
        "top1_share": float(p[0]),
        "top5_share": float(p[:5].sum()),
        "shannon_rank": shannon_rank(p),
        "participation_ratio": participation_ratio(ev),
        "sig_rank": significant_rank(D, A),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None,
                    help="text-encoder hidden layer; default = final projected joint space")
    ap.add_argument("--csv", default=str(DATA / "negation_dataset.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    aff_col = "affirmative"
    neg_col = "negated" if "negated" in df.columns else "negative"
    print(f"{len(df)} pairs | {df['category'].nunique()} categories | layer={args.layer}\n")

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"device = {device}")

    E_aff = l2n(embed(df[aff_col].tolist(), model, tokenizer, device, args.layer))
    E_neg = l2n(embed(df[neg_col].tolist(), model, tokenizer, device, args.layer))
    DELTA = E_neg - E_aff
    print(f"delta: {DELTA.shape}\n")

    cats = df["category"].values
    rows = []
    for c in sorted(set(cats)):
        m = cats == c
        r = analyse(DELTA[m], E_aff[m])
        r["category"] = c
        r["superclass"] = df.loc[m, "superclass"].iloc[0] if "superclass" in df else ""
        rows.append(r)

    pooled = analyse(DELTA, E_aff)
    pooled["category"] = "== POOLED (all 18) =="
    pooled["superclass"] = ""

    per_cat = pd.DataFrame(rows).sort_values("rho", ascending=False)
    table = pd.concat([per_cat, pd.DataFrame([pooled])], ignore_index=True)
    cols = ["category", "superclass", "n", "rho", "top1_share", "top5_share",
            "shannon_rank", "participation_ratio", "sig_rank"]

    pd.set_option("display.width", 200, "display.max_colwidth", 40)
    print("=" * 118)
    print("PER-CATEGORY RANK  (rho = translation share; sig_rank vs within-category affirmative null)")
    print("=" * 118)
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- H2 test: is each category a distinct direction? -------------------------------
    names = per_cat["category"].tolist()
    C = np.vstack([l2n(DELTA[cats == c].mean(axis=0, keepdims=True))[0] for c in names])
    M = C @ C.T
    off = M[~np.eye(len(names), dtype=bool)]
    intra = np.array([float(np.mean(l2n(DELTA[cats == c]) @ l2n(DELTA[cats == c].mean(axis=0, keepdims=True))[0]))
                      for c in names])

    print("\n" + "=" * 118)
    print("CATEGORY CENTROID COSINE MATRIX (H2 wants: high diagonal-ish intra, low off-diagonal)")
    print("=" * 118)
    short = [n[:22] for n in names]
    print(pd.DataFrame(M, index=short, columns=[f"{i:>5}" for i in range(len(names))])
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print("\ncolumn index = row order above")
    print(f"\nmean intra-category coherence   = {intra.mean():.3f}  (delta vs its own centroid)")
    print(f"mean inter-category cosine      = {off.mean():.3f}  (centroid vs centroid)")
    print(f"max  inter-category cosine      = {off.max():.3f}")
    print(f"separation (intra - inter)      = {intra.mean() - off.mean():.3f}")

    out = {
        "model": MODEL_NAME, "layer": args.layer, "csv": Path(args.csv).name,
        "per_category": table[cols].to_dict("records"),
        "intra_coherence": float(intra.mean()),
        "inter_cosine_mean": float(off.mean()),
        "inter_cosine_max": float(off.max()),
        "centroid_cosine_matrix": M.tolist(),
        "category_order": names,
    }
    tag = f"layer{args.layer}" if args.layer is not None else "final"
    dest = CACHE / f"per_category_{tag}.json"
    dest.write_text(json.dumps(out, indent=2))
    table[cols].to_csv(CACHE / f"per_category_{tag}.csv", index=False)
    print(f"\nSaved -> {dest.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
