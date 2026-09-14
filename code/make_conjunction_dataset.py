"""Word bank + pair generator for the conjunction-arithmetic experiment (notebook 04).

The question notebook 04 asks is whether a compound noun phrase decomposes arithmetically:

    e("a cat and a dog")  ==  e("a cat") + e("a dog") + a

with a bias vector `a` that is the *same* for every pair. Testing that statistically needs many
pairs, and the pairs have to be controlled along two axes the bias could plausibly depend on:

* **semantic distance** — `same_cat` marks pairs drawn from one category ("cat"/"dog") against
  pairs drawn from two ("cat"/"piano"). If `a` is constant, it must not care.
* **word identity** — every word appears in several pairs, so the per-word contribution can be
  averaged out instead of being confounded with the pair.

Concrete countable nouns only: mass nouns ("water") would need a different determiner and the
determiner is part of what gets embedded, so mixing them in would add a nuisance factor.

Run as a script to (re)write the csv:  python make_conjunction_dataset.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "conjunction_pairs.csv"
OUT_CSV3 = Path(__file__).resolve().parent.parent / "data" / "conjunction_triples.csv"

# 12 categories x 20 nouns. Categories are coarse but disjoint, which is all `same_cat` needs.
WORDS: dict[str, list[str]] = {
    "animal": ["cat", "dog", "horse", "rabbit", "elephant", "tiger", "bear", "sheep", "cow",
               "monkey", "fox", "wolf", "deer", "goat", "camel", "zebra", "lion", "mouse",
               "donkey", "squirrel"],
    "bird_fish": ["bird", "eagle", "owl", "parrot", "penguin", "duck", "chicken", "pigeon",
                  "swan", "crow", "fish", "shark", "dolphin", "whale", "crab", "turtle",
                  "frog", "snake", "lizard", "butterfly"],
    "food": ["apple", "banana", "orange", "lemon", "tomato", "potato", "carrot", "onion",
             "pizza", "sandwich", "cake", "cookie", "egg", "burger", "salad", "pear",
             "peach", "grape", "melon", "pancake"],
    "furniture": ["chair", "table", "sofa", "bed", "desk", "shelf", "cabinet", "mirror",
                  "lamp", "carpet", "curtain", "bench", "stool", "wardrobe", "drawer",
                  "pillow", "blanket", "clock", "vase", "painting"],
    "vehicle": ["car", "bus", "truck", "bicycle", "motorcycle", "train", "airplane", "boat",
                "ship", "helicopter", "tractor", "van", "taxi", "scooter", "canoe", "ferry",
                "ambulance", "jeep", "sled", "wagon"],
    "clothing": ["shirt", "jacket", "coat", "dress", "skirt", "hat", "scarf", "glove", "shoe",
                 "boot", "sock", "sweater", "belt", "tie", "helmet", "sandal", "raincoat",
                 "uniform", "apron", "hoodie"],
    "tool": ["hammer", "screwdriver", "wrench", "saw", "drill", "shovel", "rake", "ladder",
             "brush", "knife", "axe", "chisel", "pliers", "broom", "bucket", "rope",
             "nail", "ruler", "clamp", "trowel"],
    "nature": ["tree", "flower", "rock", "river", "mountain", "cloud", "beach", "forest",
               "lake", "island", "waterfall", "cave", "desert", "hill", "valley", "meadow",
               "glacier", "volcano", "canyon", "pond"],
    "building": ["house", "church", "castle", "bridge", "tower", "barn", "stadium", "hotel",
                 "school", "hospital", "factory", "lighthouse", "windmill", "cottage",
                 "temple", "museum", "library", "station", "garage", "shed"],
    "instrument": ["guitar", "piano", "violin", "drum", "flute", "trumpet", "harp", "banjo",
                   "cello", "saxophone", "accordion", "clarinet", "tambourine", "keyboard",
                   "harmonica", "ukulele", "trombone", "xylophone", "bagpipe", "organ"],
    "container": ["box", "bottle", "cup", "basket", "jar", "bag", "suitcase", "barrel",
                  "envelope", "wallet", "backpack", "crate", "kettle", "teapot", "mug",
                  "pot", "bowl", "plate", "tray", "can"],
    "electronics": ["phone", "laptop", "camera", "television", "radio", "printer", "monitor",
                    "speaker", "microphone", "tablet", "router", "projector", "console",
                    "headphone", "calculator", "scanner", "modem", "fan", "toaster", "microwave"],
}

# "a" vs "an" is decided by the *sound*, not the letter: "a uniform", "an hour".
AN_EXCEPTIONS = {"hour", "honest", "umbrella"}
A_EXCEPTIONS = {"uniform", "unicorn", "university", "european", "one", "ukulele"}


def article(word: str) -> str:
    if word in A_EXCEPTIONS:
        return "a"
    if word in AN_EXCEPTIONS or word[0] in "aeiou":
        return "an"
    return "a"


def phrase(word: str) -> str:
    return f"{article(word)} {word}"


def build_dataset(n_same: int = 700, n_cross: int = 700, seed: int = 0) -> pd.DataFrame:
    """Unordered word pairs: `n_same` within one category, `n_cross` across two.

    Sampling is done by shuffling the full enumeration rather than by drawing with replacement,
    so pairs are unique and the word-occurrence counts stay near-uniform.
    """
    import random

    rng = random.Random(seed)
    cats = sorted(WORDS)

    same = [(a, b, c, c) for c in cats for a, b in itertools.combinations(WORDS[c], 2)]
    cross = [(a, b, ca, cb)
             for ca, cb in itertools.combinations(cats, 2)
             for a in WORDS[ca] for b in WORDS[cb]]
    rng.shuffle(same)
    rng.shuffle(cross)

    rows = []
    for a, b, ca, cb in same[:n_same] + cross[:n_cross]:
        if rng.random() < 0.5:            # kill the alphabetical ordering the enumeration imposes
            a, b, ca, cb = b, a, cb, ca
        rows.append({"word_a": a, "word_b": b, "phrase_a": phrase(a), "phrase_b": phrase(b),
                     "cat_a": ca, "cat_b": cb, "same_cat": ca == cb})

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "pair_id", [f"p{i:05d}" for i in range(len(df))])
    return df


def build_triples(n: int = 700, seed: int = 1) -> pd.DataFrame:
    """Word triples for the A, B and C section: does a second connective cost a second bias?

    Half the triples are drawn inside one category and half across three, mirroring `build_dataset`
    so the same-category control carries over from pairs to triples.
    """
    import random

    rng = random.Random(seed)
    cats = sorted(WORDS)
    rows, seen = [], set()
    while len(rows) < n:
        if len(rows) < n // 2:
            c = rng.choice(cats)
            trio = rng.sample(WORDS[c], 3)
            cs = [c, c, c]
        else:
            cs = rng.sample(cats, 3)
            trio = [rng.choice(WORDS[c]) for c in cs]
        if len(set(trio)) < 3 or tuple(trio) in seen:
            continue
        seen.add(tuple(trio))
        rows.append({"word_a": trio[0], "word_b": trio[1], "word_c": trio[2],
                     "phrase_a": phrase(trio[0]), "phrase_b": phrase(trio[1]),
                     "phrase_c": phrase(trio[2]),
                     "cat_a": cs[0], "cat_b": cs[1], "cat_c": cs[2],
                     "same_cat": len(set(cs)) == 1})

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "triple_id", [f"t{i:05d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    df = build_dataset()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    n_words = len({w for c in WORDS.values() for w in c})
    print(f"wrote {len(df)} pairs ({df.same_cat.sum()} same-category) over {n_words} words "
          f"in {len(WORDS)} categories -> {OUT_CSV}")
    print(df.head().to_string(index=False))

    df3 = build_triples()
    df3.to_csv(OUT_CSV3, index=False)
    print(f"\nwrote {len(df3)} triples ({df3.same_cat.sum()} same-category) -> {OUT_CSV3}")
    print(df3.head(3).to_string(index=False))
