"""
face_sampler.py — Indexes the curated FairFace dataset and samples face image
paths by demographic filters (race, gender, age_group).
"""

import csv
import random
from pathlib import Path

_FAIRFACE_DIR = Path(__file__).parent.parent / "data" / "fairface"

_AGE_GROUP_MAP = {
    "0-2":          "child",
    "3-9":          "child",
    "10-19":        "child",
    "20-29":        "adult",
    "30-39":        "adult",
    "40-49":        "adult",
    "50-59":        "adult",
    "60-69":        "elderly",
    "more than 70": "elderly",
}


class FaceSampler:
    """
    Indexes the FairFace label CSV and samples face image paths by demographic
    attributes. Searches both val/ and train/ image folders (train/ supplements
    elderly images underrepresented in val/).
    """

    def __init__(self, fairface_dir=None, csv_name="curated_fairface_val.csv"):
        fairface_dir = Path(fairface_dir or _FAIRFACE_DIR)
        csv_path     = fairface_dir / csv_name
        img_dirs     = [fairface_dir / "val", fairface_dir / "train"]

        if not csv_path.exists():
            raise FileNotFoundError(f"FairFace label CSV not found at {csv_path}")
        if not any(d.exists() for d in img_dirs):
            raise FileNotFoundError(f"No image folder found under {fairface_dir} (expected val/ or train/)")

        self._index = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                age_group = _AGE_GROUP_MAP.get(row["age"])
                if age_group is None:
                    continue
                filename = Path(row["file"]).name
                abs_path = next(
                    (str(d / filename) for d in img_dirs if (d / filename).exists()),
                    None,
                )
                if abs_path is None:
                    continue
                self._index.append({
                    "path":      abs_path,
                    "gender":    row["gender"],
                    "race":      row["race"],
                    "age_group": age_group,
                })

        if not self._index:
            raise RuntimeError("FairFace index is empty — check that val/ images exist and match the CSV.")

    def sample(self, n, gender=None, race=None, age_group=None, rng=None, exclude=None):
        """
        Return n face image paths matching the given demographic filters,
        sampled without replacement. exclude prevents the same face appearing
        in both groups of a scenario.
        """
        pool = self._index
        if gender:    pool = [r for r in pool if r["gender"]    == gender]
        if race:      pool = [r for r in pool if r["race"]      == race]
        if age_group: pool = [r for r in pool if r["age_group"] == age_group]
        if exclude:
            exclude_set = set(exclude)
            pool = [r for r in pool if r["path"] not in exclude_set]

        if len(pool) < n:
            raise ValueError(
                f"Not enough faces matching filters — found {len(pool)}, need {n}. "
                f"(gender={gender}, race={race}, age_group={age_group})"
            )
        chosen = (rng or random).sample(pool, n)
        return [r["path"] for r in chosen]
