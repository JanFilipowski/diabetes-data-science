from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json, hashlib, joblib, numpy as np, pandas as pd

ART_ROOT = Path("artifacts")

def run_id_from_cfg(cfg, tag: str | None = None) -> str:
    """Stabilny identyfikator eksperymentu bazujący na konfiguracji (i opcjonalnym tagu)."""
    blob = json.dumps(asdict(cfg), sort_keys=True)
    h = hashlib.md5(blob.encode()).hexdigest()[:8]
    return f"{h}{('-' + tag) if tag else ''}"

def artdir(run_id: str) -> Path:
    p = ART_ROOT / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path: Path, obj): path.write_text(json.dumps(obj, indent=2))
def load_json(path: Path): return json.loads(path.read_text())

def save_npz(path: Path, **arrays):  # np.savez_compressed wrapper
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def save_parquet(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def dump_joblib(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path: Path):
    return joblib.load(path)
