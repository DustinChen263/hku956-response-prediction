from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import yaml
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('rmssd_ms', 'mean_hr', 'arousal', 'valence')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n', encoding='utf8')


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf8'))
    # All relative paths are relative to the project, not the caller's cwd.
    base = ROOT
    cfg['_config_path'] = str(path)
    for key in ('raw_dir', 'audio_dir', 'processed_dir', 'splits_dir', 'features_dir', 'results_dir', 'reports_dir'):
        p = Path(cfg['paths'][key]).expanduser()
        cfg['paths'][key] = str(p if p.is_absolute() else base / p)
    if cfg.get('manifest'):
        p = Path(cfg['manifest'])
        cfg['manifest'] = str(p if p.is_absolute() else base / p)
    return cfg


def folder(cfg, name):
    p = Path(cfg['paths'][name + '_dir'])
    p.mkdir(parents=True, exist_ok=True)
    return p


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', default=str(ROOT / 'configs/default.yaml'))
    return p


def read_manifest(cfg):
    p = folder(cfg, 'processed') / 'manifest.csv'
    return pd.read_csv(p, dtype={'record_id': str, 'participant_id': str, 'song_id': str})


def validate_manifest(df):
    required = ['record_id', 'participant_id', 'song_id', 'audio_path', 'duration_s', 'arousal', 'valence']
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f'Missing manifest columns: {sorted(missing)}')
    if df[required[:3]].isna().any().any() or df.record_id.duplicated().any():
        raise ValueError('Identifiers must be non-null; record_id must be unique')
    durations = pd.to_numeric(df.duration_s, errors='coerce')
    if (durations.isna() | ~np.isfinite(durations) | (durations <= 0)).any():
        raise ValueError('Every record needs a finite positive duration_s')
    for target in ('arousal', 'valence'):
        vals = pd.to_numeric(df[target], errors='raise')
        if ((vals.dropna() < -10) | (vals.dropna() > 10)).any():
            raise ValueError(f'{target} outside documented [-10, 10] scale')


def environment():
    import importlib.metadata
    versions = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'PyYAML', 'librosa', 'soundfile'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions}


def dataset_signature(cfg):
    base = folder(cfg, 'processed')
    return {name: digest(base / name) for name in ('manifest.csv', 'labels.csv')}


def assert_splits(groups):
    sets = [set(groups[k]) for k in ('train', 'validation', 'test')]
    if any(not s for s in sets):
        raise ValueError('Every participant partition must be nonempty')
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Participant leakage between partitions')
