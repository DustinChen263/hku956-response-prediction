"""Data boundaries: only explicitly selected IDs get numeric labels parsed."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import folder, read_manifest, assert_splits, digest, fingerprint


def load_design(cfg):
    manifest = read_manifest(cfg)
    feature_path = folder(cfg, 'features') / 'audio_features.npz'
    if not feature_path.exists():
        raise FileNotFoundError('No audio_features.npz. Extract verified audio before training; do not train on song IDs.')
    meta = json.loads((feature_path.parent / 'feature_metadata.json').read_text())
    if meta['array_sha256'] != digest(feature_path):
        raise ValueError('Feature array checksum mismatch')
    if bool(meta['synthetic']) != bool(cfg.get('synthetic')) or meta['config'] != cfg['audio']:
        raise ValueError('Feature provenance/config does not match this experiment')
    with np.load(feature_path, allow_pickle=False) as data:
        ids, X = data['record_ids'].astype(str), data['X'].astype(float)
    if len(ids) != len(set(ids)) or len(X) != len(ids) or not np.isfinite(X).all():
        raise ValueError('Invalid feature array')
    frame = manifest.set_index('record_id').loc[ids].reset_index()
    provenance = {r['record_id']: r for r in meta['records']}
    hashes = {}
    for row in frame.itertuples():
        record = provenance.get(row.record_id)
        if record is None or not np.isclose(float(row.audio_start_s), record['start_s']) or not np.isclose(float(row.duration_s), record['duration_s']):
            raise ValueError('Cached segment no longer matches manifest; re-extract audio')
        path = str(row.audio_path)
        if path not in hashes:
            hashes[path] = digest(Path(path))
        if hashes[path] != record['audio_sha256']:
            raise ValueError('Audio content changed; re-extract features')
    # No label values are returned through the design matrix interface.
    frame = frame.drop(columns=[c for c in ('arousal', 'valence') if c in frame])
    return frame, X


def load_labels(cfg, ids, targets):
    requested = set(ids)
    mapping = {}
    with (folder(cfg, 'processed') / 'labels.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['record_id'] not in requested:
                continue
            if row['record_id'] in mapping:
                raise ValueError('Duplicate label record')
            mapping[row['record_id']] = [float(row.get(t) or 'nan') for t in targets]
    missing = requested - set(mapping)
    if missing:
        raise ValueError(f'Missing label rows: {sorted(missing)[:3]}')
    y = np.array([mapping[i] for i in ids], dtype=float)
    if np.isinf(y).any():
        raise ValueError('Infinite labels')
    return y


def split_data(cfg, frame, X, definition, partition, targets):
    groups = definition['groups']
    assert_splits(groups)
    keep = frame.participant_id.isin(groups[partition]).to_numpy()
    if definition.get('song_disjoint'):
        keep &= frame.song_id.isin(definition['songs'][partition]).to_numpy()
    sub, x = frame.loc[keep].reset_index(drop=True), X[keep]
    y = load_labels(cfg, sub.record_id.tolist(), targets)
    if cfg['training']['cohort'] == 'complete':
        use = np.isfinite(y).all(axis=1)
    elif cfg['training']['cohort'] == 'masked':
        use = np.isfinite(y).any(axis=1)
    else:
        raise ValueError('cohort must be complete or masked')
    sub, x, y = sub.loc[use].reset_index(drop=True), x[use], y[use]
    if not len(sub):
        raise ValueError(f'No usable {partition} samples with configured cohort and targets')
    if set(sub.participant_id) != set(groups[partition]):
        absent = set(groups[partition]) - set(sub.participant_id)
        raise ValueError(f'Participants have no eligible {partition} records: {sorted(absent)}. Audit cohort explicitly.')
    return sub, x, y


def run_signature(cfg, definition):
    paths = [folder(cfg, 'processed') / 'manifest.csv', folder(cfg, 'processed') / 'labels.csv',
             folder(cfg, 'features') / 'audio_features.npz']
    source = Path(__file__).parent
    code = {p.name: digest(p) for p in sorted(source.glob('*.py'))}
    return fingerprint({'config': {k: v for k,v in cfg.items() if not k.startswith('_')},
                        'definition': definition, 'data': {str(p): digest(p) for p in paths}, 'code': code})
