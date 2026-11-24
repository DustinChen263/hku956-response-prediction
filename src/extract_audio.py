"""Deterministic, content-addressed features for verified listening segments."""
import json
import os
from pathlib import Path
import numpy as np
from .common import parser, load_config, folder, read_manifest, digest, fingerprint, write_json


def acoustic_features(y, sr, n_mfcc=13):
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    arrays = [mfcc, librosa.feature.spectral_centroid(y=y, sr=sr),
              librosa.feature.spectral_bandwidth(y=y, sr=sr),
              librosa.feature.spectral_rolloff(y=y, sr=sr),
              librosa.feature.zero_crossing_rate(y), librosa.feature.rms(y=y)]
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = float(np.asarray(librosa.feature.tempo(onset_envelope=onset, sr=sr)).flat[0])
    return np.concatenate([v for arr in arrays for v in (arr.mean(axis=1), arr.std(axis=1))] + [np.array([tempo])]).astype(np.float32)


class MERTEncoder:
    def __init__(self, config):
        if not config.get('mert_revision') or len(str(config['mert_revision'])) != 40:
            raise ValueError('MERT requires a pinned 40-character commit in audio.mert_revision; no implicit main download')
        try:
            import torch
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
        except ImportError as exc:
            raise RuntimeError('MERT optional dependencies missing: install requirements-mert.txt') from exc
        self.torch, self.config = torch, config
        kwargs = dict(revision=config['mert_revision'], trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(config['mert_model'], **kwargs)
        self.model = AutoModel.from_pretrained(config['mert_model'], **kwargs).eval()
        self.model.requires_grad_(False)
        self.sr = self.processor.sampling_rate

    def encode(self, y, sr):
        import librosa
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        length = int(self.config['chunk_s'] * self.sr)
        embeddings, weights = [], []
        for start in range(0, len(y), length):
            chunk = y[start:start + length]
            if len(chunk) < self.sr // 10:
                continue
            data = self.processor(chunk, sampling_rate=self.sr, return_tensors='pt')
            with self.torch.inference_mode():
                out = self.model(**data, output_hidden_states=True)
            state = out.hidden_states[self.config['mert_layer']][0]
            embeddings.append(state.mean(dim=0).cpu().numpy())
            weights.append(len(chunk))
        if not embeddings:
            raise ValueError('Audio too short for encoder')
        return np.average(embeddings, axis=0, weights=weights).astype(np.float32)


def segment_features(path, start, duration, config, cache_dir, encoder=None):
    os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(cache_dir) / 'numba'))
    import librosa
    import soundfile as sf
    info = sf.info(path)
    if start < 0 or duration <= 0 or start + duration > info.duration + 0.05:
        raise ValueError(f'Segment [{start}, {start + duration}] exceeds audio duration {info.duration}')
    provenance = {'audio_sha256': digest(Path(path)), 'start_s': float(start), 'duration_s': float(duration),
                  'config': config, 'implementation': 'features-v1'}
    key = fingerprint(provenance)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f'{key}.npy'
    if cached.exists():
        return np.load(cached, allow_pickle=False), provenance
    y, sr = librosa.load(path, sr=config['sample_rate'], mono=True, offset=start, duration=duration)
    if len(y) < duration * sr - sr * 0.05:
        raise ValueError('Decoded audio shorter than requested segment')
    if config['backend'] == 'acoustic':
        feat = acoustic_features(y, sr, config['n_mfcc'])
    elif config['backend'] == 'mert':
        if encoder is None:
            raise ValueError('MERT encoder not initialized')
        feat = encoder.encode(y, sr)
    else:
        raise ValueError('Unknown audio backend; no silent fallback')
    if not np.isfinite(feat).all():
        raise ValueError('Nonfinite audio features')
    np.save(cached, feat, allow_pickle=False)
    write_json(cached.with_suffix('.json'), provenance)
    return feat, provenance


def extract_audio(cfg, dry_run=False):
    df = read_manifest(cfg)
    out = folder(cfg, 'features')
    rows, ids, failures, provenance = [], [], [], []
    config = cfg['audio']
    encoder = None
    for r in df.to_dict('records'):
        try:
            path = str(r.get('audio_path', ''))
            if not path or path == 'nan' or not Path(path).is_file():
                raise ValueError('missing_audio')
            if str(r.get('alignment_verified', '')).lower() not in ('true', '1'):
                raise ValueError('unverified_audio_alignment')
            if not np.isfinite(float(r['audio_start_s'])):
                raise ValueError('missing_audio_start_s')
            # Audio summarizes the full actual exposure; physiology can have a fixed subwindow.
            if dry_run:
                ids.append(r['record_id'])
                continue
            if config['backend'] == 'mert' and encoder is None:
                encoder = MERTEncoder(config)
            feat, meta = segment_features(path, float(r['audio_start_s']), float(r['duration_s']), config, out / 'cache', encoder)
            ids.append(r['record_id'])
            rows.append(feat)
            provenance.append({'record_id': r['record_id'], **meta})
        except (ValueError, OSError, RuntimeError) as exc:
            failures.append({'record_id': r['record_id'], 'reason': str(exc)})
    import pandas as pd
    pd.DataFrame(failures, columns=['record_id', 'reason']).to_csv(out / 'feature_failures.csv', index=False)
    write_json(out / 'extraction_status.json', {'synthetic': bool(cfg.get('synthetic')), 'dry_run': dry_run,
                                               'eligible': len(ids), 'failed': len(failures), 'backend': config['backend']})
    if dry_run:
        print(f'Dry run: {len(ids)} eligible, {len(failures)} unavailable')
        return
    if not rows:
        raise RuntimeError(f'No verified usable audio. See {out / "feature_failures.csv"}; existing features were not overwritten.')
    np.savez_compressed(out / 'audio_features.npz', record_ids=np.asarray(ids, dtype=str), X=np.stack(rows))
    write_json(out / 'feature_metadata.json', {'synthetic': bool(cfg.get('synthetic')), 'config': config,
                'records': provenance, 'dimension': len(rows[0]), 'array_sha256': digest(out / 'audio_features.npz')})
    print(f'Audio features: {len(rows)} records x {len(rows[0])} dimensions; {len(failures)} failures')


def main():
    p = parser(__doc__)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    extract_audio(load_config(args.config), args.dry_run)

if __name__ == '__main__':
    main()
