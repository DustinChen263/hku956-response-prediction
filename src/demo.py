"""Build and run an isolated synthetic software check. Never real research evidence."""
from pathlib import Path
import argparse
import copy
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from .common import ROOT, load_config, write_json


def generate_demo():
    base = ROOT / 'demo'
    raw = base / 'raw'
    audio_dir, signals = raw / 'audio', raw / 'signals'
    audio_dir.mkdir(parents=True, exist_ok=True)
    signals.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(956)
    sr, duration, nsongs = 8000, 32, 12
    t = np.arange(sr * duration) / sr
    for song in range(nsongs):
        frequency = 120 + song * 45
        y = .2 * np.sin(2 * np.pi * frequency * t) * (0.7 + .3 * np.sin(2 * np.pi * (1 + song / 6) * t))
        y += .04 * np.sin(2 * np.pi * frequency * 2.03 * t)
        sf.write(audio_dir / f's{song}.wav', y, sr)
    records = []
    for person in range(30):
        offset = rng.normal(0, 2)
        for order, song in enumerate(rng.choice(nsongs, size=6, replace=False)):
            rid = f'SYNTHETIC_p{person}_o{order}'
            interval = 800 - song * 8 + offset * 3
            values = interval + (15 + song * 2) * np.sin(np.arange(60) * .7) + rng.normal(0, 2, 60)
            times = np.cumsum(values / 1000)
            use = times <= duration
            ibi_file = signals / f'{rid}_ibi.csv'
            hr_file = signals / f'{rid}_hr.csv'
            pd.DataFrame({'time_s': times[use], 'ibi_ms': values[use]}).to_csv(ibi_file,index=False)
            pd.DataFrame({'time_s': np.arange(duration), 'hr_bpm': 60_000 / interval + rng.normal(0, .5, duration)}).to_csv(hr_file,index=False)
            records.append({'record_id': rid, 'participant_id': f'p{person:02}', 'song_id': f's{song}',
                            'audio_path': str(audio_dir / f's{song}.wav'), 'audio_start_s': 0, 'audio_end_s': duration,
                            'alignment_verified': True, 'ibi_path': str(ibi_file), 'hr_path': str(hr_file),
                            'arousal': np.clip(-5 + song * .8 + offset * .3 + rng.normal(0,.5),-10,10),
                            'valence': np.clip(3 * np.sin(song / 3) + offset * .2 + rng.normal(0,.5),-10,10),
                            'duration_s': duration, 'synthetic': True})
    pd.DataFrame(records).to_csv(raw / 'manifest.csv', index=False)
    cfg = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())
    cfg.update(synthetic=True, adapter='normalized', manifest=str(raw / 'manifest.csv'))
    for key in cfg['paths']:
        cfg['paths'][key] = str(base / key.removesuffix('_dir'))
    cfg['paths']['raw_dir'] = str(raw)
    cfg['paths']['audio_dir'] = str(audio_dir)
    cfg['physiology']['target'] = 'rmssd_ms'
    cfg['training'].update(targets=['rmssd_ms','arousal','valence'], epochs=180, patience=25,
                           hidden_sizes=[12,24], l2_values=[.001], ridge_alphas=[1,10], bootstrap_samples=200)
    config_path = ROOT / 'configs/demo.yaml'
    config_path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    (base / 'SYNTHETIC_ONLY.txt').write_text('All records, audio, labels and results in this folder are SYNTHETIC SOFTWARE TEST DATA. Not HKU956.\n')
    return config_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generate-only',action='store_true')
    p.add_argument('--cv', action='store_true')
    args = p.parse_args()
    path = generate_demo()
    print('SYNTHETIC ONLY:', path, flush=True)
    if args.generate_only:
        return
    from .audit import audit
    from .build_labels import build_labels
    from .make_splits import make_splits
    from .extract_audio import extract_audio
    from .train import train
    from .evaluate import evaluate
    from .recommend import recommend
    from .report import report
    cfg = load_config(path)
    audit(cfg)
    build_labels(cfg)
    make_splits(cfg)
    extract_audio(cfg)
    train(cfg)
    evaluate(cfg)
    if args.cv:
        train(cfg,'cv')
        evaluate(cfg,suite='cv')
    recommend(cfg,-3,3,5)
    report(cfg)

if __name__ == '__main__':
    main()
