"""Participant macro metrics and paired cluster bootstrap, never row-wise inference."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .common import parser, load_config, folder, write_json, digest
from .experiment import load_design, split_data, run_signature


def compute_metrics(y, pred):
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = np.asarray(y)[valid], np.asarray(pred)[valid]
    n = len(y)
    if not n:
        return {'n': 0, 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'pearson': np.nan}
    variance = np.sum((y - y.mean()) ** 2)
    return {'n': n, 'mae': float(np.mean(np.abs(y - pred))), 'rmse': float(np.sqrt(np.mean((y - pred) ** 2))),
            'r2': float(1 - np.sum((y - pred) ** 2) / variance) if n > 1 and variance > 1e-12 else np.nan,
            'pearson': float(np.corrcoef(y, pred)[0,1]) if n > 1 and np.std(y) > 1e-12 and np.std(pred) > 1e-12 else np.nan}


def summarize(predictions):
    metrics, people = [], []
    keys = ['model', 'seed', 'target', 'split']
    for key, df in predictions.groupby(keys):
        base = dict(zip(keys, key))
        if 'synthetic' in df:
            base['synthetic'] = bool(df.synthetic.astype(str).str.lower().isin(['true','1']).all())
        observed = df[np.isfinite(df.actual) & np.isfinite(df.predicted)]
        person_mae = []
        for pid, sub in observed.groupby('participant_id'):
            stat = compute_metrics(sub.actual.to_numpy(), sub.predicted.to_numpy())
            people.append({**base, 'participant_id': pid, **stat})
            person_mae.append(stat['mae'])
        stat = compute_metrics(df.actual.to_numpy(), df.predicted.to_numpy())
        stat.update(macro_mae=float(np.mean(person_mae)) if person_mae else np.nan, participants=len(person_mae))
        if key[2] == 'rmssd_ms' and len(observed):
            stat['log1p_mae'] = float(np.mean(np.abs(np.log1p(observed.actual) - np.log1p(np.maximum(observed.predicted, 0)))))
        metrics.append({**base, **stat})
    return pd.DataFrame(metrics), pd.DataFrame(people)


def paired_bootstrap(predictions, repeats=2000, seed=17):
    rng = np.random.default_rng(seed)
    output = []
    pairs = [('multi_mlp', 'single_mlp'), ('ridge', 'mean'), ('multi_mlp', 'mean'), ('av_mlp', 'multi_mlp'), ('matched_multi_mlp', 'single_mlp')]
    for left, right in pairs:
        for target in predictions.target.unique():
            a = predictions[(predictions.model == left) & (predictions.target == target)]
            b = predictions[(predictions.model == right) & (predictions.target == target)]
            if a.empty or b.empty:
                continue
            common_seeds = set(a.seed) & set(b.seed)
            # Deterministic baseline is paired with each neural seed; never pool seeds as participants.
            seed_pairs = [(s, s) for s in sorted(common_seeds)]
            if not seed_pairs and b.seed.nunique() == 1:
                seed_pairs = [(s, int(b.seed.iloc[0])) for s in sorted(a.seed.unique())]
            for sa, sb in seed_pairs:
                merged = a[a.seed == sa].merge(b[b.seed == sb], on=['record_id','participant_id','target','split'], suffixes=('_a','_b'), validate='one_to_one')
                merged = merged[np.isfinite(merged.actual_a) & np.isfinite(merged.actual_b)]
                if merged.empty:
                    continue
                if not np.allclose(merged.actual_a, merged.actual_b):
                    raise ValueError('Paired labels disagree')
                merged['delta'] = np.abs(merged.actual_a - merged.predicted_a) - np.abs(merged.actual_b - merged.predicted_b)
                deltas = merged.groupby('participant_id').delta.mean().to_numpy()
                samples = rng.choice(deltas, size=(repeats, len(deltas)), replace=True).mean(axis=1)
                output.append({'model_a': left, 'model_b': right, 'seed_a': sa, 'seed_b': sb, 'target': target,
                               'macro_mae_delta_a_minus_b': float(deltas.mean()), 'ci_low': float(np.quantile(samples,.025)),
                               'ci_high': float(np.quantile(samples,.975)), 'participants': len(deltas),
                               'paired_records': len(merged), 'synthetic': bool(predictions.get('synthetic', pd.Series(False)).astype(str).str.lower().isin(['true','1']).all()), 'warning': 'few clusters; descriptive uncertainty only' if len(deltas) < 10 else ''})
    return pd.DataFrame(output)


def evaluate_partition(cfg, output, partition='test'):
    output = Path(output)
    registry_path = output / 'registry.json'
    registry = json.loads(registry_path.read_text())
    if registry['signature'] != run_signature(cfg, registry['definition']):
        raise ValueError('Protocol/data/source changed since training; use a new experiment directory')
    done = output / f'evaluation_{partition}.json'
    if done.exists():
        meta = json.loads(done.read_text())
        if meta['signature'] == registry['signature']:
            if digest(output / f'predictions_{partition}.csv') != meta['predictions_sha256']:
                raise ValueError('Cached predictions checksum mismatch')
            print(f'Cached {partition} evaluation: {output}')
            return pd.read_csv(output / f'predictions_{partition}.csv')
        raise ValueError('Evaluation lock mismatch')
    frame, X = load_design(cfg)
    sub, x, y = split_data(cfg, frame, X, registry['definition'], partition, registry['targets'])
    rows = []
    for entry in registry['models']:
        # Only load local artifacts produced by this project. Pickle is not a model exchange format.
        if digest(output / entry['file']) != entry['sha256']:
            raise ValueError('Model artifact checksum mismatch')
        with (output / entry['file']).open('rb') as f:
            model = pickle.load(f)
        pred = model.predict(x, sub.song_id.to_numpy())
        for j,t in enumerate(entry['targets']):
            actual = y[:, registry['targets'].index(t)]
            for i, row in sub.iterrows():
                rows.append({'record_id': row.record_id, 'participant_id': row.participant_id, 'song_id': row.song_id,
                             'duration_s': row.duration_s, 'model': entry['kind'], 'seed': entry['seed'], 'target': t,
                             'actual': actual[i], 'predicted': pred[i,j], 'split': partition, 'synthetic': registry['synthetic']})
    predictions = pd.DataFrame(rows)
    metrics, people = summarize(predictions)
    predictions.to_csv(output / f'predictions_{partition}.csv', index=False)
    metrics.to_csv(output / f'metrics_{partition}.csv', index=False)
    people.to_csv(output / f'participant_metrics_{partition}.csv', index=False)
    bootstrap = paired_bootstrap(predictions, cfg['training']['bootstrap_samples'])
    bootstrap.to_csv(output / f'paired_bootstrap_{partition}.csv', index=False)
    write_json(done, {'signature': registry['signature'], 'prediction_rows': len(predictions), 'participants': sub.participant_id.nunique(),
                     'synthetic': registry['synthetic'], 'predictions_sha256': digest(output / f'predictions_{partition}.csv')})
    print(f'Evaluated {partition}: {len(sub)} listening records; {len(metrics)} metric rows')
    return predictions


def evaluate(cfg, partition='test', suite='main'):
    out = folder(cfg, 'results')
    if suite == 'main':
        return evaluate_partition(cfg, out, partition)
    folds = sorted(out.glob('cv_*/registry.json'))
    if not folds:
        raise ValueError('No trained CV folds')
    metrics = []
    for registry in folds:
        evaluate_partition(cfg, registry.parent, 'test')
        table = pd.read_csv(registry.parent / 'metrics_test.csv')
        table['fold'] = registry.parent.name
        metrics.append(table)
    pd.concat(metrics).to_csv(out / 'cv_metrics.csv', index=False)


def main():
    p = parser(__doc__)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--suite', choices=['main','cv'], default='main')
    args = p.parse_args()
    evaluate(load_config(args.config), args.split, args.suite)

if __name__ == '__main__':
    main()
