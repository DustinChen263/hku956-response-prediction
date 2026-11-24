"""Tune only on explicit participant-disjoint validation data."""
import copy
import itertools
import json
import pickle
from pathlib import Path
import numpy as np
from .common import parser, load_config, folder, write_json, environment, digest
from .experiment import load_design, split_data, run_signature
from .models import ResponseModel, validation_score


def candidates(kind, config, n_features=None, n_targets=None):
    common = {k: config[k] for k in ('learning_rate', 'epochs', 'patience', 'min_delta', 'pca_components')}
    if kind == 'ridge':
        return [{**common, 'alpha': a} for a in config['ridge_alphas']]
    if kind in ('single_mlp', 'multi_mlp', 'av_mlp', 'matched_multi_mlp'):
        config = dict(config)
        if kind == 'matched_multi_mlp':
            # Match total trainable parameters to all independent single-task networks.
            d, tasks = n_features, n_targets
            config['hidden_sizes'] = [max(1, round(tasks * h * (d + 2) / (d + tasks + 1))) for h in config['hidden_sizes']]
        grid = [{**common, 'hidden': h, 'l2': l2} for h,l2 in itertools.product(config['hidden_sizes'], config['l2_values'])]
        if len(grid) > 12:
            raise ValueError('Maximum 12 MLP configurations per family')
        return grid
    return [common]


def train_partition(cfg, definition, output):
    frame, X = load_design(cfg)
    tcfg = cfg['training']
    targets = tcfg['targets']
    train, xt, yt = split_data(cfg, frame, X, definition, 'train', targets)
    val, xv, yv = split_data(cfg, frame, X, definition, 'validation', targets)
    # Test samples and their numeric labels never enter model fitting or selection.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = run_signature(cfg, definition)
    registry_path = output / 'registry.json'
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        if registry['signature'] == signature:
            print(f'Cached locked training run: {output}')
            return registry
        raise ValueError('Different existing run: choose a new results_dir; protocol/artifacts are immutable')
    write_json(output / 'protocol.json', {'signature': signature, 'config': cfg, 'definition': definition,
                'environment': environment(), 'train_records': train.record_id.tolist(), 'validation_records': val.record_id.tolist(),
                'synthetic': bool(cfg.get('synthetic'))})
    registry = {'signature': signature, 'synthetic': bool(cfg.get('synthetic')), 'definition': definition,
                'models': [], 'search': [], 'targets': targets, 'cohort': tcfg['cohort'], 'backend': cfg['audio']['backend']}
    for kind in tcfg['models']:
        indices = [j for j,t in enumerate(targets) if t in ('arousal', 'valence')] if kind == 'av_mlp' else list(range(len(targets)))
        if not indices:
            continue
        names = [targets[j] for j in indices]
        ytrain, yval = yt[:, indices], yv[:, indices]
        best = None
        for params in candidates(kind, tcfg, min(xt.shape[1], int(tcfg['pca_components']), len(xt)-1) if tcfg.get('pca_components') else xt.shape[1], len(names)):
            model = ResponseModel(kind, names, params, seed=tcfg['seeds'][0])
            model.fit(xt, ytrain, xv, yval, val.participant_id.to_numpy(), train.song_id.to_numpy())
            raw = model.predict(xv, val.song_id.to_numpy())
            score = validation_score(model.scaler.transform(raw), model.scaler.transform(yval), val.participant_id.to_numpy())
            registry['search'].append({'kind': kind, 'params': params, 'seed': tcfg['seeds'][0], 'score': score})
            if best is None or score < best[0]:
                best = (score, params, model)
        seeds = tcfg['seeds'] if 'mlp' in kind else [tcfg['seeds'][0]]
        for seed in seeds:
            score, params, model = best
            if seed != tcfg['seeds'][0]:
                model = ResponseModel(kind, names, params, seed)
                model.fit(xt, ytrain, xv, yval, val.participant_id.to_numpy(), train.song_id.to_numpy())
                raw = model.predict(xv, val.song_id.to_numpy())
                score = validation_score(model.scaler.transform(raw), model.scaler.transform(yval), val.participant_id.to_numpy())
            filename = f'{kind}_seed{seed}.pkl'
            with (output / filename).open('wb') as f:
                pickle.dump(model, f)
            registry['models'].append({'kind': kind, 'seed': seed, 'file': filename, 'targets': names,
                        'params': params, 'validation_score': score, 'parameter_count': model.parameter_count_,
                        'sha256': digest(output / filename),
                        'constant_training_targets': model.scaler.constant_targets_,
                        'best_epochs': [m.best_epoch_ for m in model.models if hasattr(m, 'best_epoch_')]})
        print(f'{kind}: validation standardized macro MAE {best[0]:.4f}, {len(seeds)} seed(s)')
    registry['fit_count'] = len(registry['search']) + sum(max(0, len(tcfg['seeds']) - 1) for k in tcfg['models'] if 'mlp' in k)
    write_json(registry_path, registry)
    return registry


def train(cfg, suite='main'):
    split = json.loads((folder(cfg, 'splits') / 'participants.json').read_text())
    out = folder(cfg, 'results')
    if suite == 'main':
        return train_partition(cfg, split, out)
    if not split['cv']:
        raise ValueError('No CV definitions; song-disjoint stress test uses a separate protocol')
    for fold in split['cv']:
        definition = {'groups': fold['groups'], 'fold': fold['fold'], 'song_disjoint': False}
        if set(definition['groups']['test']) & set(split['groups']['test']):
            raise ValueError('Final holdout entered CV')
        train_partition(cfg, definition, out / f"cv_{fold['fold']}")


def main():
    p = parser(__doc__)
    p.add_argument('--suite', choices=['main', 'cv'], default='main')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        frame, X = load_design(cfg)
        definition = json.loads((folder(cfg, 'splits') / 'participants.json').read_text())
        for partition in ('train', 'validation'):
            f, x, y = split_data(cfg, frame, X, definition, partition, cfg['training']['targets'])
            print(partition, len(f), x.shape, y.shape)
    else:
        train(cfg, args.suite)

if __name__ == '__main__':
    main()
