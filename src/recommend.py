"""Goal-distance ranking of verified audio segments; no intervention claims."""
import json
import pickle
import numpy as np
import pandas as pd
from .common import parser, load_config, folder, digest
from .experiment import load_design, run_signature


def rank_candidates(predictions, targets, goals, scales, weights=None):
    weights = weights or {k: 1.0 for k in goals}
    score = np.zeros(len(predictions))
    for target, goal in goals.items():
        if target not in targets:
            raise ValueError(f'Model cannot predict {target}')
        j = targets.index(target)
        if not np.isfinite(goal) or scales[j] <= 0 or weights.get(target, 0) < 0:
            raise ValueError('Invalid goal, scale or weight')
        score -= weights.get(target, 0) * np.abs(predictions[:, j] - goal) / scales[j]
    return score


def recommend(cfg, arousal, valence, top_k=10, physiological_target=None):
    out = folder(cfg, 'results')
    registry = json.loads((out / 'registry.json').read_text())
    if registry['signature'] != run_signature(cfg, registry['definition']):
        raise ValueError('Model no longer matches data/config/protocol')
    if not -10 <= arousal <= 10 or not -10 <= valence <= 10 or top_k < 1:
        raise ValueError('A/V goals must be [-10,10]; top_k must be positive')
    # Select by development validation only. Do not inspect any test outcomes.
    eligible = [r for r in registry['models'] if r['kind'] not in ('song_mean', 'av_mlp') and set(registry['targets']).issubset(r['targets'])]
    selected = min(eligible, key=lambda r: (r['validation_score'], r['kind'], r['seed']))
    if digest(out / selected['file']) != selected['sha256']:
        raise ValueError('Model artifact checksum mismatch')
    with (out / selected['file']).open('rb') as f:
        model = pickle.load(f)
    frame, X = load_design(cfg)
    # Distinct exposures of the same recording may be different candidates.
    keys = ['song_id', 'audio_path', 'audio_start_s', 'duration_s']
    keep = ~frame.duplicated(keys)
    candidates, x = frame.loc[keep, keys].reset_index(drop=True), X[keep]
    pred = model.predict(x)
    goals = {'arousal': arousal, 'valence': valence}
    if physiological_target is not None:
        target = cfg['physiology']['target']
        goals[target] = physiological_target
    score = rank_candidates(pred, model.targets, goals, model.scaler.raw_scale_)
    for j, name in enumerate(model.targets):
        candidates['predicted_' + name] = pred[:, j]
    candidates['score'] = score
    candidates['model'] = selected['kind']
    candidates['synthetic'] = registry['synthetic']
    ranked = candidates.sort_values('score', ascending=False).head(top_k)
    ranked.to_csv(out / 'recommendations.csv', index=False)
    print('模型预测，尚未验证推荐干预效果。音频模型不区分用户；生理分量默认关闭。')
    print(ranked.drop(columns=['audio_path']).to_string(index=False))
    return ranked


def main():
    p = parser(__doc__)
    p.add_argument('--target-arousal', type=float, required=True)
    p.add_argument('--target-valence', type=float, required=True)
    p.add_argument('--physiological-target', type=float)
    p.add_argument('--top-k', type=int, default=10)
    args = p.parse_args()
    recommend(load_config(args.config), args.target_arousal, args.target_valence, args.top_k, args.physiological_target)

if __name__ == '__main__':
    main()
