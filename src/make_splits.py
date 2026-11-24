import json
import numpy as np
from .common import parser, load_config, folder, read_manifest, write_json, assert_splits, fingerprint


def make_splits(cfg):
    df = read_manifest(cfg)
    scfg = cfg['split']
    ids = np.array(sorted(df.participant_id.unique()))
    counts = scfg['counts']
    if sum(counts) != len(ids) or min(counts) < 1:
        raise ValueError(f'Configured counts {counts} do not match {len(ids)} participants; explicitly revise config')
    rng = np.random.default_rng(scfg['seed'])
    rng.shuffle(ids)
    a, b = counts[0], counts[0] + counts[1]
    groups = dict(train=ids[:a].tolist(), validation=ids[a:b].tolist(), test=ids[b:].tolist())
    assert_splits(groups)
    output = {'groups': groups, 'seed': scfg['seed'], 'song_disjoint': scfg.get('song_disjoint', False),
              'participant_ids_hash': fingerprint(sorted(df.participant_id.unique()))}
    if output['song_disjoint']:
        songs = np.array(sorted(df.song_id.unique()))
        rng.shuffle(songs)
        cuts = [round(len(songs) * counts[0] / len(ids)), round(len(songs) * b / len(ids))]
        output['songs'] = {k: v.tolist() for k, v in zip(groups, np.split(songs, cuts))}
    sets = {k: set(df.loc[df.participant_id.isin(v), 'song_id']) for k, v in groups.items()}
    output['song_overlap_before_filtering'] = {f'{x}/{y}': len(sets[x] & sets[y]) for x,y in [('train','validation'),('train','test'),('validation','test')]}
    # Cross-validation is confined to development participants, with separate inner validation groups.
    dev = np.concatenate([ids[:a], ids[a:b]])
    rng.shuffle(dev)
    folds = np.array_split(dev, scfg['cv_folds'])
    if min(map(len, folds)) < 1:
        raise ValueError('Too many CV folds')
    output['cv'] = []
    for index, outer in enumerate(folds):
        rest = [x for x in dev if x not in set(outer)]
        nval = max(1, round(len(rest) * 0.15))
        inner = {'train': rest[nval:], 'validation': rest[:nval], 'test': outer.tolist()}
        assert_splits(inner)
        output['cv'].append({'fold': index, 'groups': inner})
    if output['song_disjoint']:
        # Keep optional song stress-test separate from participant-only CV.
        output['cv'] = []
    path = folder(cfg, 'splits') / 'participants.json'
    if path.exists() and json.loads(path.read_text()) != output:
        raise ValueError('Refusing to overwrite an existing different split. Use a separate experiment directory.')
    write_json(path, output)
    print('Participant split:', {k: len(v) for k,v in groups.items()}, 'song overlap:', output['song_overlap_before_filtering'])
    return output


def main():
    args = parser('Create immutable participant-level partitions').parse_args()
    make_splits(load_config(args.config))

if __name__ == '__main__':
    main()
