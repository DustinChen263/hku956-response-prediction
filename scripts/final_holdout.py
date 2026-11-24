"""Locked one-time holdout evaluation of the existing whole-track models.

No fitting, refitting, model selection or threshold search is performed here.
Historical signed training configurations remain untouched; this separate
protocol supersedes their previous evaluation-deferred status.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common import load_config,digest,write_json
from src.experiment import run_signature
from src.evaluate import evaluate,summarize,paired_bootstrap
from scripts.song_level_mlp import average_seed_predictions

REPORT=ROOT/'reports/final_holdout_v1'
OUTPUT=ROOT/'results/final_holdout_v1'
CONFIGS=['configs/song_level_v1.yaml','configs/song_level_mlp_v1.yaml']


def freeze():
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=REPORT/'protocol.json'
    if lock.exists():
        verify_lock();print('Existing final-evaluation protocol verified.');return
    split=json.loads((ROOT/'data/splits/participants.json').read_text())
    manifest=pd.read_csv(ROOT/'data/processed/song_level_v1/manifest.csv')
    cohort=manifest[(manifest.partition=='test')&manifest.main_eligible_before_features]
    assert len(cohort)==111 and set(cohort.participant_id)==set(split['groups']['test'])
    paths=[Path(__file__),ROOT/'scripts/song_level_mlp.py',ROOT/'data/splits/participants.json',
           ROOT/'data/processed/song_level_v1/manifest.csv',ROOT/'data/processed/song_level_v1/labels.csv',
           ROOT/'features/song_level_v1/audio_features.npz',ROOT/'features/song_level_v1/feature_metadata.json',
           *sorted((ROOT/'src').glob('*.py'))]
    inventories=[]
    for config in CONFIGS:
        cfg=load_config(ROOT/config);out=Path(cfg['paths']['results_dir'])
        assert not (out/'evaluation_test.json').exists(),'Holdout already opened before this freeze'
        reg=json.loads((out/'registry.json').read_text())
        assert reg['signature']==run_signature(cfg,reg['definition'])
        paths.extend([ROOT/config,out/'registry.json',out/'protocol.json'])
        for model in reg['models']:
            p=out/model['file'];assert digest(p)==model['sha256'];paths.append(p)
            inventories.append({'experiment':out.name,'model':model['kind'],'seed':model['seed'],
                                 'targets':model['targets'],'file':str(p.relative_to(ROOT)),
                                 'sha256':model['sha256'],'params':model['params']})
    assert len(inventories)==15
    write_json(lock,{'version':'final_holdout_v1_2026-10-28','status':'frozen_before_final_numeric_outcome_access',
        'authorization':'User explicitly requested next step after final-evaluation proposal',
        'historical_training_config_override':'Evaluation previously deferred; now authorized by this separate protocol. Signed training configs unchanged.',
        'public_preregistration':False,'development_results_already_inspected':True,
        'no_refit':True,'no_additional_training':True,'no_model_or_threshold_search':True,
        'input':'whole-track 37-dimensional acoustic features, not verified actual listening exposure',
        'primary_targets':['mean_hr','arousal','valence'],'hrv_required':False,
        'cohort_record_ids':cohort.record_id.tolist(),'test_participants':split['groups']['test'],
        'cohort_rule':'Same audio and three-primary-label-complete cohort as prior baselines, no post-test exclusions',
        'models':inventories,'neural_seed_rule':'Per-record equal prediction average of seeds 17,42,93; individual seeds also reported',
        'primary_metric':'participant macro MAE separately per target',
        'secondary_metrics':['record MAE','RMSE','R2','Pearson','individual participant MAE'],
        'primary_comparison':'multi_mlp minus single_mlp per target',
        'secondary_comparisons':['ridge vs mean','multi_mlp vs mean','av_mlp vs multi_mlp on AV','matched_multi_mlp vs single_mlp; candidate-budget matching limitation retained'],
        'uncertainty':'2000 participant-cluster bootstrap draws, fixed seed 17, descriptive only; four participants; no significance claims',
        'winner_policy':'Report all predeclared families; do not select or tune based on holdout scores',
        'test_access':'One evaluation per signed model registry; subsequent identical invocations verify and reuse cached predictions',
        'files_sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths}})
    print('Final evaluation frozen: 111 records, four people, 15 model files; no refit.')


def verify_lock():
    lock=json.loads((REPORT/'protocol.json').read_text())
    for path,h in lock['files_sha256'].items():
        if digest(ROOT/path)!=h:raise ValueError(f'Frozen artifact changed: {path}')
    return lock


def run():
    lock=verify_lock();OUTPUT.mkdir(parents=True,exist_ok=True)
    predictions=[]
    for path in CONFIGS:
        cfg=load_config(ROOT/path)
        p=evaluate(cfg,'test','main')
        predictions.append(p)
    allseed=pd.concat(predictions,ignore_index=True)
    assert set(allseed.record_id)==set(lock['cohort_record_ids'])
    assert set(allseed.participant_id)==set(lock['test_participants'])
    assert not allseed.duplicated(['model','seed','record_id','target']).any()
    for (model,seed,target),sub in allseed.groupby(['model','seed','target']):
        assert set(sub.record_id)==set(lock['cohort_record_ids'])
    ensemble=average_seed_predictions(allseed)
    metrics,people=summarize(ensemble)
    seedmetrics,_=summarize(allseed)
    bootstrap=paired_bootstrap(ensemble,2000,17)
    allseed.to_csv(OUTPUT/'all_seed_predictions.csv',index=False)
    ensemble.to_csv(OUTPUT/'ensemble_predictions.csv',index=False)
    metrics.to_csv(OUTPUT/'ensemble_metrics.csv',index=False)
    people.to_csv(OUTPUT/'participant_metrics.csv',index=False)
    seedmetrics.to_csv(OUTPUT/'individual_seed_metrics.csv',index=False)
    bootstrap.to_csv(OUTPUT/'paired_bootstrap.csv',index=False)
    deltas=[]
    for t in ['mean_hr','arousal','valence']:
        a=people[(people.model=='multi_mlp')&(people.target==t)].set_index('participant_id')
        b=people[(people.model=='single_mlp')&(people.target==t)].set_index('participant_id')
        for pid in lock['test_participants']:
            deltas.append({'participant_id':pid,'target':t,'records':int(a.loc[pid,'n']),
                'single_mae':b.loc[pid,'mae'],'multi_mae':a.loc[pid,'mae'],
                'delta_multi_minus_single':a.loc[pid,'mae']-b.loc[pid,'mae']})
    pd.DataFrame(deltas).to_csv(OUTPUT/'primary_comparison_participants.csv',index=False)
    write_json(REPORT/'completion.json',{'final_holdout_evaluated':True,'records':ensemble.record_id.nunique(),
        'participants':ensemble.participant_id.nunique(),'metric_rows':len(metrics),
        'refit':False,'additional_training':False,'post_test_tuning':False,
        'protocol_sha256':digest(REPORT/'protocol.json'),
        'predictions_sha256':digest(OUTPUT/'ensemble_predictions.csv')})
    print(metrics[['model','target','n','participants','macro_mae','rmse']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['freeze','evaluate']);a=p.parse_args()
    (freeze if a.stage=='freeze' else run)()
