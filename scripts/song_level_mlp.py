"""Fixed-budget whole-track MLP comparisons, development outcomes only."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common import load_config,folder,write_json,digest,environment
from src.experiment import load_design,split_data
from src.models import ResponseModel
from src.train import candidates,train
from src.evaluate import evaluate,summarize,paired_bootstrap

CONFIG=ROOT/'configs/song_level_mlp_v1.yaml'
FAMILIES=['single_mlp','multi_mlp','matched_multi_mlp','av_mlp']


def prepare():
    base=ROOT/'configs/song_level_v1.yaml'
    cfg=yaml.safe_load(base.read_text())
    cfg['paths'].update(results_dir='results/song_level_mlp_v1',reports_dir='reports/song_level_mlp_v1')
    cfg['training']['models']=FAMILIES
    cfg['research_scope']['comparison']='MLP families on the locked whole-track baseline cohort; no final holdout evaluation'
    if CONFIG.exists() and yaml.safe_load(CONFIG.read_text())!=cfg:raise ValueError('Existing MLP configuration differs')
    CONFIG.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    loaded=load_config(CONFIG);rep=folder(loaded,'reports')
    files=[CONFIG,Path(__file__),base,ROOT/'data/splits/participants.json',
        ROOT/'data/processed/song_level_v1/manifest.csv',ROOT/'data/processed/song_level_v1/labels.csv',
        ROOT/'features/song_level_v1/audio_features.npz',ROOT/'features/song_level_v1/feature_metadata.json',
        *sorted((ROOT/'src').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    lock=rep/'protocol.json'
    if lock.exists():
        if json.loads(lock.read_text())['files_sha256']!=hashes:raise ValueError('Locked MLP protocol changed')
        return
    write_json(lock,{'status':'locked_before_probe_and_formal_MLP_fits','files_sha256':hashes,
        'config':loaded,'environment':environment(),'baseline_results_already_inspected':True,
        'public_preregistration':False,'final_holdout_evaluation':False,
        'main_comparison':'multi_mlp vs single_mlp; participant macro MAE per target',
        'secondary_comparisons':['matched_multi_mlp vs single_mlp','av_mlp vs multi_mlp on A/V','MLPs vs mean and ridge'],
        'seed_aggregation':'average predictions across seeds first; cluster bootstrap participants; also report individual seeds',
        'single_task_search':'one common hidden/l2 configuration across three independent task networks; same 4-candidate family budget',
        'parameter_matching':'matched shared widths from 3*h*(37+2)/(37+3+1), rounded; report actual selected parameter counts',
        'runtime_probe':'first predeclared multi_mlp candidate, seed 17, train/validation only; not used to change search space'})
    print('MLP protocol locked; four candidates per family, three seeds, original participant folds.',flush=True)


def probe():
    cfg=load_config(CONFIG);frame,X=load_design(cfg)
    split=json.loads((folder(cfg,'splits')/'participants.json').read_text())
    tr,xt,yt=split_data(cfg,frame,X,split,'train',cfg['training']['targets'])
    va,xv,yv=split_data(cfg,frame,X,split,'validation',cfg['training']['targets'])
    params=candidates('multi_mlp',cfg['training'],X.shape[1],3)[0]
    start=time.perf_counter()
    model=ResponseModel('multi_mlp',cfg['training']['targets'],params,17)
    model.fit(xt,yt,xv,yv,va.participant_id.to_numpy(),tr.song_id.to_numpy())
    elapsed=time.perf_counter()-start
    write_json(folder(cfg,'reports')/'runtime_probe.json',{'fit_wall_s':elapsed,
        'epochs_run':len(model.models[0].history_),'max_epochs':cfg['training']['epochs'],
        'formal_response_model_fits':144,'formal_network_fits':216,
        'upper_epoch_linear_fit_estimate_s':elapsed*500/len(model.models[0].history_)*216,
        'note':'estimate excludes hashing/evaluation and larger matched widths; no validation score used to alter protocol'})
    print(f'Probe fit {elapsed:.3f}s, {len(model.models[0].history_)} epochs; rough 500-epoch network budget {elapsed*500/len(model.models[0].history_)*216:.1f}s plus I/O',flush=True)


def average_seed_predictions(frame):
    keys=['record_id','participant_id','song_id','target','split']
    rows=[]
    for (model,*identity),group in frame.groupby(['model',*keys],sort=False):
        if group.seed.duplicated().any():raise ValueError('Duplicate model seed record')
        if not np.allclose(group.actual,group.actual.iloc[0]):raise ValueError('Seeds have different ground truth')
        expected=3 if model in FAMILIES else 1
        if len(group)!=expected:raise ValueError('Incomplete seed set')
        rows.append({'model':model,**dict(zip(keys,identity)),'seed':0,'actual':group.actual.iloc[0],
            'predicted':group.predicted.mean(),'synthetic':False,'seed_count':len(group)})
    return pd.DataFrame(rows)


def combine():
    cfg=load_config(CONFIG);out=folder(cfg,'results');rep=folder(cfg,'reports')
    baseline=ROOT/'results/song_level_v1'
    split=json.loads((folder(cfg,'splits')/'participants.json').read_text())
    final=set(split['groups']['test'])
    cv=[]
    for p in sorted(out.glob('cv_*/predictions_test.csv')):
        f=pd.read_csv(p,dtype={'song_id':str});f['fold']=p.parent.name;f['split']='development_cv'
        assert not set(f.participant_id)&final
        cv.append(f)
    neural=pd.concat(cv,ignore_index=True)
    base=pd.read_csv(baseline/'development_cv_predictions.csv',dtype={'song_id':str})
    combined=pd.concat([base,neural],ignore_index=True)
    validation=pd.concat([pd.read_csv(baseline/'predictions_validation.csv',dtype={'song_id':str}),
                          pd.read_csv(out/'predictions_validation.csv',dtype={'song_id':str})],ignore_index=True)
    for name,predictions in [('development_cv',combined),('validation',validation)]:
        assert not set(predictions.participant_id)&final
        assert not predictions.duplicated(['record_id','model','seed','target']).any()
        # Compare exactly the baseline record IDs per target (A/V head has no HR).
        for target,subset in predictions.groupby('target'):
            ids=set(subset.loc[subset.model=='mean','record_id'])
            for _,part in subset.groupby(['model','seed']):assert set(part.record_id)==ids
        ensemble=average_seed_predictions(predictions)
        metrics,people=summarize(ensemble)
        seed_metrics,_=summarize(predictions)
        predictions.to_csv(out/f'{name}_all_seed_predictions.csv',index=False)
        ensemble.to_csv(out/f'{name}_ensemble_predictions.csv',index=False)
        metrics.to_csv(out/f'{name}_ensemble_metrics.csv',index=False)
        people.to_csv(out/f'{name}_ensemble_participant_metrics.csv',index=False)
        seed_metrics.to_csv(out/f'{name}_individual_seed_metrics.csv',index=False)
        paired_bootstrap(ensemble,2000).to_csv(out/f'{name}_ensemble_paired_bootstrap.csv',index=False)
        if name=='development_cv':print(metrics[['model','target','n','participants','macro_mae']].to_string(index=False),flush=True)
    settings=[]
    for run in [out,*[p.parent for p in sorted(out.glob('cv_*/registry.json'))]]:
        reg=json.loads((run/'registry.json').read_text())
        for m in reg['models']:
            settings.append({'run':run.name,'model':m['kind'],'seed':m['seed'],
                'hidden':m['params']['hidden'],'l2':m['params']['l2'],'parameters':m['parameter_count'],
                'best_epochs':json.dumps(m['best_epochs']),'validation_score':m['validation_score']})
    pd.DataFrame(settings).to_csv(rep/'selected_models.csv',index=False)
    assert not (out/'evaluation_test.json').exists()
    assert not (baseline/'evaluation_test.json').exists()


def run():
    cfg=load_config(CONFIG);start=time.perf_counter()
    train(cfg,'main');evaluate(cfg,'validation','main')
    train(cfg,'cv');evaluate(cfg,'test','cv')
    combine()
    write_json(folder(cfg,'reports')/'runtime.json',{'training_evaluation_reporting_wall_s':time.perf_counter()-start,
        'formal_response_model_fits':144,'formal_network_fits':216,'extra_probe_network_fits':1,
        'final_holdout_evaluated':False})

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','probe','run','combine']);a=p.parse_args()
    {'prepare':prepare,'probe':probe,'run':run,'combine':combine}[a.stage]()
