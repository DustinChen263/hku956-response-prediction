"""Whole-track acoustic baseline; distinct from verified exposure-segment modeling.

Only track identity links inputs to observed responses. Never sets exposure
alignment_verified=True. Final holdout labels are not numerically parsed.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
from src.common import digest,write_json,load_config,folder,environment
from src.extract_audio import segment_features
from src.train import train
from src.evaluate import evaluate,summarize,paired_bootstrap

CONFIG=ROOT/'configs/song_level_v1.yaml'


def prepare():
    cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text())
    cfg['paths'].update(processed_dir='data/processed/song_level_v1',features_dir='features/song_level_v1',results_dir='results/song_level_v1',reports_dir='reports/song_level_v1')
    cfg['audio']['representation_scope']='whole_track'
    cfg['training']['models']=['mean','song_mean','ridge']
    cfg['research_scope']={'primary_targets':['mean_hr','arousal','valence'],'hrv_required':False,
        'input':'full Jamendo track with corresponding ID; not reconstructed playback',
        'mean_hr_endpoint':'first30_wall_clock','av_endpoint':'post_listening_rating',
        'claim':'track-level association/prediction for unseen participants; exact stimulus exposure not established',
        'final_holdout_evaluation':False}
    if CONFIG.exists() and yaml.safe_load(CONFIG.read_text())!=cfg:raise ValueError('Existing protocol differs')
    CONFIG.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    cfg=load_config(CONFIG);out=folder(cfg,'processed');rep=folder(cfg,'reports')
    a=pd.read_csv(ROOT/'data/processed/author_v2/record_alignment.csv',dtype={'song_id':str})
    with (ROOT/'data/processed/labels.csv').open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;labels=list(reader)
    labels_by_id={r['record_id']:r for r in labels}
    split=json.loads((ROOT/'data/splits/participants.json').read_text())
    partition={p:k for k,people in split['groups'].items() for p in people}
    rows=[]
    for r in a.to_dict('records'):
        lab=labels_by_id[r['record_id']]
        available=bool(r['audio_available']) and Path(str(r['audio_path'])).is_file()
        complete=lab['hr_quality']=='pass' and all(lab.get(t,'') not in ('','nan') for t in cfg['training']['targets'])
        rows.append({'record_id':r['record_id'],'participant_id':r['participant_id'],'song_id':r['song_id'],
            'partition':partition[r['participant_id']],'audio_path':r['audio_path'] if available else '',
            'audio_start_s':0. if available else np.nan,
            # duration_s is solely the feature input span required by the existing design loader.
            'duration_s':r['audio_duration_s'] if available else np.nan,
            'audio_representation':'whole_track','feature_span_semantics':'entire_recording_not_listening',
            'listening_wall_duration_s':r['published_duration_s'],'physiology_window_s':30,
            'paused_wall_s':r['paused_wall_s'],'exposure_status':r['exposure_status'],
            'alignment_verified':False,'audio_available':available,'primary_labels_available':complete,
            'main_eligible_before_features':available and complete,
            'exclusion_reason':'' if available and complete else 'missing_audio' if not available else 'primary_label_qc'})
    manifest=pd.DataFrame(rows)
    manifest.to_csv(out/'manifest.csv',index=False)
    # Preserve strings; fitting parses only explicitly selected development IDs.
    with (out/'labels.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(labels)
    stats=[]
    for part,sub in manifest.groupby('partition'):
        good=sub[sub.main_eligible_before_features]
        stats.append({'partition':part,'all_records':len(sub),'audio_records':int(sub.audio_available.sum()),
            'main_records_before_features':len(good),'main_participants':good.participant_id.nunique(),
            'main_unique_songs':good.song_id.nunique(),'records_with_pause':int((good.paused_wall_s>0).sum())})
    pd.DataFrame(stats).to_csv(rep/'cohort_counts.csv',index=False)
    primary=manifest[manifest.main_eligible_before_features]
    primary.groupby(['partition','participant_id']).size().rename('records').to_csv(rep/'participant_counts.csv')
    songsets={p:set(primary.loc[primary.partition==p,'song_id']) for p in split['groups']}
    overlaps={f'{a}__{b}':len(songsets[a]&songsets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    snapshot={'status':'locked_before_feature_extraction_and_model_selection','config':cfg,
       'environment':environment(),'song_overlap_counts':overlaps,'final_holdout_labels_numerically_parsed':False,
       'files_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [CONFIG,Path(__file__),out/'manifest.csv',out/'labels.csv',ROOT/'data/splits/participants.json']}}
    lock=rep/'protocol.json'
    if lock.exists() and json.loads(lock.read_text())['files_sha256']!=snapshot['files_sha256']:raise ValueError('Protocol changed after lock')
    if not lock.exists():write_json(lock,snapshot)
    print(pd.DataFrame(stats).to_string(index=False),flush=True)
    print('Cross-partition song overlaps:',overlaps,flush=True)


def extract_one(row,audio,cache):
    start=time.perf_counter()
    feat,meta=segment_features(row['audio_path'],0.,row['duration_s'],audio,cache)
    return row['song_id'],feat,meta,time.perf_counter()-start


def extract(limit=None,workers=2):
    cfg=load_config(CONFIG);out=folder(cfg,'features');rep=folder(cfg,'reports')
    os.environ.setdefault('NUMBA_CACHE_DIR',str(out/'cache/numba'))
    frame=pd.read_csv(folder(cfg,'processed')/'manifest.csv',dtype={'song_id':str})
    tracks=frame[frame.audio_available].drop_duplicates('song_id').sort_values('song_id').to_dict('records')
    chosen=tracks[:limit] if limit else tracks
    results={};failures=[];started=time.perf_counter();timings=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(extract_one,r,cfg['audio'],out/'cache'):r for r in chosen}
        for job in as_completed(jobs):
            row=jobs[job]
            try:
                song,feat,meta,seconds=job.result();results[song]=(feat,meta)
                timings.append({'song_id':song,'input_duration_s':row['duration_s'],'wall_worker_s':seconds})
            except Exception as exc:
                failures.append({'song_id':row['song_id'],'reason':f'{type(exc).__name__}: {exc}'})
            done=len(results)+len(failures)
            if done%20==0 or done==len(chosen):print(f'Features {done}/{len(chosen)}; failures {len(failures)}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    elapsed=time.perf_counter()-started
    pd.DataFrame(timings).to_csv(rep/('probe_timings.csv' if limit else 'feature_timings.csv'),index=False)
    pd.DataFrame(failures,columns=['song_id','reason']).to_csv(rep/('probe_failures.csv' if limit else 'feature_failures.csv'),index=False)
    if limit:
        write_json(rep/'runtime_estimate.json',{'probe_tracks':len(chosen),'workers':workers,'wall_s':elapsed,
            'total_tracks':len(tracks),'linear_total_estimate_s':elapsed/len(chosen)*len(tracks),'includes_initial_import_and_jit':True})
        print(f'Probe estimate: {elapsed/len(chosen)*len(tracks)/60:.1f} minutes for {len(tracks)} tracks (includes initialization)',flush=True)
        if failures:raise RuntimeError('Probe failed; inspect failures before full extraction')
        return
    used=frame[frame.song_id.isin(results)].copy()
    if not len(used):raise RuntimeError('No extracted features')
    X=np.stack([results[s][0] for s in used.song_id])
    np.savez_compressed(out/'audio_features.npz',record_ids=used.record_id.to_numpy(str),X=X)
    write_json(out/'feature_metadata.json',{'synthetic':False,'config':cfg['audio'],'dimension':X.shape[1],
        'array_sha256':digest(out/'audio_features.npz'),
        'scope':'whole tracks; no claim of verified listening exposure',
        'records':[{'record_id':r.record_id,**results[r.song_id][1]} for r in used.itertuples()]})
    write_json(rep/'extraction_summary.json',{'tracks':len(results),'records_with_features':len(used),
        'main_records_with_features':int(used.main_eligible_before_features.sum()),'dimension':X.shape[1],
        'failures':len(failures),'wall_s':elapsed,'final_holdout_outcomes_used':False})


def baseline():
    cfg=load_config(CONFIG)
    train(cfg,'main');evaluate(cfg,'validation','main')
    train(cfg,'cv');evaluate(cfg,'test','cv')  # Outer development folds, never the final holdout.
    final=set(json.loads((folder(cfg,'splits')/'participants.json').read_text())['groups']['test'])
    predictions=[]
    for f in sorted(folder(cfg,'results').glob('cv_*/predictions_test.csv')):
        p=pd.read_csv(f,dtype={'song_id':str});p['fold']=f.parent.name;p['split']='development_cv'
        assert not set(p.participant_id)&final
        predictions.append(p)
    combined=pd.concat(predictions,ignore_index=True)
    assert not combined.duplicated(['record_id','model','seed','target']).any()
    metrics,people=summarize(combined)
    out=folder(cfg,'results')
    combined.to_csv(out/'development_cv_predictions.csv',index=False)
    metrics.to_csv(out/'development_cv_metrics.csv',index=False)
    people.to_csv(out/'development_cv_participant_metrics.csv',index=False)
    paired_bootstrap(combined,cfg['training']['bootstrap_samples']).to_csv(out/'development_cv_paired_bootstrap.csv',index=False)
    print(metrics[['model','target','n','participants','macro_mae','rmse']].to_string(index=False),flush=True)
    assert not (out/'evaluation_test.json').exists()

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','probe','extract','baseline']);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='probe':extract(4,a.workers)
    elif a.stage=='extract':extract(None,a.workers)
    else:baseline()
