"""Observed-IBI recovery audit; never impute beats or overwrite original labels.

Search policy is fixed: earliest passing 30s window, with starts at zero or
observed interval boundaries. No RMSSD magnitude, A/V rating or model score is
used to choose a window. Later windows define a different outcome.
"""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.physiology import rmssd_qc


def candidate_starts(times, values, duration):
    starts=np.r_[0.,times-values/1000]
    return np.unique(starts[np.isfinite(starts)&(starts>=0)&(starts+30<=duration+1e-6)])


def pause_overlap(start, intervals):
    return sum(max(0.,min(start+30,b)-max(start,a)) for a,b in intervals)


def independent_rmssd(values,times,start,duration,cfg):
    """Loop implementation for independently checking selected pair arithmetic."""
    kept=[(float(v),float(t-start)) for v,t in zip(values,times)
          if t-start-v/1000>=-1e-6 and t-start<=duration+1e-6]
    valid=[]
    for i,(v,t) in enumerate(kept):
        ok=np.isfinite(v) and cfg['min_ibi_ms']<=v<=cfg['max_ibi_ms']
        if i:ok=ok and abs(v-kept[i-1][0])/max(abs(kept[i-1][0]),1)<=cfg['max_relative_change']
        valid.append(ok)
    diffs=[(kept[i][0]-kept[i-1][0])**2 for i in range(1,len(kept))
           if valid[i] and valid[i-1] and abs(kept[i][1]-kept[i-1][1]-kept[i][0]/1000)<=cfg.get('timestamp_tolerance_s',.04)]
    return (float(np.sqrt(sum(diffs)/len(diffs))) if diffs else np.nan),len(diffs)


def search_windows(values,times,duration,cfg,pauses):
    found={};checked=0
    for start in candidate_starts(times,values,duration):
        checked+=1
        q=rmssd_qc(values,times-start,30,cfg)
        if q['rmssd_quality']!='pass':continue
        overlap=pause_overlap(start,pauses)
        item={'window_start_s':float(start),'window_end_s':float(start+30),'pause_overlap_s':overlap,**q}
        found.setdefault('earliest_observed30',item)
        if overlap<=1e-6:found.setdefault('earliest_unpaused30',item)
        if len(found)==2:break
    return found,checked


def run():
    source=ROOT/'data/processed/author_v2'
    out=ROOT/'data/processed/hrv_recovery';out.mkdir(parents=True,exist_ok=True)
    report=ROOT/'reports/hrv_recovery';report.mkdir(parents=True,exist_ok=True)
    records=pd.read_csv(source/'record_alignment.csv')
    old=pd.read_csv(source/'hrv_first30_labels.csv').set_index('record_id')
    intervals=pd.read_csv(source/'exposure_intervals.csv')
    cfg=yaml.safe_load((ROOT/'configs/author_v2.yaml').read_text())['physiology']
    # Descriptive sensitivity only; it is not a validated or adopted QC protocol.
    relaxed={**cfg,'min_coverage':.6,'max_gap_s':5.}
    rows=[];windows=[];verification=[]
    for r in records.itertuples():
        f=pd.read_csv(source/'ibi'/f'{r.record_id}.csv');v=f.ibi_ms.to_numpy(float);t=f.time_s.to_numpy(float)
        paused=intervals[(intervals.record_id==r.record_id)&(intervals.state=='paused')]
        pauses=[(x.wall_start_unix_s-r.wall_start_unix_s,x.wall_end_unix_s-r.wall_start_unix_s) for x in paused.itertuples()]
        found,checked=search_windows(v,t,r.published_duration_s,cfg,pauses)
        first=old.loc[r.record_id]
        q=rmssd_qc(v,t,min(30,r.published_duration_s),relaxed) if r.published_duration_s>=30 else {'rmssd_quality':'fail','rmssd_ms':np.nan}
        base={'record_id':r.record_id,'participant_id':r.participant_id,'partition':r.partition,'audio_available':r.audio_available,
              'raw_ibi_count':len(v),'original_first30_pass':first.rmssd_quality=='pass',
              'searched_candidate_windows':checked,'search_observed30_pass':'earliest_observed30' in found,
              'search_unpaused30_pass':'earliest_unpaused30' in found,
              'sensitivity_first30_pass':q['rmssd_quality']=='pass',
              'sensitivity_first30_rmssd_ms':q['rmssd_ms'],
              'sensitivity_is_primary_label':False}
        for policy,item in found.items():
            independent,pairs=independent_rmssd(v,t,item['window_start_s'],30,cfg)
            assert pairs==item['valid_pairs'] and np.isclose(independent,item['rmssd_ms'],rtol=1e-12)
            verification.append({'record_id':r.record_id,'policy':policy,'independent_rmssd_ms':independent,'pairs':pairs,'verified':True})
            windows.append({**base,'policy':policy,'is_first30_endpoint':item['window_start_s']==0,
                            'imputed_beats':0,'audio_alignment_verified':False,**item})
        base['recovery_status']=('original_first30_pass' if base['original_first30_pass'] else
            'alternative_window_only' if found else 'empty_ibi' if not len(v) else
            'record_shorter_than30' if r.published_duration_s<30 else 'no_passing_searched_window')
        rows.append(base)
    audit=pd.DataFrame(rows);win=pd.DataFrame(windows)
    audit.to_csv(out/'record_recovery_audit.csv',index=False)
    win.to_csv(out/'alternative_window_labels.csv',index=False)
    pd.DataFrame(verification).to_csv(report/'independent_arithmetic_checks.csv',index=False)
    # Keep diagnostic candidates separate from usable labels.
    summaries=[]
    for policy,col in [('original_first30','original_first30_pass'),('earliest_observed30','search_observed30_pass'),('earliest_unpaused30','search_unpaused30_pass'),('sensitivity_first30_ONLY','sensitivity_first30_pass')]:
        for part,group in [('all',audit),*list(audit.groupby('partition'))]:
            ok=group[group[col]]
            summaries.append({'policy':policy,'partition':part,'records':len(ok),'participants':ok.participant_id.nunique(),'with_audio':int(ok.audio_available.sum()),
                              'additional_to_original_first30':int((~ok.original_first30_pass).sum())})
    counts=pd.DataFrame(summaries);counts.to_csv(report/'cohort_counts.csv',index=False)
    audit.groupby(['participant_id','partition'])[['original_first30_pass','search_observed30_pass','search_unpaused30_pass','sensitivity_first30_pass']].sum().to_csv(report/'participant_counts.csv')
    summary={'records':len(audit),'statuses':audit.recovery_status.value_counts().to_dict(),
             'thresholds_strict':cfg,'thresholds_sensitivity_only':relaxed,'imputed_beats':0,
             'independent_checks':len(verification),'searched_boundary_windows':int(audit.searched_candidate_windows.sum()),
             'search_scope':'zero and observed interval starts; earliest QC pass; no claim of exhaustive continuous-time optimization',
             'model_training':False,'original_labels_overwritten':False,
             'sha256_script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(counts.to_string(index=False));print(json.dumps(summary['statuses']))

if __name__=='__main__':run()
