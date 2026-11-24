"""Gap-preserving raw-data audit. No imputation, label overwrite or training.
Run: python scripts/audit_hku956_quality.py
"""
from pathlib import Path
import hashlib, io, json, sys, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.author_alignment import load_events, build_episodes, choose_episode, episode_events, reconstruct_exposure
OUT=ROOT/'reports/quality_audit_20261101'
RAW=ROOT/'data/raw/HKU956/HKU956'
SUP=ROOT/'data/raw/author_supplement_20261028'
# Explicit exploratory engineering choices, not validated clinical thresholds.
TOL=.002
POLICIES=[
 dict(name='E1_permissive',coverage=.50,gap=10.,pairs=20,artifact=.20,run=0.),
 dict(name='E2_intermediate',coverage=.70,gap=5.,pairs=20,artifact=.10,run=0.),
 dict(name='E3_stringent',coverage=.80,gap=3.,pairs=20,artifact=.05,run=0.),
 dict(name='E4_high_coverage',coverage=.90,gap=1.,pairs=30,artifact=.02,run=0.),
 dict(name='E5_continuous20',coverage=.80,gap=3.,pairs=20,artifact=.05,run=20.),
]

def union_gaps(intervals, duration):
    """Exact union, never sum overlapping supports. All gaps retained internally."""
    merged=[]
    for a,b in sorted(intervals):
        a=max(0.,float(a));b=min(float(duration),float(b))
        if not np.isfinite(a+b) or b<=a:continue
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    cursor=0.;gaps=[]
    for a,b in merged:
        if a>cursor:gaps.append((cursor,a))
        cursor=b
    if cursor<duration:gaps.append((cursor,float(duration)))
    return sum(b-a for a,b in merged),gaps

def window_audit(t,v,duration,tol=TOL,relative=.20):
    """t: relative interval END seconds; v: interval seconds; preserve original indices.
    A candidate NN pair must be adjacent rows AND agree in interval endpoint time.
    Abrupt-change flags apply only to truly contiguous, in-range intervals.
    Both endpoints of an abrupt pair are conservatively excluded, with no repair.
    """
    t=np.asarray(t,float);v=np.asarray(v,float);n=len(v)
    order_ok=bool(np.isfinite(t).all() and np.all(np.diff(t)>0))
    finite=np.isfinite(t)&np.isfinite(v)
    inside=finite&(t-v>=-1e-6)&(t<=duration+1e-6)&(t>=0)
    plausible=inside&(v>=.3)&(v<=2.)
    dt=np.diff(t);residual=dt-v[1:]
    adjacent=(np.abs(residual)<=tol)&(dt>0)
    abrupt=np.zeros(max(0,n-1),bool)
    if n>1:
        abrupt=adjacent&plausible[:-1]&plausible[1:]&(np.abs(np.diff(v))/np.maximum(np.abs(v[:-1]),1e-9)>relative)
    flagged=np.zeros(n,bool)
    if n>1:flagged[:-1]|=abrupt;flagged[1:]|=abrupt
    good=plausible&~flagged
    pair=good[:-1]&good[1:]&adjacent if n>1 else np.zeros(0,bool)
    # Never sort invalid timestamp rows into an apparently valid sequence.
    if not order_ok:pair[:]=False;good[:]=False
    involved=np.zeros(n,bool)
    if n>1:involved[:-1]|=pair;involved[1:]|=pair
    pcovered,pgaps=union_gaps(zip((t-v)[plausible],t[plausible]),duration)
    covered,gaps=union_gaps(zip((t-v)[good],t[good]),duration)
    paircovered,_=union_gaps(zip((t-v)[involved],t[involved]),duration)
    # Maximal runs of accepted consecutive pairs, without crossing excluded indices.
    runs=[];i=0
    while i<len(pair):
        if not pair[i]:i+=1;continue
        j=i
        while j+1<len(pair) and pair[j+1]:j+=1
        runs.append((i,j+1));i=j+1
    longest=max(runs,key=lambda ij:t[ij[1]]-(t[ij[0]]-v[ij[0]]),default=None)
    squared=np.diff(v*1000)**2
    observed=finite&(t>=0)&(t<=duration+1e-6)
    count=int(observed.sum());median=float(np.median(v[plausible])) if plausible.any() else np.nan
    expected=duration/median if np.isfinite(median) else np.nan
    inwindow_pair=observed[:-1]&observed[1:]
    rawgap=inwindow_pair&(residual>tol)
    overlap=inwindow_pair&(residual < -tol)
    artifact_count=int((inside&~((v>=.3)&(v<=2.))).sum()+flagged.sum())
    metrics={
      'duration_s':float(duration),'ibi_endpoints_in_window':count,'ibi_fully_contained_count':int(inside.sum()),
      'ibi_timestamp_span_s':float(np.ptp(t[observed])) if count else np.nan,
      'ibi_plausible_count':int(plausible.sum()),'ibi_flagged_artifact_count':artifact_count,
      'ibi_artifact_fraction':artifact_count/int(inside.sum()) if inside.any() else np.nan,
      'plausible_expected_beats_estimate':expected,
      'plausible_count_coverage_estimate':int(plausible.sum())/expected if expected>0 else np.nan,
      'plausible_interval_coverage':pcovered/duration,'candidate_nn_coverage':covered/duration,
      'valid_pair_support_coverage':paircovered/duration,
      'timestamp_order_valid':order_ok,'timestamp_adjacent_pair_count':int((adjacent&inwindow_pair).sum()),
      'raw_discontinuity_count':int(rawgap.sum()),'raw_discontinuity_total_s':float(residual[rawgap].sum()),
      'raw_discontinuity_max_s':float(residual[rawgap].max()) if rawgap.any() else 0.,
      'overlap_inconsistency_count':int(overlap.sum()),
      'valid_consecutive_nn_pairs':int(pair.sum()),'continuous_run_count':len(runs),
      'longest_continuous_run_s':float(t[longest[1]]-(t[longest[0]]-v[longest[0]])) if longest else 0.,
      'longest_continuous_run_pairs':int(longest[1]-longest[0]) if longest else 0,
      'rmssd_observed_pairs_ms':float(np.sqrt(squared[pair].mean())) if pair.any() else np.nan,
      'rmssd_longest_run_ms':float(np.sqrt(squared[longest[0]:longest[1]].mean())) if longest else np.nan,
      'uncovered_gap_count_gt_tolerance':sum(b-a>tol for a,b in gaps),
      'uncovered_total_s':duration-covered,
      'max_uncovered_gap_s':max((b-a for a,b in gaps),default=0.),
      'leading_uncovered_s':gaps[0][1] if gaps and gaps[0][0]==0 else 0.,
      'trailing_uncovered_s':duration-gaps[-1][0] if gaps and gaps[-1][1]==duration else 0.,
      'plausible_max_uncovered_gap_s':max((b-a for a,b in pgaps),default=0.)}
    return metrics,gaps,pair,good

def retention(m,p,alignment=True):
    reasons=[]
    for ok,why in [
      (alignment,'unmatched_log'),(m['duration_s']>=30,'duration_lt30'),
      (m['timestamp_order_valid'],'timestamp_order'),
      (m['overlap_inconsistency_count']==0,'interval_overlap'),
      (m['candidate_nn_coverage']>=p['coverage'],'coverage'),
      (m['max_uncovered_gap_s']<=p['gap'],'max_gap'),
      (m['valid_consecutive_nn_pairs']>=p['pairs'],'pairs'),
      (m['ibi_artifact_fraction']<=p['artifact'],'artifact_fraction'),
      (m['longest_continuous_run_s']>=p['run'],'continuous_run')]:
        if not ok:reasons.append(why)
    return not reasons,';'.join(reasons)

def longest_true(mask):
    best=cur=0
    for x in mask:
        cur=cur+1 if x else 0;best=max(best,cur)
    return best

def read_column(path):
    if not path.is_file() or not path.stat().st_size:return np.array([])
    a=pd.read_csv(path,header=None).to_numpy(float)
    if a.shape[1]!=1:raise ValueError(f'Unexpected raw columns: {path}')
    return a[:,0]

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    ratings=pd.read_csv(RAW/'3. AV_ratings.csv',dtype={'song_id':str})
    assert not ratings.duplicated(['participant_id','song_no','song_id']).any()
    events=load_events(SUP/'raw_log.download','Asia/Hong_Kong')
    episodes,unclosed=build_episodes(events)
    cached=pd.read_csv(ROOT/'data/processed/author_v2/record_alignment.csv').set_index('record_id')
    recovered=pd.read_csv(ROOT/'reports/jamendo_recovery_all.csv',dtype={'song_id':str}).set_index('song_id')
    groups=json.loads((ROOT/'data/splits/participants.json').read_text())['groups']
    parts={person:part for part,people in groups.items() for person in people}
    audio={}
    for sid,a in recovered.iterrows():
        p=Path(str(a.local_path))
        if a.status=='downloaded_candidate' and p.is_file():
            info=sf.info(p);audio[sid]={'audio_file_duration_s':info.duration,'audio_sample_rate_hz':info.samplerate,'audio_path':str(p)}
    rows=[];gaprows=[];sensitivity=[];inventory=[];res_end=[];res_start=[];flags=[];pairrows=[]
    archive=zipfile.ZipFile(SUP/'timestamped_ibi.download')
    zipnames={n for n in archive.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')}
    expected_names=set();expected_files=set();bounds=[]
    for r in ratings.itertuples():
        rid=f'{r.participant_id}_{r.song_no}_{r.song_id}'
        key=f'{r.song_no}_{r.song_id}.csv'
        name=f'IBI_timestamped/{r.participant_id}/IBI/{key}';expected_names.add(name)
        content=archive.read(name);ib=pd.read_csv(io.BytesIO(content))
        assert list(ib.columns)==['timestamp','IBI']
        absolute=ib.timestamp.to_numpy(float);v=ib.IBI.to_numpy(float)
        signals={}
        for s in ['IBI','HR','BVP']:
            p=RAW/'1. physiological_signals'/r.participant_id/s/key;expected_files.add(str(p))
            x=read_column(p);signals[s]=x
            inventory.append({'record_id':rid,'signal':s,'path':str(p.relative_to(ROOT)),'exists':p.is_file(),
              'bytes':p.stat().st_size if p.is_file() else 0,'samples':len(x),'columns':1,'has_timestamps':False,
              'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ''})
        same=np.array_equal(v,signals['IBI'],equal_nan=True)
        found,pre=choose_episode(episodes,r.participant_id,r.song_id,r.play_duration,absolute)
        match=len(found)==1
        if not match:raise ValueError(f'Cannot anchor audit window uniquely: {rid}')
        ep=found.iloc[0].to_dict();D=float(ep['log_duration_s']);t=absolute-ep['wall_start_unix_s']
        bouts,exposure=reconstruct_exposure(episode_events(events,ep),rid)
        prior=cached.loc[rid]
        agreement=ep['start_event_id']==prior.start_event_id and ep['end_event_id']==prior.end_event_id
        finite_v=v[np.isfinite(v)];finite_t=t[np.isfinite(t)]
        base={'record_id':rid,'participant_id':r.participant_id,'song_no':r.song_no,'song_id':r.song_id,
          'partition':parts[r.participant_id],'published_listening_duration_s':r.play_duration,
          'listening_duration_s':D,'wall_start_unix_s':ep['wall_start_unix_s'],'wall_end_unix_s':ep['wall_end_unix_s'],
          'log_match_count':len(found),'log_candidates_before_ibi_bounds':pre,'log_match_agrees_prior':agreement,
          'log_start_kind':ep['start_kind'],'published_minus_log_duration_s':r.play_duration-D,
          **exposure,'audio_available':r.song_id in audio,
          **audio.get(r.song_id,{'audio_file_duration_s':np.nan,'audio_sample_rate_hz':np.nan,'audio_path':''}),
          'ibi_count':len(v),'ibi_original_matches_timestamped':same,
          'ibi_nonfinite_values':int((~np.isfinite(v)).sum()),'ibi_nonfinite_timestamps':int((~np.isfinite(t)).sum()),
          'ibi_min_ms':float(finite_v.min()*1000) if len(finite_v) else np.nan,
          'ibi_max_ms':float(finite_v.max()*1000) if len(finite_v) else np.nan,
          'ibi_outside_300_2000_ms_count':int(((v<.3)|(v>2.)).sum()),
          'ibi_nonpositive_count':int((v<=0).sum()),'ibi_duplicate_timestamps':int((np.diff(t)==0).sum()),
          'ibi_decreasing_timestamps':int((np.diff(t)<0).sum()),
          'ibi_timestamp_span_s':float(np.ptp(finite_t)) if len(finite_t) else np.nan,
          'first_ibi_endpoint_s':float(t[0]) if len(t) else np.nan,'last_ibi_endpoint_s':float(t[-1]) if len(t) else np.nan,
          'ibi_endpoints_outside_wall_count':int(((t<0)|(t>D)).sum()),
          'ibi_intervals_crossing_start_count':int(((t-v<0)&(t>=0)).sum()),
          'arousal_available_in_range':bool(np.isfinite(r.arousal_rating) and -10<=r.arousal_rating<=10),
          'valence_available_in_range':bool(np.isfinite(r.valence_rating) and -10<=r.valence_rating<=10)}
        if len(t)>1:
            res_end.extend(np.abs(np.diff(t)-v[1:]));res_start.extend(np.abs(np.diff(t)-v[:-1]))
        for s,hz in [('HR',1),('BVP',64)]:
            x=signals[s];finite=np.isfinite(x);valid=x[finite];low=s.lower()
            base.update({f'{low}_samples':len(x),f'{low}_nonfinite_count':int((~finite).sum()),
             f'{low}_sample_duration_s':len(x)/hz,f'{low}_duration_minus_wall_s':len(x)/hz-D,
             f'{low}_finite_sample_coverage':min(1.,int(finite[:int(np.ceil(D*hz))].sum())/(D*hz)),
             f'{low}_minimum':float(valid.min()) if len(valid) else np.nan,f'{low}_maximum':float(valid.max()) if len(valid) else np.nan,
             f'{low}_constant':bool(len(valid)>0 and np.ptp(valid)==0),
             f'{low}_longest_identical_run_s':(longest_true(np.diff(x)==0)+1)/hz if len(x) else 0})
        hr=signals['HR'];hr_good=np.isfinite(hr)&(hr>=30)&(hr<=220)
        base['hr_outside_30_220_count']=int((np.isfinite(hr)&~hr_good).sum())
        for window,duration in [('full',D),('first30',min(30.,D))]:
            m,gaps,pair,good=window_audit(t,v,duration)
            base.update({f'{window}_{k}':value for k,value in m.items()})
            for i in range(1,len(v)):
                if 0<=t[i-1]<=duration and 0<=t[i]<=duration:
                    pairrows.append({'record_id':rid,'window':window,'left_original_row':i-1,'right_original_row':i,
                      'timestamp_delta_s':t[i]-t[i-1],'current_ibi_s':v[i],
                      'adjacency_residual_s':t[i]-t[i-1]-v[i],'left_interval_eligible':bool(good[i-1]),
                      'right_interval_eligible':bool(good[i]),'pair_accepted':bool(pair[i-1]),
                      'squared_difference_ms2':float(((v[i]-v[i-1])*1000)**2)})
            h=hr[:int(np.ceil(duration))];hv=np.isfinite(h)&(h>=30)&(h<=220)
            base[f'{window}_hr_plausible_sample_coverage']=min(1.,int(hv.sum())/duration)
            base[f'{window}_mean_hr_bpm']=float(h[hv].mean()) if hv.any() else np.nan
            base[f'{window}_hr_longest_invalid_run_s']=longest_true(~hv)
            pause=sum(max(0,min(x['wall_end_unix_s']-ep['wall_start_unix_s'],duration)-max(0,x['wall_start_unix_s']-ep['wall_start_unix_s'])) for x in bouts if x['state']=='paused')
            base[f'{window}_pause_s']=pause
            for p in POLICIES:
                ok,why=retention(m,p,match)
                base[f'{window}_{p["name"]}_retained']=ok
                base[f'{window}_{p["name"]}_failure_reasons']=why
            for a,b in gaps:
                if b-a>TOL:gaprows.append({'record_id':rid,'participant_id':r.participant_id,'window':window,'gap_start_s':a,'gap_end_s':b,'gap_duration_s':b-a,'kind':'boundary' if a==0 or b==duration else 'internal','definition':'complement_of_candidate_NN_interval_union'})
            # Separate tolerance/artifact sensitivity, not an automatically selected policy.
            for tol,rel in [(TOL,.20),(.016,.20),(.04,.20),(TOL,.10),(TOL,.30)]:
                q=window_audit(t,v,duration,tol=tol,relative=rel)[0]
                ok,_=retention(q,POLICIES[2],match)
                sensitivity.append({'record_id':rid,'participant_id':r.participant_id,'window':window,'tolerance_s':tol,'relative_change_limit':rel,'E3_retained':ok,'valid_pairs':q['valid_consecutive_nn_pairs']})
        fullgood=window_audit(t,v,D)[3]
        for i in range(len(v)):
            flags.append({'record_id':rid,'row_index':i,'time_s':t[i],'ibi_ms':v[i]*1000,
             'interval_start_s':t[i]-v[i],'finite':bool(np.isfinite(t[i]) and np.isfinite(v[i])),
             'plausible_value':bool(np.isfinite(v[i]) and .3<=v[i]<=2.),
             'candidate_NN_full_window':bool(fullgood[i]),
             'abrupt_previous_contiguous_pair':bool(i and abs(t[i]-t[i-1]-v[i])<=TOL and .3<=v[i-1]<=2 and .3<=v[i]<=2 and t[i-1]-v[i-1]>=-1e-6 and abs(v[i]-v[i-1])/v[i-1]>.20),
             'abrupt_next_contiguous_pair':bool(i+1<len(v) and abs(t[i+1]-t[i]-v[i+1])<=TOL and .3<=v[i]<=2 and .3<=v[i+1]<=2 and t[i]-v[i]>=-1e-6 and abs(v[i+1]-v[i])/v[i]>.20),
             'fully_within_full_window':bool(np.isfinite(t[i]-v[i]) and t[i]-v[i]>=-1e-6 and t[i]<=D+1e-6),
             'adjacent_to_previous':bool(i and t[i]>t[i-1] and abs(t[i]-t[i-1]-v[i])<=TOL),
             'endpoint_residual_s':t[i]-t[i-1]-v[i] if i else np.nan})
        rows.append(base)
    assert expected_names==zipnames
    actual_files={str(p) for s in ['IBI','HR','BVP'] for p in (RAW/'1. physiological_signals').glob(f'*/{s}/*.csv')}
    df=pd.DataFrame(rows);assert len(df)==956 and df.record_id.nunique()==956
    df.to_csv(OUT/'record_quality_audit.csv',index=False)
    pd.DataFrame(gaprows).to_csv(OUT/'gap_events.csv',index=False)
    pd.DataFrame(flags).to_csv(OUT/'ibi_interval_diagnostics.csv',index=False)
    pd.DataFrame(pairrows).to_csv(OUT/'pair_diagnostics.csv',index=False)
    pd.DataFrame(inventory).to_csv(OUT/'raw_signal_inventory.csv',index=False)
    sens=pd.DataFrame(sensitivity);sens.to_csv(OUT/'threshold_sensitivity_records.csv',index=False)
    summary=[];participant=[]
    for window in ['full','first30']:
        for p in POLICIES:
            col=f'{window}_{p["name"]}_retained'
            for scope,g in [('all',df),*df.groupby('partition')]:
                kept=g[g[col]];with_audio=kept[kept.audio_available]
                summary.append({'window':window,'policy':p['name'],'scope':scope,'total_records':len(g),'retained_records':len(kept),
                  'retained_participants':kept.participant_id.nunique(),'retained_percent':100*len(kept)/len(g),
                  'with_audio_records':len(with_audio),'with_audio_participants':with_audio.participant_id.nunique(),
                  'without_pause_records':int((kept[f'{window}_pause_s']==0).sum()),
                  'audio_and_both_AV_records':int((kept.audio_available&kept.arousal_available_in_range&kept.valence_available_in_range).sum()),
                  'participants_with_at_least5_records':int((kept.groupby('participant_id').size()>=5).sum()),**p})
            for person,g in df.groupby('participant_id'):
                k=g[g[col]];participant.append({'participant_id':person,'partition':g.partition.iloc[0],'window':window,'policy':p['name'],
                  'total_records':len(g),'retained_records':len(k),'retention_fraction':len(k)/len(g),
                  'with_audio_records':int(k.audio_available.sum()),'median_full_coverage':g.full_candidate_nn_coverage.median()})
    counts=pd.DataFrame(summary);counts.to_csv(OUT/'retention_summary.csv',index=False)
    pp=pd.DataFrame(participant);pp.to_csv(OUT/'participant_retention.csv',index=False)
    ss=sens.groupby(['window','tolerance_s','relative_change_limit']).apply(lambda g:pd.Series({'retained_records':int(g.E3_retained.sum()),'retained_participants':g.loc[g.E3_retained,'participant_id'].nunique()}),include_groups=False).reset_index()
    ss.to_csv(OUT/'threshold_sensitivity_summary.csv',index=False)
    hrsummary=[]
    for window in ['full','first30']:
        for cov in [.80,.95,1.]:
            k=df[(df[f'{window}_hr_plausible_sample_coverage']>=cov)&(df[f'{window}_duration_s']>=30)]
            hrsummary.append({'window':window,'min_coverage':cov,'records':len(k),'participants':k.participant_id.nunique(),'with_audio_records':int(k.audio_available.sum())})
    pd.DataFrame(hrsummary).to_csv(OUT/'hr_fallback_summary.csv',index=False)
    provenance=[]
    for p in [RAW/'README',RAW/'3. AV_ratings.csv',RAW/'2. original_song_audio.csv',SUP/'raw_log.download',SUP/'timestamped_ibi.download',ROOT/'data/processed/author_v2/record_alignment.csv',ROOT/'reports/jamendo_recovery_all.csv',ROOT/'data/splits/participants.json',Path(__file__)]:
        provenance.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    stats={'records':len(df),'participants':df.participant_id.nunique(),'unique_songs':df.song_id.nunique(),
      'empty_ibi_records':int((df.ibi_count==0).sum()),'all_original_ibi_match':bool(df.ibi_original_matches_timestamped.all()),
      'all_alignment_matches_prior':bool(df.log_match_agrees_prior.all()),'unused_signal_files':sorted(actual_files-expected_files),
      'missing_signal_files':sorted(expected_files-actual_files),'timestamp_nonfinite_records':int((df.ibi_nonfinite_timestamps>0).sum()),
      'timestamp_nonmonotonic_records':int(((df.ibi_duplicate_timestamps+df.ibi_decreasing_timestamps)>0).sum()),
      'ibi_range_flag_records':int((df.ibi_outside_300_2000_ms_count>0).sum()),
      'ibi_nonfinite_records':int((df.ibi_nonfinite_values>0).sum()),
      'endpoints_outside_wall_records':int((df.ibi_endpoints_outside_wall_count>0).sum()),
      'interval_crosses_start_records':int((df.ibi_intervals_crossing_start_count>0).sum()),
      'zero_valid_pairs_full_records':int((df.full_valid_consecutive_nn_pairs==0).sum()),
      'raw_gap_records':int((df.full_raw_discontinuity_count>0).sum()),
      'records_with_pause':int((df.paused_wall_s>0).sum()),'exposure_status_counts':df.exposure_status.value_counts().to_dict(),
      'audio_records':int(df.audio_available.sum()),'audio_songs':len(audio),
      'endpoint_convention_pair_matches_2ms':int((np.asarray(res_end)<=TOL).sum()),
      'startpoint_convention_pair_matches_2ms':int((np.asarray(res_start)<=TOL).sum()),'total_successive_rows':len(res_end),
      'endpoint_residual_abs_quantiles':np.quantile(res_end,[0,.25,.5,.75,.9,.99,1]).tolist(),
      'minimum_duration_s':float(df.listening_duration_s.min()),'median_duration_s':float(df.listening_duration_s.median()),
      'maximum_duration_s':float(df.listening_duration_s.max()),'records_under30s':int((df.listening_duration_s<30).sum()),
      'hr_incomplete_records':int((df.full_hr_plausible_sample_coverage<1).sum()),
      'hr_duration_mismatch_records':int((df.hr_duration_minus_wall_s!=0).sum()),
      'bvp_incomplete_finite_records':int((df.bvp_finite_sample_coverage<1).sum()),
      'bvp_duration_mismatch_records':int((df.bvp_duration_minus_wall_s!=0).sum()),
      'full_coverage_quantiles':df.full_candidate_nn_coverage.quantile([0,.25,.5,.75,1]).to_dict(),
      'training_performed':False,'imputed_intervals':0,'HR_converted_to_HRV':False,
      'policy_scope':'exploratory engineering sensitivity, not validated thresholds',
      'ibi_bounds_ms':[300,2000],'relative_change_limit':.20,'adjacency_tolerance_s':TOL,
      'policies':POLICIES,'input_provenance':provenance}
    (OUT/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    plot(df,counts,pp)
    print(json.dumps({k:v for k,v in stats.items() if k not in ['input_provenance','policies']},indent=2))
    print(counts[counts.scope=='all'][['window','policy','retained_records','retained_participants','with_audio_records','participants_with_at_least5_records']].to_string(index=False))

def plot(df,counts,pp):
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].hist(df.ibi_count,bins=40,color='#397ca5');axes[0,0].set(xlabel='Observed IBI rows per full record',ylabel='Records',title='IBI availability (including empty files)')
    axes[0,1].hist(df.full_candidate_nn_coverage,bins=np.linspace(0,1,21),alpha=.75,label='Candidate NN interval coverage')
    axes[0,1].hist(df.full_valid_pair_support_coverage,bins=np.linspace(0,1,21),histtype='step',lw=2,label='Valid-pair support coverage')
    axes[0,1].set(xlabel='Estimated fraction of listening window',ylabel='Records',title='Coverage is not timestamp span');axes[0,1].legend(fontsize=8)
    axes[1,0].hist(df.full_max_uncovered_gap_s,bins=40,color='#ba6650');axes[1,0].set(xlabel='Longest unsupported gap incl. boundaries (s)',ylabel='Records',title='Candidate NN support gaps')
    for w,c in [('full','#397ca5'),('first30','#d18a37')]:
        k=counts[(counts.scope=='all')&(counts.window==w)];axes[1,1].plot(k.policy,k.retained_records,'o-',label=w,color=c)
    axes[1,1].set(ylabel='Retained recordings',title='Exploratory policies; minimum duration 30 s');axes[1,1].tick_params(axis='x',rotation=25);axes[1,1].legend()
    fig.tight_layout();fig.savefig(OUT/'quality_distributions.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,9),sharex=True)
    for ax,w in zip(axes,['full','first30']):
        table=pp[pp.window==w].pivot(index='policy',columns='participant_id',values='retained_records').reindex([p['name'] for p in POLICIES])
        im=ax.imshow(table,aspect='auto',cmap='Blues',vmin=0,vmax=table.to_numpy().max())
        ax.set_yticks(range(len(table)),table.index);ax.set_title(f'{w}: retained records per participant (zeros shown)')
        ax.set_xticks(range(len(table.columns)),table.columns,rotation=60,ha='right')
        for i in range(len(table)):
            for j in range(len(table.columns)):ax.text(j,i,str(table.iloc[i,j]),ha='center',va='center',fontsize=7,color='black' if table.iloc[i,j]<table.to_numpy().max()*.6 else 'white')
        fig.colorbar(im,ax=ax,label='Records')
    fig.tight_layout();fig.savefig(OUT/'participant_retention.png',dpi=170);plt.close(fig)

if __name__=='__main__':run()
