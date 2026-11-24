"""Auditable author-supplement linkage, exposure evidence and timestamped IBI QC.

Ratings and test outcomes are not loaded. No audio-start=0 default or hidden-seek
interpolation is used. The original split, labels and experiments remain intact.
"""
from __future__ import annotations
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
from src.common import digest,write_json
from src.physiology import rmssd_qc

CLOSURES={'survey-popup','survey-popup-onend'}
MOTION={'song-start','song-play','song-pause','song-skip','survey-popup','survey-popup-onend','song-end'}
POSITION=re.compile(r'\bcurrent\s+(-?\d+(?:\.\d+)?)s\b')
PLAYED=re.compile(r'\bplayed\s+(-?\d+(?:\.\d+)?)s\b')


def media_fields(text):
    text=str(text)
    pos=POSITION.search(text);played=PLAYED.search(text)
    return (float(pos.group(1)) if pos else np.nan,float(played.group(1)) if played else np.nan)


def load_events(path,timezone):
    # Deliberately omit IP address, numeric account identifiers and survey answers.
    columns=['timestamp','session_id','element_clicked','content_id','content_name','id','username']
    events=pd.read_csv(path,usecols=columns,dtype={'content_id':str,'username':str,'session_id':str})
    if events.id.duplicated().any():raise ValueError('Duplicate event IDs')
    events=events.rename(columns={'username':'participant_id','content_id':'song_id','id':'event_id','element_clicked':'event'})
    times=pd.to_datetime(events.timestamp,errors='raise').dt.tz_localize(timezone,ambiguous='raise',nonexistent='raise')
    events['wall_unix_s']=times.astype('int64')/1e9
    fields=events.content_name.map(media_fields)
    events['media_position_s']=[v[0] for v in fields]
    events['played_counter_s']=[v[1] for v in fields]
    events=events.drop(columns=['content_name'])
    events=events.sort_values(['participant_id','session_id','wall_unix_s','event_id'],na_position='last').reset_index(drop=True)
    return events


def build_episodes(events):
    episodes=[];unclosed=[]
    for (person,session),frame in events.groupby(['participant_id','session_id'],dropna=False):
        active=None
        for row in frame.to_dict('records'):
            event=row['event']
            if event=='song-start' or (event=='song-play' and active is None):
                if active is not None:
                    unclosed.append({'participant_id':person,'session_id':session,'start_event_id':active['event_id'],'reason':'next_start_without_popup'})
                active=row
            elif event in CLOSURES and active is not None and row['song_id']==active['song_id']:
                duration=row['wall_unix_s']-active['wall_unix_s']
                if duration>=0:
                    episodes.append({'episode_id':f"{person}:{session}:{active['event_id']}:{row['event_id']}",
                        'participant_id':person,'session_id':session,'song_id':row['song_id'],
                        'start_event_id':active['event_id'],'end_event_id':row['event_id'],
                        'start_kind':active['event'],'end_kind':event,
                        'wall_start_unix_s':active['wall_unix_s'],'wall_end_unix_s':row['wall_unix_s'],
                        'log_duration_s':duration})
                active=None
        if active is not None:
            unclosed.append({'participant_id':person,'session_id':session,'start_event_id':active['event_id'],'reason':'session_without_popup'})
    return pd.DataFrame(episodes),pd.DataFrame(unclosed)


def choose_episode(candidates,person,song,duration,ibi_times,duration_tolerance=1.0,boundary_tolerance=1.0):
    selected=candidates[(candidates.participant_id==person)&(candidates.song_id==song)&
                        ((candidates.log_duration_s-duration).abs()<=duration_tolerance)].copy()
    pre_ibi_count=len(selected)
    times=np.asarray(ibi_times,dtype=float)
    if len(times):
        if not np.isfinite(times).all():
            return selected.iloc[:0],pre_ibi_count
        selected=selected[(selected.wall_start_unix_s<=times.min()+boundary_tolerance)&
                          (selected.wall_end_unix_s>=times.max()-boundary_tolerance)]
    return selected,pre_ibi_count


def episode_events(events,episode):
    sub=events[(events.participant_id==episode['participant_id'])&(events.session_id==episode['session_id'])]
    after=(sub.wall_unix_s>episode['wall_start_unix_s'])|((sub.wall_unix_s==episode['wall_start_unix_s'])&(sub.event_id>=episode['start_event_id']))
    before=(sub.wall_unix_s<episode['wall_end_unix_s'])|((sub.wall_unix_s==episode['wall_end_unix_s'])&(sub.event_id<=episode['end_event_id']))
    return sub[after&before&(sub.song_id==episode['song_id'])&sub.event.isin(MOTION)].sort_values(['wall_unix_s','event_id'])


def reconstruct_exposure(frame,record_id,tolerance=1.5):
    """Represent play/pause bouts. Positions inside a bout are never invented.

    Back-inferred initial positions are diagnostics under a no-hidden-seek,
    unit-playback-rate assumption. They do not count as verified exposure.
    """
    rows=frame.to_dict('records')
    if len(rows)<2:return [],{'exposure_status':'insufficient_events','alignment_verified':False}
    start=rows[0]
    playing=True
    boundary=start
    intervals=[];zero_time_position_changes=0;state_conflicts=0
    for event in rows[1:]:
        kind=event['event']; end_time=event['wall_unix_s']
        # A redundant play event while playing is still a media-position anchor;
        # split for evidence, but it does not introduce a physiological gap.
        if kind not in {'song-play','song-pause','song-skip','song-end',*CLOSURES}:continue
        duration=end_time-boundary['wall_unix_s']
        a=boundary['media_position_s'];b=event['media_position_s']
        observed_both=np.isfinite(a) and np.isfinite(b)
        if duration<0:raise ValueError('Negative event duration')
        if duration==0:
            if observed_both and abs(b-a)>tolerance:zero_time_position_changes+=1
        else:
            item={'record_id':record_id,'interval_index':len(intervals),'state':'playing' if playing else 'paused',
                  'wall_start_unix_s':boundary['wall_unix_s'],'wall_end_unix_s':end_time,'wall_duration_s':duration,
                  'start_event_id':boundary['event_id'],'end_event_id':event['event_id'],
                  'start_event_kind':boundary['event'],'end_event_kind':kind,
                  'media_start_observed_s':a,'media_end_observed_s':b,'media_start_inferred_s':np.nan,
                  'media_delta_minus_wall_s':b-a-duration if observed_both else np.nan,
                  'alignment_verified':False}
            if not playing:
                item['position_evidence']='pause_position_changed' if observed_both and abs(b-a)>tolerance else 'pause'
            elif observed_both:
                item['position_evidence']='two_anchor_consistent' if abs(b-a-duration)<=tolerance else 'media_time_inconsistent'
            elif not np.isfinite(a) and np.isfinite(b):
                inferred=b-duration;item['media_start_inferred_s']=inferred
                item['position_evidence']='unobserved_start_zero_consistent' if abs(inferred)<=tolerance else 'unobserved_start_seek_ambiguous'
            else:item['position_evidence']='missing_media_position'
            intervals.append(item)
        if kind=='song-pause':
            if not playing:state_conflicts+=1
            playing=False
        elif kind=='song-play':
            playing=True
        elif kind in CLOSURES:
            boundary=event
            break
        elif kind in {'song-end','song-skip'}:
            playing=False
        # End/skip stops playback; any delay to survey closure remains wall-clock time.

        boundary=event
    play=[v for v in intervals if v['state']=='playing']
    pause=[v for v in intervals if v['state']=='paused']
    evidence=Counter(v['position_evidence'] for v in play)
    bad=any(v['position_evidence'] in {'media_time_inconsistent','unobserved_start_seek_ambiguous','missing_media_position'} for v in play)
    if not play:status='no_positive_play_interval'
    elif state_conflicts:status='state_sequence_conflict'
    elif bad or zero_time_position_changes:status='partial_or_seek_ambiguous'
    elif all(v['position_evidence']=='two_anchor_consistent' for v in play):status='two_anchor_consistent_not_continuously_observed'
    else:status='zero_start_consistent_under_unverified_assumption'
    return intervals,{'exposure_status':status,'playing_wall_s':sum(v['wall_duration_s'] for v in play),
        'paused_wall_s':sum(v['wall_duration_s'] for v in pause),'play_interval_count':len(play),
        'pause_interval_count':len(pause),'position_anchored_interval_count':evidence['two_anchor_consistent'],
        'zero_time_position_changes':zero_time_position_changes,'state_conflicts':state_conflicts,
        'alignment_verified':False}


def fixed_window_qc(values_seconds,timestamps,start,end,settings,min_duration=30):
    duration=end-start
    if duration<min_duration-1e-6:
        return {'rmssd_ms':np.nan,'rmssd_quality':'fail','rmssd_reason':'short_record','ibi_beats':0,
                'valid_pairs':0,'coverage':np.nan,'longest_gap_s':np.nan,'artifact_fraction':np.nan}
    return rmssd_qc(np.asarray(values_seconds)*1000,np.asarray(timestamps)-start,duration,settings)


def run(config_path):
    config_path=Path(config_path).resolve();cfg=yaml.safe_load(config_path.read_text())
    raw=ROOT/'data/raw/author_supplement_20261028';out=ROOT/'data/processed/author_v2';report=ROOT/'reports/author_v2'
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    # No arousal/valence values are loaded, including for held-out participants.
    ratings=pd.read_csv(ROOT/'data/raw/HKU956/HKU956/3. AV_ratings.csv',
        usecols=['participant_id','song_no','song_id','play_duration'],dtype={'participant_id':str,'song_id':str})
    split=json.loads((ROOT/'data/splits/participants.json').read_text())['groups']
    partition={person:part for part,people in split.items() for person in people}
    events=load_events(raw/'raw_log.download',cfg['log_timezone'])
    events.to_csv(out/'sanitized_events.csv',index=False)
    episodes,unclosed=build_episodes(events)
    episodes.to_csv(out/'candidate_log_episodes.csv',index=False);unclosed.to_csv(out/'unclosed_log_episodes.csv',index=False)
    recovered=pd.read_csv(ROOT/'reports/jamendo_recovery_all.csv',dtype={'song_id':str}).set_index('song_id')
    records=[];all_intervals=[];qc_rows=[];clock_rows=[];pair_end=[];pair_start=[];used=[]
    with zipfile.ZipFile(raw/'timestamped_ibi.download') as archive:
        names={n for n in archive.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')}
        expected={f'IBI_timestamped/{r.participant_id}/IBI/{r.song_no}_{r.song_id}.csv' for r in ratings.itertuples()}
        if expected!=names:raise ValueError('IBI archive identities differ from rating record identities')
        for row in ratings.itertuples():
            rid=f'{row.participant_id}_{row.song_no}_{row.song_id}'
            ibi=pd.read_csv(io.BytesIO(archive.read(f'IBI_timestamped/{row.participant_id}/IBI/{row.song_no}_{row.song_id}.csv')))
            if list(ibi.columns)!=['timestamp','IBI']:raise ValueError(f'Unexpected IBI columns: {rid}')
            t=ibi.timestamp.to_numpy(float);v=ibi.IBI.to_numpy(float)
            if len(t)>1:
                pair_end.extend(abs(np.diff(t)-v[1:]));pair_start.extend(abs(np.diff(t)-v[:-1]))
            original=ROOT/'data/raw/HKU956/HKU956/1. physiological_signals'/row.participant_id/'IBI'/f'{row.song_no}_{row.song_id}.csv'
            old=pd.read_csv(original,header=None).iloc[:,0].to_numpy(float) if original.stat().st_size else np.array([])
            same=np.array_equal(old,v)
            found,precount=choose_episode(episodes,row.participant_id,row.song_id,row.play_duration,t,
                cfg['log_duration_tolerance_s'],cfg['ibi_boundary_tolerance_s'])
            record={'record_id':rid,'participant_id':row.participant_id,'song_id':row.song_id,'song_no':row.song_no,
                'partition':partition[row.participant_id],'published_duration_s':row.play_duration,'ibi_raw_beats':len(t),
                'ibi_values_identical_to_original':same,'candidates_before_ibi_bounds':precount,'match_candidate_count':len(found),
                'matching_status':'unique' if len(found)==1 else ('unmatched' if not len(found) else 'ambiguous'),
                'audio_available':row.song_id in recovered.index and recovered.loc[row.song_id,'status']=='downloaded_candidate',
                'alignment_verified':False,'strict_audio_training_eligible':False}
            if len(found)!=1:
                records.append(record);continue
            episode=found.iloc[0].to_dict();record.update(episode);used.append(episode['episode_id'])
            record['duration_delta_s']=episode['log_duration_s']-row.play_duration
            if len(t):
                clock_rows.append({'record_id':rid,'participant_id':row.participant_id,'first_ibi_minus_start_s':t.min()-episode['wall_start_unix_s'],
                    'end_minus_last_ibi_s':episode['wall_end_unix_s']-t.max(),'ibi_beats':len(t)})
            normalized=pd.DataFrame({'time_s':t-episode['wall_start_unix_s'],'ibi_ms':v*1000})
            idir=out/'ibi';idir.mkdir(exist_ok=True);normalized.to_csv(idir/f'{rid}.csv',index=False)
            window_events=episode_events(events,episode)
            intervals,status=reconstruct_exposure(window_events,rid,cfg['media_consistency_tolerance_s'])
            record.update(status)
            if record['audio_available']:
                audio=recovered.loc[row.song_id];record['audio_path']=audio.local_path;record['audio_duration_s']=audio.decoded_duration_s
                for interval in intervals:
                    a=interval['media_start_observed_s'];b=interval['media_end_observed_s']
                    interval['media_bounds_within_audio']=bool(np.isfinite(a) and np.isfinite(b) and min(a,b)>=-cfg['media_consistency_tolerance_s'] and max(a,b)<=audio.decoded_duration_s+cfg['media_consistency_tolerance_s'])
            all_intervals.extend(intervals)
            for name,end in [('wall_first30',episode['wall_start_unix_s']+30),('wall_full',episode['wall_end_unix_s'])]:
                if name=='wall_first30' and episode['log_duration_s']<30:
                    end=episode['wall_end_unix_s']
                stat=fixed_window_qc(v,t,episode['wall_start_unix_s'],end,cfg['physiology'],cfg['min_analysis_duration_s'])
                pause_overlap=sum(max(0,min(x['wall_end_unix_s'],end)-max(x['wall_start_unix_s'],episode['wall_start_unix_s'])) for x in intervals if x['state']=='paused')
                qc_rows.append({'record_id':rid,'participant_id':row.participant_id,'partition':partition[row.participant_id],
                    'window':name,'window_duration_s':end-episode['wall_start_unix_s'],'pause_overlap_s':pause_overlap,
                    'ibi_original_order_valid':bool(np.isfinite(t).all() and (np.diff(t)>0).all()),**stat})
            # Supplemental physiology windows: first 30s after an observed play anchor.
            # Never concatenate intervals across pauses. These are not new independent ratings.
            anchored=[x for x in intervals if x['state']=='playing' and x['position_evidence']=='two_anchor_consistent' and x['wall_duration_s']>=30]
            for index,interval in enumerate(anchored):
                begin=interval['wall_start_unix_s'];stat=fixed_window_qc(v,t,begin,begin+30,cfg['physiology'],30)
                qc_rows.append({'record_id':rid,'participant_id':row.participant_id,'partition':partition[row.participant_id],
                    'window':f'anchored_bout30_{index}','window_duration_s':30,'pause_overlap_s':0,
                    'audio_media_start_observed_s':interval['media_start_observed_s'],
                    'exposure_verification':'endpoint_consistency_only','ibi_original_order_valid':bool(np.isfinite(t).all() and (np.diff(t)>0).all()),**stat})
            records.append(record)
    if len(used)!=len(set(used)):raise ValueError('A log episode was reused for multiple ratings')
    records=pd.DataFrame(records);intervals=pd.DataFrame(all_intervals);qc=pd.DataFrame(qc_rows)
    records.to_csv(out/'record_alignment.csv',index=False);intervals.to_csv(out/'exposure_intervals.csv',index=False)
    qc.to_csv(out/'hrv_qc_all_windows.csv',index=False);pd.DataFrame(clock_rows).to_csv(out/'clock_alignment.csv',index=False)
    # A derived table of strict physiological labels is kept separate from v1 labels.
    primary=qc[qc.window=='wall_first30'].copy();primary.to_csv(out/'hrv_first30_labels.csv',index=False)
    evidence=records.merge(primary[['record_id','rmssd_quality','rmssd_reason','coverage','valid_pairs','pause_overlap_s']],on='record_id',how='left')
    evidence['physiology_qc_pass']=evidence.rmssd_quality=='pass'
    evidence['no_logged_pause_in_first30']=evidence.pause_overlap_s==0
    evidence['conditional_media_consistency']=evidence.exposure_status.isin(['zero_start_consistent_under_unverified_assumption','two_anchor_consistent_not_continuously_observed'])
    evidence['recording_identity_basis']='author_endorsed_Jamendo_ID_source; historical_bit_identity_not_checked'
    evidence.to_csv(out/'joint_evidence_and_qc.csv',index=False)
    raw_gaps=[]
    for path in sorted((out/'ibi').glob('*.csv')):
        f=pd.read_csv(path)
        if len(f)>1:
            error=np.diff(f.time_s)-f.ibi_ms.to_numpy()[1:]/1000
            raw_gaps.append({'record_id':path.stem,'positive_gap_count':int((error>cfg['physiology']['timestamp_tolerance_s']).sum()),
                'negative_overlap_count':int((error < -cfg['physiology']['timestamp_tolerance_s']).sum()),
                'largest_positive_gap_s':float(max(0,error.max()))})
    pd.DataFrame(raw_gaps).to_csv(report/'ibi_continuity_audit.csv',index=False)
    qc_summary=qc.assign(passed=qc.rmssd_quality=='pass').groupby(['partition','window']).agg(
        records=('record_id','size'),passed=('passed','sum'),participants=('participant_id','nunique')).reset_index()
    qc_summary.to_csv(report/'hrv_qc_counts.csv',index=False)
    participant_qc=primary.assign(passed=primary.rmssd_quality=='pass').groupby(['partition','participant_id']).agg(
        records=('record_id','size'),passed=('passed','sum')).reset_index()
    participant_qc.to_csv(report/'participant_hrv_coverage.csv',index=False)
    cases=records[records.matching_status!='unique'].copy()
    records[records.start_kind=='song-play'].to_csv(report/'resume_boundary_cases.csv',index=False)
    cases.to_csv(report/'unresolved_record_matches.csv',index=False)
    write_json(report/'summary.json',{'records':len(records),'unique_matches':int((records.matching_status=='unique').sum()),
        'reused_episodes':len(used)-len(set(used)),'ibi_values_identical_to_original':int(records.ibi_values_identical_to_original.sum()),
        'empty_ibi_records':int((records.ibi_raw_beats==0).sum()),'log_events':len(events),'log_episodes':len(episodes),
        'resume_start_records':int((records.start_kind=='song-play').sum()),'duration_delta_max_s':float(records.duration_delta_s.abs().max()),
        'ibi_next_interval_match_fraction_2ms':float(np.mean(np.asarray(pair_end)<.002)),
        'ibi_previous_interval_match_fraction_2ms':float(np.mean(np.asarray(pair_start)<.002)),
        'exposure_statuses':{str(k):int(v) for k,v in records.exposure_status.value_counts().items()},
        'paused_records':int((records.paused_wall_s>0).sum()),'strict_audio_verified_records':0,
        'first30_hrv_pass':int((primary.rmssd_quality=='pass').sum()),
        'full_hrv_pass':int(((qc.window=='wall_full')&(qc.rmssd_quality=='pass')).sum()),
        'first30_pass_with_audio':int(records.record_id.isin(primary.loc[primary.rmssd_quality=='pass','record_id'])[records.audio_available].sum()),
        'config':cfg,'source_hashes':{'raw_log':digest(raw/'raw_log.download'),'timestamped_ibi_zip':digest(raw/'timestamped_ibi.download'),
            'participant_split':digest(ROOT/'data/splits/participants.json'),'audit_code':digest(Path(__file__))},
        'target_ratings_loaded':False,'models_trained':False})
    print(json.dumps(json.loads((report/'summary.json').read_text()),ensure_ascii=False,indent=2))
    return records,intervals,qc

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(ROOT/'configs/author_v2.yaml'))
    run(p.parse_args().config)
