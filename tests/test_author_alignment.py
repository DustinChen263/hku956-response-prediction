import numpy as np
import pandas as pd
from scripts.author_alignment import media_fields,load_events,build_episodes,choose_episode,reconstruct_exposure,fixed_window_qc


def event(index,t,kind,pos=np.nan,person='p',session='s',song='1'):
    return {'event_id':index,'wall_unix_s':float(t),'event':kind,'media_position_s':pos,
            'participant_id':person,'session_id':session,'song_id':song,'played_counter_s':np.nan}


def test_parse_media_anchor_without_treating_played_as_position():
    position,played=media_fields('Example || played 114s || current 114.390204s || 1.00')
    assert position==114.390204 and played==114
    assert np.isnan(media_fields('Example title')[0])


def test_resume_without_start_is_a_distinct_observed_boundary():
    frame=pd.DataFrame([event(1,0,'song-start',session='old'),event(2,3600,'song-play',13,session='new'),
                        event(3,3668,'survey-popup',81,session='new')])
    episodes,unclosed=build_episodes(frame)
    assert len(episodes)==1
    assert episodes.iloc[0].start_kind=='song-play'
    assert episodes.iloc[0].wall_start_unix_s==3600
    assert len(unclosed)==1


def test_repeated_song_is_resolved_by_ibi_time_not_first_match():
    candidates=pd.DataFrame([{'participant_id':'p','song_id':'1','log_duration_s':40,'wall_start_unix_s':0,'wall_end_unix_s':40},
                             {'participant_id':'p','song_id':'1','log_duration_s':40,'wall_start_unix_s':100,'wall_end_unix_s':140}])
    result,n=choose_episode(candidates,'p','1',40,[101,139])
    assert n==2 and len(result)==1 and result.iloc[0].wall_start_unix_s==100


def test_empty_ibi_and_ambiguous_duration_is_not_forced_to_match():
    candidates=pd.DataFrame([{'participant_id':'p','song_id':'1','log_duration_s':40,'wall_start_unix_s':0,'wall_end_unix_s':40},
                             {'participant_id':'p','song_id':'1','log_duration_s':40,'wall_start_unix_s':100,'wall_end_unix_s':140}])
    result,_=choose_episode(candidates,'p','1',40,[])
    assert len(result)==2


def test_pause_time_is_not_concatenated_as_audio():
    frame=pd.DataFrame([event(1,0,'song-start'),event(2,26,'song-pause',25.63),event(3,33,'song-play',25.69),event(4,122,'survey-popup-onend',114.39)])
    intervals,status=reconstruct_exposure(frame,'r')
    assert status['paused_wall_s']==7
    assert status['playing_wall_s']==115
    assert sum(i['wall_duration_s'] for i in intervals)==122
    assert intervals[-1]['media_start_observed_s']==25.69
    assert intervals[-1]['position_evidence']=='two_anchor_consistent'
    assert status['alignment_verified'] is False


def test_initial_position_is_never_silently_set_to_zero():
    frame=pd.DataFrame([event(1,0,'song-start'),event(2,10,'survey-popup',78)])
    intervals,status=reconstruct_exposure(frame,'r')
    assert np.isnan(intervals[0]['media_start_observed_s'])
    assert intervals[0]['media_start_inferred_s']==68
    assert status['exposure_status']=='partial_or_seek_ambiguous'
    assert not status['alignment_verified']


def test_observed_rewind_breaks_continuity_evidence():
    frame=pd.DataFrame([event(1,0,'song-play',20),event(2,10,'survey-popup',5)])
    intervals,status=reconstruct_exposure(frame,'r')
    assert intervals[0]['position_evidence']=='media_time_inconsistent'
    assert status['exposure_status']=='partial_or_seek_ambiguous'


def test_zero_time_seek_is_recorded_without_creating_duration():
    frame=pd.DataFrame([event(1,0,'song-play',0),event(2,10,'song-pause',10),
                        event(3,10,'song-play',60),event(4,40,'survey-popup',90)])
    intervals,status=reconstruct_exposure(frame,'r')
    assert status['zero_time_position_changes']==1
    assert sum(i['wall_duration_s'] for i in intervals)==40
    assert status['paused_wall_s']==0


def test_hrv_window_starts_at_log_boundary_not_first_detected_beat():
    values=np.ones(10)*.8;times=1000+20+np.arange(10)*.8
    cfg={'min_ibi_ms':300,'max_ibi_ms':2000,'max_relative_change':.2,'min_pairs':2,'min_coverage':.8,'max_gap_s':3}
    result=fixed_window_qc(values,times,1000,1030,cfg)
    assert result['rmssd_quality']=='fail'
    assert result['coverage']<.3
    assert result['longest_gap_s']>15


def test_short_window_cannot_be_promoted_to_30_seconds():
    result=fixed_window_qc([.8,.8],[1,2],0,27,{},min_duration=30)
    assert result['rmssd_reason']=='short_record'


def test_raw_identifiers_not_exported_by_event_loader(tmp_path):
    p=tmp_path/'log.csv'
    p.write_text('timestamp,session_id,element_clicked,content_id,content_name,id,username,IP_address,uid\n'
                 '2019-08-22 16:39:12,s,song-start,1,Example,1,p,PRIVATE_IP,PRIVATE_ID\n')
    events=load_events(p,'Asia/Hong_Kong')
    assert 'IP_address' not in events and 'uid' not in events and 'content_name' not in events
    assert events.iloc[0].wall_unix_s==1566463152


def test_song_end_to_survey_delay_remains_wall_clock_pause():
    frame=pd.DataFrame([event(1,0,'song-play',0),event(2,40,'song-end',40),event(3,45,'survey-popup-onend',40)])
    intervals,status=reconstruct_exposure(frame,'r')
    assert status['playing_wall_s']==40
    assert status['paused_wall_s']==5
    assert sum(i['wall_duration_s'] for i in intervals)==45


def test_song_skip_is_not_used_as_an_unlogged_seek_target():
    frame=pd.DataFrame([event(1,0,'song-play',10),event(2,40,'song-skip',50),event(3,42,'survey-popup',50)])
    intervals,status=reconstruct_exposure(frame,'r')
    assert status['playing_wall_s']==40 and status['paused_wall_s']==2
    assert intervals[-1]['state']=='paused'
