import numpy as np
from scripts.recover_hrv import candidate_starts,search_windows,independent_rmssd,pause_overlap
from src.physiology import rmssd_qc

CFG=dict(min_ibi_ms=300,max_ibi_ms=2000,max_relative_change=.2,min_pairs=20,min_coverage=.8,max_gap_s=3.,timestamp_tolerance_s=.04)

def test_find_later_window_does_not_fill_first_window():
    values=np.tile([790.,810.],25)
    times=40+np.cumsum(values)/1000
    assert rmssd_qc(values,times,30,CFG)['rmssd_quality']=='fail'
    found,_=search_windows(values,times,85,CFG,[])
    assert found['earliest_observed30']['window_start_s']==40
    assert found['earliest_observed30']['rmssd_ms']==20

def test_gap_is_not_concatenated():
    values=np.full(30,800.)
    times=np.r_[np.arange(1,16)*.8,40+np.arange(1,16)*.8]
    assert not search_windows(values,times,60,CFG,[])[0]

def test_no_data_and_short_record():
    assert not search_windows(np.array([]),np.array([]),60,CFG,[])[0]
    assert len(candidate_starts(np.array([1.]),np.array([1000.]),29))==0

def test_paused_window_not_accepted_for_unpaused_policy():
    values=np.full(40,800.);times=np.cumsum(values)/1000
    found,_=search_windows(values,times,32,CFG,[(0,32)])
    assert 'earliest_observed30' in found and 'earliest_unpaused30' not in found
    assert pause_overlap(5,[(0,10),(30,40)])==10

def test_independent_check_respects_gaps_outliers_and_boundaries():
    values=np.array([800.,810.,790.,1500.,800.,805.,795.,810.])
    times=np.cumsum(values)/1000;times[5:]+=4
    for start in (0.,.8,1.):
        q=rmssd_qc(values,times-start,30,CFG)
        estimate,count=independent_rmssd(values,times,start,30,CFG)
        assert count==q['valid_pairs']
        assert np.isclose(estimate,q['rmssd_candidate_ms'],equal_nan=True)
