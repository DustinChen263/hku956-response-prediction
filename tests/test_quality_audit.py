"""Invariants that protect against invented adjacency and gap bridging."""
import numpy as np
from scripts.audit_hku956_quality import window_audit,union_gaps,retention,POLICIES

def test_missing_beats_do_not_form_pairs():
    m,g,p,ok=window_audit([1,2,12,13],[1,1,1,1],14)
    assert p.tolist()==[True,False,True]
    assert m['raw_discontinuity_count']==1
    assert m['raw_discontinuity_total_s']==9
    assert m['valid_consecutive_nn_pairs']==2
    assert m['max_uncovered_gap_s']==9
    assert m['rmssd_observed_pairs_ms']==0

def test_removed_outlier_does_not_join_neighbors():
    m,g,p,ok=window_audit([1,4,5],[1,3,1],6)
    assert p.tolist()==[False,False]
    assert m['valid_consecutive_nn_pairs']==0
    assert np.isnan(m['rmssd_observed_pairs_ms'])

def test_abrupt_changes_flag_both_endpoints_only_when_adjacent():
    m,g,p,ok=window_audit([1,2.5,4],[1,1.5,1.5],5)
    assert ok.tolist()==[False,False,True]
    assert not p.any()
    m,g,p,ok=window_audit([1,10,11.5],[1,1.5,1.5],12)
    assert ok.all()  # Comparison across the gap is not an artifact test.
    assert p.tolist()==[False,True]

def test_boundary_crossing_interval_is_not_in_window_rmssd():
    m,g,p,ok=window_audit([.2,1.2,2.2],[1,1,1],3)
    assert ok.tolist()==[False,True,True]
    assert p.tolist()==[False,True]

def test_unsorted_and_duplicate_timestamps_are_not_silently_sorted():
    for t in [[1,3,2],[1,1,2]]:
        m,g,p,ok=window_audit(t,[1,1,1],4)
        assert not m['timestamp_order_valid']
        assert not p.any()
        assert m['candidate_nn_coverage']==0

def test_union_coverage_does_not_double_count_overlap():
    covered,gaps=union_gaps([(0,2),(1,3),(4,5)],6)
    assert covered==4 and gaps==[(3.,4.),(5.,6.)]

def test_empty_file_has_zero_coverage_and_full_gap():
    m,g,p,ok=window_audit([],[],30)
    assert m['candidate_nn_coverage']==0 and m['max_uncovered_gap_s']==30
    assert not retention(m,POLICIES[0])[0]

def test_rmssd_matches_manual_consecutive_pair_formula():
    v=np.array([.8,.82,.79,.81]);t=np.cumsum(v)
    m,g,p,ok=window_audit(t,v,4)
    assert p.all()
    assert np.isclose(m['rmssd_observed_pairs_ms'],np.sqrt((20**2+30**2+20**2)/3))

def test_nonfinite_and_negative_values_do_not_become_pairs():
    m,g,p,ok=window_audit([1,2,3,4],[1,np.nan,-1,1],5)
    assert not p.any()

def test_short_window_does_not_pass_even_if_all_beats_observed():
    m,*_=window_audit(np.arange(1,28),np.ones(27),27)
    ok,reason=retention(m,POLICIES[0])
    assert not ok and 'duration_lt30' in reason
