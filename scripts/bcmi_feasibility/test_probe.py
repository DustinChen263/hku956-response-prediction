"""Small numerical checks, not model training or fabricated dataset observations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from ecg_probe import rrmetrics,greedy_matches,constant_spans

def test_missed_beat_interval_not_bridged():
 m,p,rr,g,pair=rrmetrics(np.array([0,1,2,5,6,7])*1000,0,8,1000)
 assert g.tolist()==[True,True,False,True,True]
 assert pair.tolist()==[True,False,False,True]
 assert m['rmssd_candidate_ms']==0

def test_flatline_overlap_excluded_before_rmssd():
 m,p,rr,g,pair=rrmetrics(np.arange(8)*1000,0,8,1000,bad_spans=[(2.2,3.8)])
 assert not g[2] and not g[3]
 assert not pair[1:4].any()

def test_match_count_is_bounded_with_trial_boundaries():
 p=np.array([100,200,300]);q=np.array([190,290]);n=greedy_matches(p,q,20).sum()
 assert n<=min(len(p),len(q)) and 0<=2*n/(len(p)+len(q))<=1

def test_constant_span_detection():
 z=np.r_[np.arange(100),np.ones(600)*99,np.arange(100)]
 spans=constant_spans(z,1000)
 assert len(spans)==1 and spans[0][1]-spans[0][0]>=.6

def test_known_rmssd_arithmetic():
 p=np.array([0,800,1620,2410,3220])
 m,*_=rrmetrics(p,0,4,1000)
 assert m['valid_consecutive_nn_pairs']==3
 assert np.isclose(m['rmssd_candidate_ms'],np.sqrt((400+900+400)/3))
