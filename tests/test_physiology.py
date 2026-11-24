import numpy as np
from src.physiology import rmssd_qc, mean_hr_qc, read_signal


def qc(**overrides):
    return dict(min_ibi_ms=300,max_ibi_ms=2000,max_relative_change=.5,min_pairs=1,min_coverage=0,max_gap_s=100,**overrides)


def test_known_rmssd():
    v = np.array([800.,810.,830.,820.])
    t = np.cumsum(v/1000)
    result = rmssd_qc(v,t,float(t[-1]),qc())
    assert np.isclose(result['rmssd_ms'],np.sqrt((100+400+100)/3))
    assert result['valid_pairs'] == 3


def test_do_not_bridge_rejected_beat():
    v = np.array([800.,810.,3000.,820.,830.])
    t = np.cumsum(v/1000)
    result = rmssd_qc(v,t,float(t[-1]),qc())
    # 820 follows an outlier and is also conservatively rejected by local change rule.
    assert result['valid_pairs'] == 1
    assert result['rmssd_ms'] == 10


def test_do_not_bridge_timestamp_gap():
    v = np.array([800.,810.,820.,830.])
    t = np.cumsum(v/1000)
    t[2:] += 5
    result = rmssd_qc(v,t,float(t[-1]),qc())
    assert result['valid_pairs'] == 2
    assert result['longest_gap_s'] >= 5 - 1e-8


def test_untimed_ibi_never_becomes_valid_label():
    result = rmssd_qc([800,810,820],None,30,qc())
    assert np.isnan(result['rmssd_ms'])
    assert result['rmssd_reason'] == 'missing_timestamps'


def test_hr_is_not_rmssd():
    result = mean_hr_qc({'values':np.array([60.,70.,80.]),'times':None},3,
                        {'min_hr_bpm':30,'max_hr_bpm':220,'hr_sample_rate':1,'min_coverage':.8})
    assert result['mean_hr'] == 70
    assert 'rmssd_ms' not in result


def test_file_read_preserves_first_numeric_sample(tmp_path):
    p = tmp_path/'ibi.csv'
    p.write_text('0.8\n0.9\n1.0\n')
    assert len(read_signal(p,'ibi')['values']) == 3


def test_nonmonotonic_times_fail():
    assert rmssd_qc([800,810,820],[1,2,1.5],30,qc())['rmssd_reason'] == 'invalid_timestamps'
