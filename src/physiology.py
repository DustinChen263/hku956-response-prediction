"""Conservative interval quality control. Never bridge a removed beat or a gap."""
from pathlib import Path
import numpy as np
import pandas as pd


def read_signal(path, kind):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    first = path.open().readline().strip()
    try:
        float(first.split(',')[0])
        numeric = True
    except ValueError:
        numeric = False
    if numeric:
        values = pd.read_csv(path, header=None).to_numpy(float)
        if values.shape[1] != 1:
            raise ValueError(f'{path}: numeric files must have one column; use named columns for timed signals')
        return {'values': values[:, 0], 'times': None}
    df = pd.read_csv(path)
    value_col = 'ibi_ms' if kind == 'ibi' else 'hr_bpm'
    if value_col not in df or 'time_s' not in df:
        raise ValueError(f'{path}: expected time_s,{value_col}')
    return {'values': df[value_col].to_numpy(float), 'times': df.time_s.to_numpy(float)}


def rmssd_qc(values_ms, times_s, duration_s, cfg):
    values = np.asarray(values_ms, dtype=float)
    missing = dict(rmssd_ms=np.nan, rmssd_candidate_ms=np.nan, ibi_beats=len(values),
                   valid_pairs=0, coverage=np.nan, longest_gap_s=np.nan, artifact_fraction=np.nan,
                   rmssd_quality='fail', rmssd_reason='missing_timestamps')
    if times_s is None:
        # A diagnostic candidate is intentionally not exposed as a valid target.
        valid = np.isfinite(values) & (values >= cfg['min_ibi_ms']) & (values <= cfg['max_ibi_ms'])
        pairs = valid[:-1] & valid[1:]
        if pairs.any():
            missing['rmssd_candidate_ms'] = float(np.sqrt(np.mean(np.diff(values)[pairs] ** 2)))
        return missing
    times = np.asarray(times_s, dtype=float)
    if len(times) != len(values) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        missing['rmssd_reason'] = 'invalid_timestamps'
        return missing
    # End-of-interval timestamps. Only fully contained intervals contribute.
    starts = times - values / 1000
    within = (starts >= -1e-6) & (times <= duration_s + 1e-6)
    values, times, starts = values[within], times[within], starts[within]
    if len(values) < 2:
        missing['rmssd_reason'] = 'too_few_beats'
        return missing
    valid = np.isfinite(values) & (values >= cfg['min_ibi_ms']) & (values <= cfg['max_ibi_ms'])
    # Conservative local artifact rule, preserving original indices.
    relative = np.abs(np.diff(values)) / np.maximum(np.abs(values[:-1]), 1)
    valid[1:] &= relative <= cfg['max_relative_change']
    # Consecutive timestamps must agree with the next interval length within device tolerance.
    contiguous = np.abs(np.diff(times) - values[1:] / 1000) <= cfg.get('timestamp_tolerance_s', 0.04)
    pairs = valid[:-1] & valid[1:] & contiguous
    covered = float(np.sum(values[valid]) / 1000)
    # Gaps include invalid beats and leading/trailing missing segments.
    cursor, longest = 0.0, 0.0
    for start, end in zip(starts[valid], times[valid]):
        longest = max(longest, float(start - cursor))
        cursor = max(cursor, float(end))
    longest = max(longest, duration_s - cursor)
    reasons = []
    if pairs.sum() < cfg['min_pairs']:
        reasons.append('too_few_pairs')
    if covered / duration_s < cfg['min_coverage']:
        reasons.append('low_coverage')
    if longest > cfg['max_gap_s']:
        reasons.append('long_gap')
    estimate = float(np.sqrt(np.mean(np.diff(values)[pairs] ** 2))) if pairs.any() else np.nan
    return dict(rmssd_ms=estimate if not reasons else np.nan, rmssd_candidate_ms=estimate,
                ibi_beats=len(values), valid_pairs=int(pairs.sum()), coverage=min(covered / duration_s, 1.0),
                longest_gap_s=longest, artifact_fraction=float(1 - valid.mean()),
                rmssd_quality='pass' if not reasons else 'fail', rmssd_reason=';'.join(reasons))


def mean_hr_qc(signal, duration_s, cfg):
    if signal is None:
        return {'mean_hr': np.nan, 'hr_quality': 'missing', 'hr_coverage': 0.0}
    vals, times = signal['values'], signal['times']
    if times is None:
        times = np.arange(len(vals)) / cfg['hr_sample_rate']
    elif not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        return {'mean_hr': np.nan, 'hr_quality': 'invalid_timestamps', 'hr_coverage': 0.0}
    vals = vals[(times >= 0) & (times < duration_s)]
    valid = np.isfinite(vals) & (vals >= cfg['min_hr_bpm']) & (vals <= cfg['max_hr_bpm'])
    coverage = min(float(valid.sum() / (duration_s * cfg['hr_sample_rate'])), 1.0)
    ok = valid.any() and coverage >= cfg['min_coverage']
    return {'mean_hr': float(vals[valid].mean()) if ok else np.nan,
            'hr_quality': 'pass' if ok else 'low_coverage', 'hr_coverage': coverage}
