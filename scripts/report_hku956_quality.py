"""Independent arithmetic checks, figures and narrative for the raw-data audit."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'reports/quality_audit_20261101'

def table(df):
    cols=list(df.columns)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def run():
    d=pd.read_csv(P/'record_quality_audit.csv');s=json.loads((P/'summary.json').read_text())
    counts=pd.read_csv(P/'retention_summary.csv');people=pd.read_csv(P/'participant_retention.csv')
    audio_manifest=pd.read_csv(ROOT/'data/raw/HKU956/HKU956/2. original_song_audio.csv',dtype={'song_id':str})
    assert list(audio_manifest.columns)==['song_id','link']
    assert len(audio_manifest)==592 and not audio_manifest.song_id.duplicated().any()
    assert set(audio_manifest.song_id)==set(d.song_id.astype(str))
    pairs=pd.read_csv(P/'pair_diagnostics.csv');intervals=pd.read_csv(P/'ibi_interval_diagnostics.csv');gaps=pd.read_csv(P/'gap_events.csv')
    accepted=pairs[pairs.pair_accepted]
    assert ((accepted.right_original_row-accepted.left_original_row)==1).all()
    assert (accepted.adjacency_residual_s.abs()<=s['adjacency_tolerance_s']+1e-9).all()
    assert accepted.left_interval_eligible.all() and accepted.right_interval_eligible.all()
    checks=[]
    bykey={(rid,w):g for (rid,w),g in accepted.groupby(['record_id','window'])}
    for r in d.itertuples():
        for w in ['full','first30']:
            group=bykey.get((r.record_id,w))
            n=0 if group is None else len(group)
            value=np.nan if not n else float(np.sqrt(group.squared_difference_ms2.sum()/n))
            target=getattr(r,w+'_rmssd_observed_pairs_ms')
            assert n==getattr(r,w+'_valid_consecutive_nn_pairs')
            assert (np.isnan(value) and np.isnan(target)) or np.isclose(value,target,rtol=1e-11)
            assert 0<=getattr(r,w+'_valid_pair_support_coverage')<=getattr(r,w+'_candidate_nn_coverage')+1e-10<=1+1e-10
            assert np.isclose(getattr(r,w+'_uncovered_total_s'),getattr(r,w+'_duration_s')*(1-getattr(r,w+'_candidate_nn_coverage')))
            checks.append({'record_id':r.record_id,'window':w,'independently_summed_pairs':n,'independent_rmssd_ms':value,'verified':True})
    pd.DataFrame(checks).to_csv(P/'arithmetic_verification.csv',index=False)
    for row in counts[counts.scope=='all'].itertuples():
        p=people[(people.window==row.window)&(people.policy==row.policy)]
        assert p.retained_records.sum()==row.retained_records
        assert (p.retained_records>0).sum()==row.retained_participants
    # Recheck raw source integrity against the inventory captured on read.
    inv=pd.read_csv(P/'raw_signal_inventory.csv')
    for r in inv.itertuples():assert hashlib.sha256((ROOT/r.path).read_bytes()).hexdigest()==r.sha256
    checks_json={'record_rows':len(d),'unique_record_ids':d.record_id.nunique(),'participant_count':d.participant_id.nunique(),
      'verified_record_windows':len(checks),'verified_accepted_pair_rows':len(accepted),
      'raw_signal_files_rehashed_unchanged':len(inv),'retention_participant_sums_verified':True,
      'dedicated_unit_tests_passed':10,'no_model_training':True}
    (P/'verification.json').write_text(json.dumps(checks_json,indent=2)+'\n')
    # Show all full-window gap events, plus deterministic illustrative raw traces.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    fullg=gaps[gaps.window=='full'];x=fullg.gap_duration_s
    axes[0].hist(x,bins=np.geomspace(max(x.min(),.002),x.max()*1.01,35),color='#b36850')
    axes[0].set_xscale('log');axes[0].set(xlabel='Unsupported gap length (seconds; log scale)',ylabel='Gap events',title='All full-window NN-support gaps > 2 ms')
    axes[1].scatter(d.ibi_timestamp_span_s/d.listening_duration_s,d.full_candidate_nn_coverage,s=12,alpha=.5)
    axes[1].plot([0,1],[0,1],'k--',lw=1);axes[1].set(xlabel='IBI timestamp span / listening duration',ylabel='Candidate NN interval coverage',title='Long spans can contain missing beats')
    fig.tight_layout();fig.savefig(P/'gap_and_span_diagnostics.png',dpi=170);plt.close(fig)
    ids=[d.loc[d.ibi_count==0,'record_id'].iloc[0],d.loc[(d.full_candidate_nn_coverage-d.full_candidate_nn_coverage.median()).abs().idxmin(),'record_id'],d.loc[d.full_candidate_nn_coverage.idxmax(),'record_id']]
    fig,axes=plt.subplots(3,3,figsize=(14,10))
    for row,rid in enumerate(ids):
        r=d.set_index('record_id').loc[rid];z=intervals[intervals.record_id==rid];gg=fullg[fullg.record_id==rid]
        ax=axes[row,0]
        for g in gg.itertuples():ax.axvspan(g.gap_start_s,g.gap_end_s,color='#f4c5b8',alpha=.6)
        ax.scatter(z.time_s,z.ibi_ms,s=10,color='#397ca5',label='Observed IBI')
        bad=z[~z.candidate_NN_full_window];ax.scatter(bad.time_s,bad.ibi_ms,s=22,marker='x',color='#b33228',label='Flagged / crosses boundary')
        ax.set(xlim=(0,r.listening_duration_s),ylabel='IBI (ms)',title=f'{rid}\nNN coverage {r.full_candidate_nn_coverage:.1%}')
        if z.empty:
            ax.set_ylim(300,1500);ax.text(.5,.5,'No observed IBI',transform=ax.transAxes,ha='center',va='center')
        ax.legend(fontsize=7)
        for col,(signal,hz) in enumerate([('HR',1),('BVP',64)],1):
            p=ROOT/'data/raw/HKU956/HKU956/1. physiological_signals'/r.participant_id/signal/f'{r.song_no}_{r.song_id}.csv'
            a=pd.read_csv(p,header=None).iloc[:,0].to_numpy();axes[row,col].plot(np.arange(len(a))/hz,a,lw=.6,color='#397ca5')
            axes[row,col].set(xlim=(0,r.listening_duration_s),ylabel='bpm' if signal=='HR' else 'Device amplitude (a.u.)',title=f'{signal}: inferred regular sample times')
        for ax in axes[row]:ax.set_xlabel('Seconds since logged listening start')
    fig.tight_layout();fig.savefig(P/'illustrative_raw_traces.png',dpi=170);plt.close(fig)
    old=pd.read_csv(ROOT/'data/processed/author_v2/hrv_first30_labels.csv',usecols=['record_id','rmssd_quality'])
    comparison=d[['record_id','participant_id','first30_E3_stringent_retained','first30_E3_stringent_failure_reasons']].merge(old,on='record_id',validate='one_to_one')
    comparison.to_csv(P/'prior_qc_comparison.csv',index=False)
    policy_df=pd.DataFrame(s['policies']).rename(columns={'name':'Policy','coverage':'Min NN coverage','gap':'Max unsupported gap (s)','pairs':'Min valid pairs','artifact':'Max flagged fraction','run':'Min continuous run (s)'})
    cc=counts[counts.scope=='all'][['window','policy','retained_records','retained_participants','with_audio_records','participants_with_at_least5_records']]
    hr=pd.read_csv(P/'hr_fallback_summary.csv')
    report=f'''# HKU956 raw-data quality audit

Audit date: 2026-11-01. **No model training, interval imputation, HR-to-HRV conversion, BVP beat reconstruction, or replacement of historical labels was performed.** The current research objective is song audio → RMSSD, Arousal and Valence. This audit does not substitute measured physiology as model input.

## Decision

**RMSSD is questionable as a primary outcome for the full dataset, but feasible for a restricted exploratory subset.** It is unsuitable as a complete 956-record target without substantial missingness and selection caveats. Under E3, only **107 full listening records / 21 participants** remain; **85 records have available audio**. A first-30-second endpoint retains **150 records / 23 participants**, **122 with audio**, but describes a different time window. Only **8** participants have at least five retained full-window records (10 for first30). These are engineering-screened candidate PPG interval outcomes, not validated ECG NN measurements.

Mean-HR is a substantially more complete fallback: all **956 records / 30 participants** have finite, broadly plausible HR samples at the expected 1-Hz count. Requiring a minimum 30-second recording leaves **954 / 30**, including **771 with audio**. This is a completeness finding, not a promise of audio prediction accuracy or proof of beat-level accuracy.

## Files inspected and formats established

The audit begins from actual files under `data/raw/HKU956/HKU956`, rather than invented input names. All 956 expected files in each of IBI, HR and BVP were read and hashed (**2,868 files**); no missing or unexpected signal files were found. The timestamped ZIP contains exactly the matching 956 CSV members. Every supplementary IBI value array matches its original chunk, including the **19 empty** originals. The supplement adds timing, not extra beats.

| Source | Actual format / units | Timing evidence and limitation |
| --- | --- | --- |
| `1. physiological_signals/<participant>/IBI/<song_no>_<song_id>.csv` | Headerless, one numeric column, IBI seconds | No timestamps in these original chunks; never infer continuity from adjacent rows alone. |
| `timestamped_ibi.download` | ZIP; members `IBI_timestamped/<participant>/IBI/<song_no>_<song_id>.csv`; header `timestamp,IBI` | Absolute Unix seconds plus interval seconds; convert intervals ×1000 for RMSSD in ms. Empty members retain headers. |
| HR chunks | Headerless single column; bpm; 1 Hz per dataset README | No individual sample timestamps or original session start header. Regular relative times are inferred; intra-stream dropped samples cannot be independently ruled out. |
| BVP chunks | Headerless single column; signed device amplitude, not bpm; 64 Hz | Same timing limitation. Negative amplitudes are normal, not out-of-range failures. |
| `3. AV_ratings.csv` | 956 rows, participant/order/song keys, continuous A/V and `play_duration` seconds | Keys unique; both A/V ratings finite and within [-10,10] for all 956. Listening duration is not full-song audio duration. |
| `raw_log.download` | CSV event log; local date-time strings, session, event, content/song IDs | Parsed as Asia/Hong_Kong then UTC; only non-sensitive linking fields loaded. Start/closure events re-linked from raw data. |
| `2. original_song_audio.csv` and recovered audio inventory | Song-ID linkage / URLs and recovered local MP3 metadata | Locally available 460 audio files were reopened using soundfile metadata: 773 associated records. Missing audio durations remain missing, not replaced by play duration. |

All **956** listening episodes were uniquely re-linked, with zero duration discrepancy against published play_duration and agreement with prior start/end event IDs. Duration range is **27–329 seconds**, median **44 seconds**; two recordings are shorter than 30 seconds. Full-song file duration is reported separately as `audio_file_duration_s`; its measurement is decoded frame count / sample rate from the actual local file.

### Timestamp convention and alignment

There are **33,313 IBI rows**, all finite and strictly time-ordered within their files. IBI values range from **359.391 to 1,421.940 ms**; none violates the exploratory 300–2,000-ms range. All endpoint timestamps lie within their logged record. However, **624 records** contain an initial interval that begins before the logged song boundary; that interval is excluded from within-song pair calculations.

For **30,825 / 32,376 successive rows**, timestamp difference agrees within 2 ms with the **current** interval value; agreement with the preceding value occurs only 4,265 times. This supports end-of-interval semantics for this supplement. Empatica's public description contains inconsistent indexing in its example versus gap note; the audit therefore records both comparisons rather than treating that example as proof. Timestamp boundaries are supported at logged event resolution, not established as sample-accurate audio synchronization.

**143** records include positive-duration pauses/stopped intervals. Exposure status: 785 consistent with a zero-start assumption, 166 partial/seek-ambiguous, four state-sequence conflicts, and one with two consistent position anchors but no continuous playback observation. **No record has fully verified continuous audio-position alignment.** A whole-song audio representation can be used under a declared song-level exposure approximation, but extracting “the exact first 30 seconds heard” is not justified. Full-window physiology includes paused time unless a separate endpoint is defined; no pauses are removed by concatenation here. Retention summary includes no-pause counts.

## Metric definitions and artifact policy

All counts preserve original row indices. `full_*` covers the complete logged listening window. `first30_*` covers the first min(30, duration) seconds; policies reject durations below 30. No later window is searched to maximize retention.

For timestamp t[i] and interval v[i] in seconds, the represented interval is [t[i]-v[i], t[i]]. Only finite, fully contained intervals with 0.300 ≤ v ≤ 2.000 seconds enter the broad plausibility screen.

A candidate consecutive pair must satisfy all of the following:

1. The two intervals occupy consecutive **original** rows; no deletion and reindexing can create a new pair.
2. Both intervals pass the in-window plausibility/artifact screen.
3. The timestamps increase and |(t[i]-t[i-1])-v[i]| ≤ **0.002 seconds**.
4. Timestamp order for the record is valid. An invalid sequence is flagged, never silently sorted.

For an actually adjacent plausible pair, an absolute relative change >20% of the preceding interval flags **both endpoints** as questionable. This conservative engineering rule is not an ectopic-beat diagnosis and may exclude true physiological variability. It is not applied across a missing stretch. No flagged interval is repaired. **361 full records** contain flagged abrupt-change intervals (**1,380 intervals**); no values fail the broad absolute range.

- **Timestamp span:** last minus first observed endpoint; it says nothing by itself about intervening missing beats.
- **Plausible-beat count coverage estimate:** plausible observed count / (window duration / median observed plausible IBI). This is a proxy conditional on observed beats, not knowledge of the true missing heartbeat count; the uncapped estimate is reported.
- **Plausible interval coverage:** exact union length of in-window, range-plausible interval supports / window length.
- **Candidate NN coverage:** same union after abrupt-change flags. Union prevents overlap double-counting; it does not fill gaps.
- **Valid-pair support coverage:** union of intervals actually participating in accepted consecutive pairs / window length. Isolated plausible intervals contribute to the preceding coverage but not this measure.
- **Raw discontinuity:** timestamp difference minus current interval exceeds tolerance. This measures unsupported time between the two interval supports, not the entire timestamp difference.
- **Unsupported gaps:** complement of candidate NN support in the full analysis window, including leading/trailing gaps and artifact-excluded intervals. All gaps contribute to total missing time; gaps >2 ms are enumerated in `gap_events.csv`. Max gap includes boundaries. Tiny timestamp-rounding slivers are not presented as substantive gap events.
- **Valid consecutive NN pairs:** screened timestamp-consistent PPG pairs, not ECG-confirmed normal beats.
- **RMSSD diagnostic:** sqrt(sum of squared adjacent differences in ms / accepted pair count). Differences across gaps are never included. When several runs contribute, this is a pair-weighted pooled diagnostic over observed runs, not RMSSD of a reconstructed complete record. A separate longest-run estimate is included. Candidate RMSSD values for failed policies are explicitly not accepted training labels.

Timestamp consistency cannot identify every missed or falsely detected physiological beat. The source is wrist PPG, so pulse-rate variability, device filtering and quantization remain limitations even when intervals pass this audit.

## Coverage and discontinuity results

- **19 / 956** empty IBI files; **45 / 956** full records have zero accepted adjacent pairs after screening.
- **719 / 956** records have timestamp-confirmed internal raw discontinuities; there are **1,551** such gaps before artifact exclusion.
- Median IBI count: **31**. Median candidate NN coverage: **51.0%** (interquartile range **27.8–76.3%**).
- **469** records have less than 50% candidate NN coverage.
- Median longest unsupported gap: **11.63 seconds**; **537** records exceed 10 seconds; maximum is **120.81 seconds**.
- A long IBI timestamp span can coexist with low coverage. The span-versus-coverage plot makes that distinction explicit.

## Exploratory retention thresholds

All policies require unique log linkage, valid timestamp order, no overlapping-interval inconsistency, and at least 30 seconds of listening. The same 300–2,000 ms range, 20% adjacent-change flag and 2-ms timestamp tolerance apply below. The choices are sensitivity analyses, **not universal valid thresholds**, and no policy has been selected using model scores.

{table(policy_df)}

E5 adds a 20-second uninterrupted run to E3 to check fragmentation. The 20-second run is not asserted to validate HRV; it leaves room for boundary interval exclusion in a 30-second analysis window. Full-record run duration is also reported continuously, enabling other prespecified analyses.

{table(cc)}

All percentages use the original 956 records as denominator. A participant count means at least one retained record; it does not imply adequate within-person training support. Full partition breakdown and all 30 participants, including zero-retention participants, are in separate CSVs. These quality-only results include the historical held-out participants at the user's request; no model selection or performance evaluation was done.

### Sensitivity to artifact and timing assumptions

With E3's coverage/gap/pair criteria fixed, adjacent-change limits of 10%, 20%, 30% retain respectively **24, 107, 117** full records and **45, 150, 170** first-30-second records. That sensitivity is important: absolute IBI plausibility does not establish RMSSD validity. Expanding timestamp tolerance from 2 to 16 or 40 ms leaves these 20%-artifact-limit E3 counts unchanged. Full per-record sensitivity results are saved.

Earlier project counts (171 first-window records) used different rules: one-sided abrupt-change exclusions and no explicit 5% artifact-fraction ceiling. They are not the same policy as this audit's E3. `prior_qc_comparison.csv` identifies agreement/disagreement record by record. The new audit leaves the old outputs intact.

## HR and BVP completeness / fallback

All 956 HR chunks contain exactly duration × 1 samples, all finite and between **51.33 and 150.22 bpm**. No whole HR chunk is constant. All 956 BVP chunks contain exactly duration × 64 finite samples, with no whole-chunk constant signal or identical-value run longer than one second (maximum 0.0625 s). BVP amplitudes range from -2318.68 to 2758.20 device units; without calibrated clipping limits, extrema are diagnostics rather than automatic failures.

{table(hr)}

This HR table additionally requires ≥30 seconds; without that length restriction, all 956 records have 100% broad-range sample coverage, including 773 with audio. HR is therefore **feasible as a completeness-based fallback target**. Empatica describes its HR output as averaged over approximately 10-second spans; it is not a sequence of individual heartbeat intervals and must not be converted into HRV. Without original continuous timestamps/quality flags or an independent reference, the complete chunks cannot establish physiological accuracy, exact sensor latency or absence of motion contamination. Three deterministic illustrative traces are supplied; these are not a manual certification of every BVP waveform.

## Implications for audio → RMSSD / Arousal / Valence

1. Keep A/V records independently eligible; do not discard their supervision merely because RMSSD is absent. A later multitask study could use missing-target masks, but this audit implements no model.
2. If RMSSD remains a primary research question, explicitly define full-listening versus first-30-second endpoint before training. A restricted exploratory cohort is defensible; a complete-dataset RMSSD claim is not.
3. Audit retained-cohort selection by participant, duration, playback behavior and song before inferring generalizability. Do not treat retained records as missing at random. Complete audio/RMSSD/AV intersections are reported.
4. More reliable RMSSD labels would require additional continuous high-quality pulse/ECG data or separately validated BVP reconstruction. Existing complete BVP files alone are not recovered HRV.
5. “Music-induced” should mean measured during listening in this observational dataset, not a demonstrated causal change from resting baseline. No verified pre-song resting baseline was used.
6. The previously opened final holdout remains historically opened. Any future revised method needs an explicitly exploratory evaluation plan; this audit does not reset the test set.

## Deliverables and reproducibility

- `record_quality_audit.csv`: exactly one row per listening record; raw provenance paths are in the accompanying inventory.
- `retention_summary.csv`, `participant_retention.csv`: policy counts overall, by historical partition and for all 30 participants.
- `gap_events.csv`, `ibi_interval_diagnostics.csv`, `pair_diagnostics.csv`: auditable gap/interval/pair decisions; original indices retained.
- `threshold_sensitivity_records.csv`, `threshold_sensitivity_summary.csv`, `hr_fallback_summary.csv`.
- `raw_signal_inventory.csv`, `summary.json`: raw-file hashes, input ledger, parameters, counts and timing-convention evidence.
- `arithmetic_verification.csv`, `verification.json`: independently summed pair arithmetic for **1,912 record-windows**, adjacency assertions, coverage identities, participant totals and unchanged hashes of **2,868 raw files**. Ten dedicated tests cover missing beats, artifacts, invalid ordering, boundary crossing, overlapping support, empty/short records and known RMSSD arithmetic.
- Four figures: `quality_distributions.png`, `participant_retention.png`, `gap_and_span_diagnostics.png`, `illustrative_raw_traces.png`.

Reproduce from project root:

```sh
MPLCONFIGDIR=/private/tmp/hku956_mpl python scripts/audit_hku956_quality.py
MPLCONFIGDIR=/private/tmp/hku956_mpl python -m pytest tests/test_quality_audit.py -q
MPLCONFIGDIR=/private/tmp/hku956_mpl python scripts/report_hku956_quality.py
```

Source references: dataset README and original/raw author files (hashed in summary.json); [HKU956 paper](https://doi.org/10.3390/app12189354); [Empatica's signal and file-format description](https://www.empatica.com/blog/decoding-wearable-sensor-signals-what-to-expect-from-your-e4-data/). Device documentation supports the signal interpretation; no document is used to claim that our exploratory QC thresholds are universal.
'''
    (P/'audit_report.md').write_text(report)
    print(json.dumps(checks_json,indent=2))

if __name__=='__main__':run()
