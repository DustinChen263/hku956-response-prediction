"""Assemble evidence-based feasibility deliverables; never fit models."""
from pathlib import Path
import json,hashlib,re,struct
import numpy as np
import pandas as pd
import soundfile as sf
from inventory import ROOT,OUT,S,D

def table(df):
 def fmt(v):
  if pd.isna(v):return 'not_verified'
  if isinstance(v,float):return f'{v:.4f}'
  return str(v).replace('|','/')
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in r)+' |' for r in df.itertuples(index=False,name=None))

def run():
 phase=json.loads((OUT/'phase1_summary.json').read_text());runs=pd.read_csv(OUT/'run_inventory.csv');labels=pd.read_csv(OUT/'emotion_label_audit.csv');q=pd.read_csv(OUT/'ecg_quality_summary.csv');audio=pd.read_csv(OUT/'audio_availability_audit.csv');inv=pd.read_csv(OUT/'dataset_inventory.csv')
 retention=pd.read_csv(OUT/'ecg_retention_summary.csv');sensitivity=pd.read_csv(OUT/'ecg_threshold_sensitivity.csv');full=json.loads((S/'sample_download_log.json').read_text());headers=json.loads((S/'header_download_log.json').read_text())
 full+=json.loads((S/'additional_audio_download_log.json').read_text())
 version=json.loads((S/'version_checks.json').read_text());protocol=json.loads((OUT/'ecg_probe_protocol.json').read_text())
 for ds in phase:
  assert version[ds]['version_tag_tree_sha']==version[ds]['downloaded_tree_sha']
  phase[ds]['waveform_duration_source']='WAV header declarations; not complete decode of library'
  phase[ds]['confirmed_truncated_waveform_files']=1 if ds=='ds002724' else 0
  phase[ds]['used_unique_waveforms_basis']='documented lookup only; actual heard identity not_verified'
  phase[ds]['numerically_verified_rating_trials']=36
  phase[ds]['actual_heard_audio_identity']='not_verified'
 assert len(labels)==3420 and labels.trial_id.nunique()==3420
 assert runs.groupby('dataset').trials.sum().to_dict()=={'ds002722':1692,'ds002724':1728}
 assert len(q)==72 and ((q.detector_agreement_F1_150ms>=0)&(q.detector_agreement_F1_150ms<=1)).all()
 assert labels.within_edf_duration.all()
 assert all(r['status']!='error' for r in headers)
 for record in full:
  p=D/record['dataset']/record['path'];blob=p.read_bytes()
  assert len(blob)==record['expected_bytes'] and record['md5_matches']
  pointer=(S/record['dataset']/(record['path']+'.annex_pointer')).read_text();md5=re.search(r'--([0-9a-f]{32})',pointer)[1]
  assert hashlib.md5(blob).hexdigest()==md5
 # Add actual decoded-waveform checks for downloaded audio only.
 audio['full_waveform_finite_fraction']=np.nan;audio['decoded_rms']=np.nan;audio['decoded_duration_s']=np.nan
 audio['header_declared_duration_s']=audio.duration_s;audio['riff_size_minus_manifest_bytes']=0;audio['waveform_integrity_status']='header_only_not_fully_decoded'
 for idx,r in audio.iterrows():
  header=(S/r.dataset/(r.path+'.header')).read_bytes()
  assert header[:4]==b'RIFF' and header[8:12]==b'WAVE'
  audio.loc[idx,'riff_size_minus_manifest_bytes']=struct.unpack('<I',header[4:8])[0]+8-r.bytes
  p=D/r.dataset/r.path
  if p.exists():
   x,fs=sf.read(p);assert fs==r.sample_rate_hz
   audio.loc[idx,'decoded_duration_s']=len(x)/fs
   audio.loc[idx,'waveform_integrity_status']='fully_decoded_matches_header' if len(x)==r.frames else 'published_file_truncated'
   audio.loc[idx,'status']='full_file_downloaded_MD5_verified'
   audio.loc[idx,'full_waveform_finite_fraction']=float(np.isfinite(x).mean());audio.loc[idx,'decoded_rms']=float(np.sqrt(np.mean(x*x)))
 audio['actual_heard_identity_validation']='not_verified; recorded library waveforms exist, trial linkage evaluated separately'
 audio.to_csv(OUT/'audio_availability_audit.csv',index=False)
 counts_by_run=runs.set_index(['dataset','participant_id','session_id','run_id']).trials.to_dict()
 inv['run_trial_count']=[counts_by_run.get((r.dataset,r.participant_id,r.session_id,r.run_id),np.nan) for r in inv.itertuples()]
 for idx,r in inv.iterrows():
  if (D/r.dataset/r.path).is_file():inv.loc[idx,'status']='full_file_downloaded_MD5_verified'
  elif isinstance(r.manifest_md5,str) and len(r.manifest_md5)==32 and (S/r.dataset/r.path).is_file():
   assert hashlib.md5((S/r.dataset/r.path).read_bytes()).hexdigest()==r.manifest_md5
   inv.loc[idx,'status']='small_annex_source_downloaded_MD5_verified'
 inv.to_csv(OUT/'dataset_inventory.csv',index=False)
 # Summaries distinguish a declared lookup from validated playback identity.
 labels['exact_heard_audio_identity_status']='not_verified'
 labels['quality_processing_scope']=np.where(labels.rating_samples_status=='verified_actual_continuous_channel_samples','representative_run_processed','headers_and_metadata_only')
 labels.to_csv(OUT/'emotion_label_audit.csv',index=False)
 samplemap=labels[labels.rating_samples_status=='verified_actual_continuous_channel_samples'][['dataset','participant_id','session_id','run_id','trial_index','trial_id','edf_file','onset_s','end_s','audio_file','audio_filename_target_agreement','valence_mean','arousal_mean','exact_heard_audio_identity_status']]
 samplemap.to_csv(OUT/'example_record_mapping.csv',index=False)
 # Pair arithmetic is recomputed from the saved original RR indices.
 rr=pd.read_csv(OUT/'representative_rr_intervals.csv');verified=0
 for row in q.itertuples():
  g=rr[rr.trial_id==row.trial_id].sort_values('rr_index');mask=g.accepted_pair_with_next.to_numpy(bool)
  assert not mask[-1]
  ids=np.where(mask)[0]
  for i in ids:
   assert g.candidate_NN.iloc[i] and g.candidate_NN.iloc[i+1]
   assert g.rr_index.iloc[i+1]==g.rr_index.iloc[i]+1
   assert np.isclose(g.end_relative_s.iloc[i],g.start_relative_s.iloc[i+1])
  x=g.rr_ms.to_numpy();value=np.sqrt(np.mean((x[ids+1]-x[ids])**2)) if len(ids) else np.nan
  assert len(ids)==row.valid_consecutive_nn_pairs and (np.isclose(value,row.rmssd_candidate_ms) or np.isnan(value) and np.isnan(row.rmssd_candidate_ms))
  verified+=1
 overlap=sorted(set(phase['ds002724']['participant_ids'])&set(phase['ds002722']['participant_ids']))
 c=labels[labels.dataset=='ds002722'];bad=c[c.audio_filename_target_agreement==False];zero=c[~c.event_timing_unambiguous]
 codevariation=c.dropna(subset=['stimulus_code']).groupby('stimulus_code').target_a.nunique()
 anomaly=pd.concat([c[~c.documented_trial_audio_lookup].assign(issue='missing_audio_code'),zero.assign(issue='duplicate_zero_onset'),bad.assign(issue='audio_filename_vs_target_conflict')],ignore_index=True)
 anomaly.to_csv(OUT/'mapping_anomalies.csv',index=False)
 comparison=pd.DataFrame([
 {'dataset':'BCMI Training ds002724','participants':10,'participant_sessions':24,'runs':96,'trials':1728,'waveform_library_files':216,'audio_records_before_quality':0,'audio_count_definition':'no uniquely identified trial variant','physiology':'continuous ECG','sampling_hz':1000,'analysis_window':'nominal40s; event halves21s offset20s','AV':'continuous FEELTRACE nominal0-9; 36 trials numerically checked','representative_RMSSD_intermediate':'20/36; not population yield','confirmed_training_ready_count':'not_verified','participant_overlap':'all10 also in Calibration','assessment':'C_as_released_for_exact_audio_regression; potentially recoverable with original trial log'},
 {'dataset':'BCMI Calibration ds002722','participants':19,'participant_sessions':19,'runs':94,'trials':1692,'waveform_library_files':108,'audio_records_before_quality':1662,'audio_count_definition':'documented lookup + nonzero unique onset only; mapping conflicts unresolved','physiology':'continuous ECG','sampling_hz':1000,'analysis_window':'21s events; waveform19.38-20.65s','AV':'continuous FEELTRACE0-9; 36 trials numerically checked','representative_RMSSD_intermediate':'24/36; not population yield','confirmed_training_ready_count':'not_verified','participant_overlap':'includes all10 Training participants','assessment':'B_conditional; audio lookup validation and ultra-short RMSSD interpretation unresolved'},
 {'dataset':'HKU956 existing local audit','participants':30,'participant_sessions':'not_verified','runs':'not_verified','trials':956,'waveform_library_files':460,'audio_records_before_quality':773,'audio_count_definition':'recovered song-ID-linked audio; exact listening position unverified','physiology':'intermittent PPG-derived IBI; HR/BVP also available','sampling_hz':'IBI intermittent; BVP64; HR1','analysis_window':'full listening variable27-329s; first30 alternative','AV':'post-listening continuous ratings[-10,10]','representative_RMSSD_intermediate':'not comparable policy; full audit E3:107/956,85 with audio','confirmed_training_ready_count':'85 exploratory full-window candidate RMSSD+audio, not ECG-validated','participant_overlap':'not_verified across BCMI/HKU; no evidence of overlap','assessment':'limited by IBI sparsity; larger song library'}])
 comparison.to_csv(OUT/'dataset_comparison.csv',index=False)
 summary={'assessment_date':'2026-11-02','recommendation':'B','recommendation_text':'Conditionally suitable for further feasibility work, not ready for proposed model training. Calibration is the nearer candidate; Training needs exact per-trial audio variant identifiers.',
  'dataset_assessments':{'ds002724':'C_as_released_for_exact_audio_prediction','ds002722':'B_conditional'},
  'datasets':phase,'shared_participant_ids':overlap,'distinct_participants_across_both':19,'datasets_merged':False,
  'calibration_mapping':{'listed_trials':1692,'missing_audio_code_trials':18,'duplicate_zero_onset_trials':12,'documented_lookup_and_timing_candidates':1662,'filename_target_conflicts':len(bad),'unique_stimulus_codes':len(codevariation),'target_classes_per_stimulus_code_range':[int(codevariation.min()),int(codevariation.max())],
  'exact_heard_audio_identity':'not_verified','complete_audio_ecg_AV_records_after_quality':None,'full_dataset_RMSSD_usable_count':None,'why_no_yield_estimate':'only2 participants per dataset sampled, not randomly; audio mapping unresolved; short calibration windows'},
  'representative_ecg':{'runs':4,'trials':72,'participants_per_dataset':2,'retention':retention.to_dict('records'),'interpolation_used':False,'ECG_ground_truth_peak_validation':'not_verified','full_dataset_pipeline_run':False},
  'rating_numeric_verified_trials':72,'other_rating_values':'not_verified','training_performed':False,
  'published_truncated_waveforms':['ds002724/stimuli/3-8_1.wav'],
  'downloaded_sample_file_bytes':sum(r['bytes'] for r in full),'downloaded_sample_files':len(full),'header_probes':len(headers),
  'full_library_plus_edf_bytes':sum(x['edf_bytes']+x['audio_bytes'] for x in phase.values()),
  'external_access':'OpenNeuro web page rendered no readable text; public GitHub metadata and anonymous S3 binary range/full sample downloads succeeded. Shell networking required approved read-only escalation.',
  'license':'metadata CC0 vs README CC BY4.0 conflict in both datasets; preserve attribution; redistribution terms require clarification',
  'verified_rr_arithmetic_trials':verified,'numerical_unit_tests_passed':5,'limitations':['Trial-level exact audio identity unresolved','21s Calibration RMSSD reliability not established','Non-random 4-run ECG pilot cannot estimate population usable count','Plain EDF regular grid does not prove no original acquisition gaps','Repeated stimuli and shared participants require grouped splitting']}
 (OUT/'feasibility_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 example=samplemap.groupby('dataset').head(1)[['dataset','participant_id','session_id','run_id','trial_index','onset_s','end_s','audio_file','valence_mean','arousal_mean']]
 counts=pd.DataFrame([{'Dataset':ds,'People':v['participants'],'Person-sessions':v['participant_sessions'],'Runs':v['runs'],'Listening trials':v['trials'],'WAV files':v['waveform_files'],'EDF GB':v['edf_bytes']/1e9,'WAV GB':v['audio_bytes']/1e9} for ds,v in phase.items()])
 sens=sensitivity.groupby(['dataset','relative_change_limit']).intermediate_pass.sum().rename('passing_trials_out_of36').reset_index()
 policy=pd.DataFrame(protocol['policies'])
 report=f'''# BCMI-MIdAS feasibility assessment for audio → ECG-derived RMSSD, Arousal and Valence

Assessment completed **2026-11-02**. Target releases: OpenNeuro **ds002724 v1.0.1 (Training)** and **ds002722 v1.0.1 (Calibration)**. No ML models were trained, no datasets merged, no long gaps interpolated, and no HKU956 artifacts changed. A separate environment and directory were used.

## Executive recommendation — B: conditionally suitable, not ready for training

**Do not replace HKU956 or download the entire BCMI datasets yet.** BCMI has genuine continuous ECG and available WAV libraries, and a small pilot shows that R-peak/RR processing is technically possible. However, three major issues prevent a defensible ready-to-train Audio+RMSSD+A/V cohort:

1. **Exact audio-to-trial identity:** Training only gives the two target-state codes; each transition has three different WAV versions and the used version is absent. Calibration has an explicit audio lookup, but it conflicts with target-state evidence in **1,291/1,674** code-bearing trials. This inconsistency needs an explanation or original playback logs; it is not safe to silently repair or trust a guessed mapping.
2. **Short and changing analysis windows:** Calibration is actually a **21-second event window** with approximately **20-second WAV files**, not the 40 seconds claimed by its README. Training offers nominal 40-second trials but changes affective target halfway through. A valid automatic RR sequence is not proof that a 20-second RMSSD is a reliable longer-term physiological endpoint.
3. **Limited independent support and heterogeneous ECG:** Training has **10 available people**, Calibration **19**, and all ten Training IDs occur in Calibration. Repeated stimuli and related sessions create leakage risks. Only four runs from four people were processed; automated pass rates cannot establish a full-dataset yield.

**Dataset-specific decisions:** Training is **C as currently released for exact-audio regression**, because trial waveform identity is unresolved, with potential recovery if original logs are supplied. Calibration is **B**, conditional on validating its audio-code mapping and accepting/validating the ultra-short physiological endpoint. Overall B means further targeted feasibility work is warranted, not that a suitable training cohort has already been established.

## Evidence scope and reproducibility

The assessment read the full GitHub metadata-only snapshots, **all 190 EDF headers**, **all 324 real WAV headers**, every event/channel sidecar, and the supplied Calibration MATLAB loader. Four full EDF runs and five WAV examples (including one targeted damaged-file check) were downloaded and checked against the dataset's annex MD5 identifiers. Total full sample downloads: **{summary['downloaded_sample_file_bytes']/1e6:.2f} MB**. Other recordings were not processed.

Both repository tree hashes exactly match their published `1.0.1` tags:

- ds002724: `ba70e2155c351b4369ceae37383c9e16bd9c7760`
- ds002722: `e171dcd1ccfde6fd341f0d8c096703f7e65106b0`

Anonymous public S3 requests succeeded; no dataset login, application, or user credentials were needed. OpenNeuro's main HTML returned no readable body through the text browser, so primary repository files and actual binary objects were used. Native shell networking initially failed under sandbox DNS restrictions; approved read-only network access worked. No essential check was marked unavailable merely because the web UI was dynamic.

## 1. Verified structure and counts

{table(counts)}

Counts above come from actual file inventories and events, **not the nominal counts in the papers**. Full listed repository content is about **9.150 GB** for Training and **6.546 GB** for Calibration (decimal GB, including small metadata/resource-fork files). Combined EDF+real-WAV payload is **{summary['full_library_plus_edf_bytes']/1e9:.3f} GB** before environment, decoded arrays and derived outputs. Streaming/chunked processing would avoid holding all recordings in memory.

### Training — ds002724

- Present participant folders: sub-08 through sub-17, **10 people**. README inconsistently mentions 16 and 17 participants.
- **24 participant-session combinations**, **96 runs**, 18 trials per run: **1,728 listening trials**. The 3,456 event rows are paired within trials; they are not 3,456 independent songs.
- Seven? No nominal session completion is assumed: exact counts are in `participant_session_counts.csv`. Each available session contains four runs/72 trials. Participant totals range from 72 to216, with incomplete session coverage.
- **216 real WAV files**: 72 directed transitions × three named variants. There are also 144 macOS resource-fork files ending in `.wav`; these were excluded. The 216 real files have distinct manifest hashes, which establishes file identity differences, not perceptual independence.

### Calibration — ds002722

- **19 people**, one implicit calibration session per person, **94 runs**, **1,692 listening trials**.
- Eighteen people have90 trials; sub-06 has72 trials and lacks run5. Do not calculate 19×90 and call it available data.
- **108 real WAV files**, plus108 macOS resource forks excluded. The documented event lookup references90 filenames, but this is not a validated count of unique heard stimuli.
- `sub-06/eeg/sub-06_task-run1_events.tsv` has18 target events with no paired audio-code events.
- `sub-17/eeg/sub-17_task-run2_events.tsv` has12 target/audio pairs sharing onset0, followed by six positive-time pairs. All18 trial identities are retained in the audit, but the12 zero-time trials are excluded from unambiguous temporal mapping. Do not collapse them into one trial or assign invented times.

### Licenses and access

Both `dataset_description.json` files say **CC0**; both READMEs say **CC BY4.0**, University of Reading. These are conflicting statements, not a verified single consistent license. Preserve attribution and source records, and clarify terms before redistribution. The conflict did not prevent anonymous research downloads. The actual source links and byte sizes for every listed file are in `dataset_inventory.csv`.

## 2. ECG format, timing and representative quality

All190 binary EDF headers confirm37 channels:32 EEG plus GSR, **ECG**, **VA1**, **VA2**, and **VAtarg**. ECG is channel34 in one-based indexing (33 zero-based), labeled `ECG`, with physical units `uV`. Every channel in the inspected headers has1,000 samples per1-second EDF record, giving **1,000Hz**. EDF header durations and the annex file byte sizes are structurally consistent for all190 files.

The four full files are:

- `data/raw/bcmi_feasibility/ds002724/sub-08/ses-3/eeg/sub-08_ses-3_task-run1_eeg.edf` —1,008seconds.
- `data/raw/bcmi_feasibility/ds002724/sub-17/ses-1/eeg/sub-17_ses-1_task-run4_eeg.edf` —1,031seconds.
- `data/raw/bcmi_feasibility/ds002722/sub-01/eeg/sub-01_task-run1_eeg.edf` —859seconds.
- `data/raw/bcmi_feasibility/ds002722/sub-19/eeg/sub-19_task-run5_eeg.edf` —853seconds.

Selection was the lexicographically first and last available run per dataset, fixed **before signal inspection**. This is a feasibility sample, not a random sample representative of all subjects. Each contributes18 trials, for72 total.

### Missingness and artifacts

All four files contain the declared sample count and finite ECG values. However, a plain EDF regular grid does not independently prove there were no dropped, duplicated or padded samples in original acquisition. Absolute date fields are anonymized/nonstandard; relative sample time and event onset seconds are the relevant alignment basis.

Full-run identical-sample stretches reach0.979s/1.906s in the two Calibration recordings and0.658s/0.132s in the Training recordings. Most lie outside the assessed trial windows; one assessed Training trial contains a0.637-second identical-sample stretch. All RR intervals intersecting ≥0.5-second constant stretches are excluded before constructing NN pairs. No accepted RMSSD pair bridges them.

EDF digital extrema occur very briefly in the assessed trials (maximum sustained run1ms), so a digital maximum alone is not proof of hardware saturation; it can reflect the EDF export's observed scaling range. Some displayed trials nevertheless show clear large-amplitude transients and baseline movement consistent with artifacts. Motion causation is a concern rather than independently proven by these ECG-only plots. No accelerometer reference was processed.

### R-peak/RR experiment

An isolated environment uses **NeuroKit2 0.2.12**. Each continuous run is physically scaled from EDF integer samples, polarity-oriented using `ecg_invert`, filtered with `ecg_clean(method='neurokit', powerline=50)`, and processed with `ecg_peaks(method='neurokit', correct_artifacts=False)`. A separately cleaned Hamilton2002 detector provides a sensitivity check. Detector events are compared within150ms to allow method-specific QRS localization delays; **agreement is not ground-truth R-peak accuracy**.

Observed RR intervals are differences between detected peaks. Candidate NN screening uses300–2,000ms and flags both intervals around an adjacent relative change above20%. Flagged constant sections invalidate intersecting RR intervals. RMSSD uses only adjacent surviving original RR indices, with three consecutive detected peaks wholly inside a trial. Removed intervals are never joined, and no interpolation is performed. These are **artifact-screened candidate NN intervals**, not clinically validated NN annotations.

`representative_ecg_peaks.png` shows lower- and higher-scoring examples per dataset; `representative_rr_quality.png` shows their detected RR and flagged intervals. Visual review confirms interpretable QRS sequences in some sections and substantial artifacts in others. The lower-quality Training example sub-08/trial16 contains large transients near30–33s despite fairly high detector agreement, illustrating why agreement alone cannot certify signal quality. Only selected displayed segments were visually inspected; every beat of all72 trials was **not** manually adjudicated.

### Exploratory thresholds and sample retention

All policies additionally reject a trial with a ≥0.5s constant run or ≥50ms consecutive samples at digital extremes. The policy names are descriptive engineering choices, not medical standards. The same21s/40s trial window definitions are retained; no arbitrary minimum30s is imposed to make the Calibration result disappear.

{table(policy)}

Here `coverage` is accepted interval duration / trial duration, `gap` includes unsupported boundary time, `artifact` is the rejected-interval fraction, `pairs` is the number of accepted consecutive NN pairs, and `agreement` is detector-event F1 within150ms.

{table(retention[['dataset','policy','sampled_trials','passing_trials','fraction','sampled_participants','passing_participants']])}

With the intermediate policy otherwise fixed, changing the adjacent RR relative-change flag gives:

{table(sens)}

These are **sample-level automated screening counts**, not validated labels, confidence intervals for population yield, or projections to all1,692/1,728 trials. All36 Calibration examples are shorter than30seconds. A21s result should be described as ultra-short-window RMSSD; reliability against a longer stable reference remains `not_verified`. Do not concatenate separate music trials or include intervening rating/beep periods merely to obtain longer HRV windows. Training's target change at20s also raises stationarity and label aggregation questions.

## 3. Arousal and Valence availability

The public calibration channel descriptions identify **VA1 as current felt valence**, **VA2 as current felt arousal**, both continuous FEELTRACE reports with nominal scale0–9 and neutral4.5. These are subjective participant reports, distinct from the intended target code in VAtarg or trial_type. The generic `microV` header entry for these user-report channels is not their scientific rating unit; blindly converting them as physiological voltages would scale the ratings incorrectly.

Training's sidecars identify VA1/VA2 as `UserReport` but omit the descriptive text. The interpretation is supported by the same channel convention, companion dataset, and original methods, rather than by a complete Training-specific codebook. This provenance distinction is recorded.

All190 headers contain both report channels. Actual numeric streams were checked for the72 representative trials: both channels are finite and within0–9 in those windows. This does **not** establish that all remaining trial ratings are present, responsive, or free of stale cursor values. Repeated/resampled cursor values at1kHz are not1,000 independent reports per second. No numeric values were fabricated for unprocessed runs; they remain `not_verified`.

The published experiment includes continuous reporting during listening; separate post-trial SAM labels were not located as standalone numeric files in these snapshots and remain `not_verified`. To make trial-level regression labels one must prespecify an aggregation of the continuous trajectories—mean, final segment, change, etc. The pilot means below are illustrations, not adopted final labels.

### Example mappings from actual channel samples

{table(example)}

Times are seconds from the EDF run start. The Calibration example's `hvha12.wav` comes from its documented code104 lookup; the exact-heard identity still belongs to the dataset-wide mapping investigation. The Training example's exact waveform is `not_verified`, even though its ECG and A/V can be extracted.

## 4. Audio availability and the decisive mapping issue

### Files exist: no synthesis reconstruction is required for the released libraries

All324 real WAV headers were retrieved anonymously and parsed. They are stereo44.1kHz files. Training WAV headers declare **39.950–43.917s**;201/216 declare exactly40s. These declarations are not all verified decoded durations. Calibration ranges **19.382–20.652s**, contradicting the40s README description. Five full WAV files were decoded, with finite samples and nonzero RMS, and their MD5 hashes matched the versioned annex manifests. One is damaged: `ds002724/stimuli/3-8_1.wav` declares40s but the published object is only3,145,728bytes and decodes to **17.8327s**. Its RIFF-declared size exceeds the actual object by3,910,406bytes. Because its MD5 matches the versioned manifest, this is a published truncated file, not a failed local download. Do not use it as a complete40s stimulus. The other four decoded examples match their headers; remaining files were header-checked only.

These are pre-generated libraries in the released Training/Calibration datasets. The real-time generator and unsaved outputs discussed for the separate **Testing ds002723** dataset should not be generalized to these datasets. No approximate synthesized music was substituted. MFCC/CLAP/MERT representations can technically be extracted from the available WAVs; this is not enough to justify attaching those vectors to the physiological trials.

### Training: missing variant identifier

Every run has36 event rows with only target codes1–9. Successive paired targets are exactly20seconds apart; they describe18 nominal40-second transitions. Both event rows have duration21s, so blindly summing them yields42s or a1-second overlap. The audit uses nominal40s for the pilot, explicitly marking precise playback offsets unresolved.

A transition1→2 corresponds to `stimuli/1-2_1.wav`, `1-2_2.wav`, or `1-2_3.wav`. The current events, channels and metadata do not state which variant was heard. VAtarg contains the target-state code, not the waveform variant. Thus there are **zero uniquely resolved exact-audio trial mappings in Training**, not zero downloadable audio files. A transition-class or averaged-audio representation would change the proposed research question; it has not been substituted here.

### Calibration: explicit lookup, substantial internal contradiction

Most trials have a target code1–9 and a musicPlayed code101–209 at the same onset. Their `events.json` defines a108-item list; in actual files observed codes map to90 listed filenames. We used the published list exactly, without guessing or remapping from desired affect.

There are **1,674 code-bearing trial mappings**, reduced to **1,662** with unique positive onsets after excluding12 zero-time records. Eighteen additional trials have no audio code.

However, **1,291/1,674** mappings yield a filename whose affect prefix disagrees with the recorded target code. Example: `sub-01_task-run1` trial3 has target6 (high valence/neutral arousal) but code173 maps to `nvha1.wav` (neutral valence/high arousal). In the downloaded EDF, VAtarg agrees with the event's target, so merely replacing target labels does not resolve this. Across people, each of the90 music codes is paired with **4–8 distinct target codes**.

This is an observed contradiction under the published filename semantics, **not proof of which field is wrong**. A target/filename disagreement might have an experimental explanation; the audit must not assume a corrective permutation, reorder files, or select the filename matching the target. Until original logs or author confirmation resolve it,1,662 is a **metadata-linkable candidate ceiling**, not a confirmed exact-heard Audio+ECG+A/V cohort. Even prefix-agreeing rows are not automatically validated merely by passing that consistency check.

The supplied `code/load_phBCMIcalibration.m` was downloaded and inspected; it only locates/loads EDF files and provides no additional audio-index or playback reconstruction rule.

### Musical diversity

Training's216 files arise from72 transitions with three variants; Calibration's108 files cover nine named affect classes with12 examples each. Distinct file hashes do not establish108/216 independent compositional structures; near-duplicate/similar music has not been exhaustively analyzed. The original generator literature concerns a constrained synthetic music system, not a broad commercial-song corpus. The documented Calibration lookup references90 files repeatedly across subjects. Exact heard stimulus diversity is consequently `not_verified` until mapping is resolved.

## 5. Cross-dataset compatibility

The10 Training IDs sub-08…sub-17 are also present in Calibration. The original dataset documentation explicitly states that participant codes are consistent across stages; ages/sex in the two participant tables agree for these shared IDs. Combining the datasets would give **19 distinct people, not29**.

Both public datasets have1kHz ECG in the same37-channel format and use continuous felt A/V, making technical loading compatible. They differ in window length, stage, target trajectory and available sessions. Raw RMSSD from21s and40s windows should not be pooled without defining and validating a common endpoint or reporting stage-specific analyses. Training's two-state sequence is not equivalent to Calibration's single affective target.

No automatic merge was performed. Any future evaluation must keep **all sessions and both datasets for a person in one partition**. Participant-independent splits are possible in principle but small, especially Training's10 people. Repeated stimuli across people require a separately defined stimulus-held-out or crossed participant/stimulus design for unseen-music claims. Trial count is not independent song count.

## 6. Comparison with HKU956 and feasible sample counts

`dataset_comparison.csv` contains the side-by-side audit. HKU956 local evidence:30 people,956 records,460 recovered songs covering773 records; full-window exploratory E3 produces107 candidate RMSSD records,85 with audio. Its interval source is intermittent wrist-PPG IBI, not ECG. The BCMI pilot policies are different and **their pass rates are not directly comparable** to the HKU956 full audit.

BCMI's advantage is accessible continuous ECG and simultaneous self-report channels. Its disadvantages include fewer people, short Calibration stimuli, constrained/repeated generated music, and unresolved audio linkage. It has **not yet been demonstrated to be a better end-to-end dataset for this specific audio-input thesis**, even if its ECG acquisition is more promising.

- Training has1,728 physiological trial candidates, but zero uniquely identified waveform variants at present.
- Calibration has1,692 listed trials;1,662 can be linked by the published lookup and timing before resolving mapping contradictions and whole-dataset quality.
- Actual numerical ECG/A/V checks cover only36 trials per dataset. Intermediate automatic QC passes24 Calibration and20 Training examples; these do not establish complete training-ready samples.
- **Confirmed full-dataset usable Audio+RMSSD+A/V count: `not_verified`.** No responsible population yield estimate can be obtained from this four-run, non-random pilot with unresolved input identity. Do not multiply24/36 by1,662 and present the result as an expected training cohort.

## 7. Answers to the six decision questions

1. **Better than HKU956?** Potentially better for obtaining ECG-derived intervals; not yet verified better for exact-audio→RMSSD/A/V because input linkage and duration limitations are unresolved.
2. **How many usable examples?**1,662 Calibration metadata-linked candidates is an upper bound before unresolved audio identity/quality; final usable count is `not_verified`. Training needs variant IDs before any exact-audio cohort can be counted.
3. **Enough musical diversity?** There are108 and216 downloadable library files, with heavy repeated exposure and a constrained generator. Whether this supports meaningful unseen-music regression is unproven; it does not justify a general music-recommendation claim.
4. **Reliable RMSSD technically feasible?** R-peak/RR extraction is demonstrably possible from representative1kHz ECG. Reliable artifact-screened NN and validated21s/40s endpoints remain conditional. Short-window precision and artifact sensitivity need assessment; neither detector output nor high coverage is sufficient proof.
5. **Three largest risks?** Incorrect audio-to-trial identity; ultra-short/nonstationary physiological endpoint plus artifacts; and limited/repeated participants/stimuli causing leakage and weak generalization.
6. **Download/preprocess now?** The targeted sample and metadata work is complete. Defer the remaining≈15.7GB full download and full ECG pipeline until mapping questions are answered. No model training should start now.

## 8. Next steps with explicit gates

**Gate1: resolve source identity with the authors.** Request the exact Training trial→WAV variant table; clarify Calibration musicPlayed indexing and the1291 filename/target conflicts; obtain missing sub-06 run1 audio codes and sub-17 run2 timestamps; clarify21s/40s duration and playback fade/tail rules. A draft question list is supplied below. No email has been sent.

**Gate2: expand ECG validation only after viable mappings.** Prespecify a stratified small sample across participants/runs, inspect detection failures manually, compare detector outputs, define flatline/artifact exclusion, and validate ultra-short RMSSD stability using suitable continuous reference sections without labeling nonmusic periods as music exposure. Decide whether21s RMSSD is an acceptable explicitly ultra-short endpoint. If not, Calibration is unsuitable for the intended physiological target despite usable emotion channels.

**Gate3: freeze labels and evaluation.** Choose one physiological window and an A/V aggregation rule; count complete records and independent stimulus groups after QC; group all sessions/stages by participant; define participant and stimulus generalization separately. A recommendation demo could later illustrate a model objective, but would not establish causal music-induced HRV control.

Suggested author questions:

- Which exact `A-B_1/2/3.wav` was played in each ds002724 trial, and where is that index saved?
- Does the ds002722 musicPlayed code use a participant/run-specific ordering? Why does the published static lookup disagree with filename affect codes and VAtarg in1,291 trials?
- Can original event logs restore sub-06/run1 audio IDs and sub-17/run2 zero-onset trials?
- What exactly defines playback end and fade timing in Calibration and Training, given21s events and variable WAV durations?
- Are missing Training subjects/sessions and sub-06 Calibration run5 intentionally excluded or omitted from the release?
- What is the intended license, given CC0 metadata and CC BY4.0 README, and are original ECG preprocessing/dropout logs available?

## 9. Deliverables and verification

Required files are all under `reports/bcmi_feasibility/`:

1. `feasibility_report.md` — this report.
2. `dataset_inventory.csv` — every published file with participant/session/run entities, verified name, byte size, source URL and download/inspection status.
3. `ecg_quality_summary.csv` —72 representative trial metrics, candidate RMSSD diagnostics, quality flags and all policy results.
4. `emotion_label_audit.csv` —3,420 trial rows; numerical A/V only for inspected trials, explicit `not_verified` status elsewhere.
5. `audio_availability_audit.csv` —324 real WAVs, parsed header metadata, manifest hashes, download state, and explicit heard-identity limitation.
6. `dataset_comparison.csv` —Training, Calibration, HKU956.
7. `feasibility_summary.json` —machine-readable conclusions, counts, unresolved checks and no-training status.

Additional artifacts include run/session inventories, mapping anomalies, example record mapping, representative peaks/RR, sensitivity and retention tables, figures, pinned source archives and request logs. Five numerical tests passed. Saved RR arithmetic for all72 trials was independently recomputed without joining rejected intervals. Full sample file hashes and all324 WAV RIFF sizes were checked; one published truncation is explicitly recorded. During development a detector-agreement boundary-count issue was corrected by matching within each trial; final scores are all bounded0–1. No clinical accuracy claim follows from these engineering checks.

Scripts: `scripts/bcmi_feasibility/inventory.py`, `ecg_probe.py`, `finalize.py`, `probe_access.py`, `fetch_samples.py`, `test_probe.py`. The two fetch scripts use public network access. Re-running inventory resets the numerical label audit to metadata-only; run the ECG probe and finalize afterward to restore the enriched audit. The isolated interpreter is `data/raw/bcmi_feasibility/venv/bin/python`. Existing HKU956 files and experiments were preserved.

## Sources

Primary publication interpretation (kept distinct from directly observed files):

- Daly et al. (2020), [Neural and physiological data from participants listening to affective music](https://doi.org/10.1038/s41597-020-0507-6): identifies related experimental stages, continuous felt reporting, preserved participant IDs, and pre-generated audio for Calibration/Training versus dynamic Testing. Public counts and durations were checked independently.
- Daly et al. (2016), [Affective brain–computer music interfacing](https://doi.org/10.1088/1741-2560/13/4/046022), [author manuscript](https://repository.essex.ac.uk/18769/1/draft_3-0.pdf), §2.2: Calibration uses20s plus1s fade-out; Training uses two20s target phases; ECG acquisition described at wrist sites. This helps resolve the misleading40s Calibration README, but does not resolve audio-code identities.
- Williams et al. (2017), [Affective Calibration of Musical Feature Sets](https://doi.org/10.1145/3059005): the generator reference cited by both dataset READMEs; it is not treated as an exact trial playback log or sufficient reconstruction recipe.

Actual primary datasets and implementation documentation:

- [Training v1.0.1](https://openneuro.org/datasets/ds002724/versions/1.0.1), [versioned files](https://github.com/OpenNeuroDatasets/ds002724/tree/1.0.1).
- [Calibration v1.0.1](https://openneuro.org/datasets/ds002722/versions/1.0.1), [versioned files](https://github.com/OpenNeuroDatasets/ds002722/tree/1.0.1).
- [Calibration codebook example](https://github.com/OpenNeuroDatasets/ds002722/blob/1.0.1/sub-01/eeg/sub-01_task-run1_events.json), [channel definitions](https://github.com/OpenNeuroDatasets/ds002722/blob/1.0.1/sub-01/eeg/sub-01_task-run1_channels.tsv).
- [Training event example](https://github.com/OpenNeuroDatasets/ds002724/blob/1.0.1/sub-08/ses-3/eeg/sub-08_ses-3_task-run1_events.tsv).
- [NeuroKit ECG API](https://neuropsychology.github.io/NeuroKit/functions/ecg.html), used with installed version0.2.12 and explicit correction-disabled settings.

Observed file evidence takes precedence over nominal descriptions; disagreements remain visible rather than being silently reconciled.
'''
 report=report.replace('Seven? No nominal session completion is assumed:', 'No nominal session completion is assumed:')
 (OUT/'feasibility_report.md').write_text(report)
 # Source integrity ledger includes scripts and local files, without duplicating downloaded binaries.
 ledger=[]
 for p in list(S.rglob('*'))+list((ROOT/'scripts/bcmi_feasibility').glob('*.py')):
  if p.is_file():ledger.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 pd.DataFrame(ledger).to_csv(OUT/'source_integrity.csv',index=False)
 (OUT/'verification.json').write_text(json.dumps({'expected_files_present':all((OUT/name).exists() for name in ['feasibility_report.md','dataset_inventory.csv','ecg_quality_summary.csv','emotion_label_audit.csv','audio_availability_audit.csv','dataset_comparison.csv','feasibility_summary.json']),
   'trial_rows':len(labels),'representative_rr_arithmetic_verified':verified,'full_samples_md5_verified':len(full),'wav_headers_and_size_checked':len(audio),'wav_size_mismatches':int((audio.riff_size_minus_manifest_bytes!=0).sum()),'edf_headers_verified':len(runs),'unit_tests_passed':5,'training_performed':False},indent=2))
 print(json.dumps({'recommendation':'B','datasets':{k:{x:v[x] for x in ['participants','trials','waveform_files','audio_and_timing_mapped_trials']} for k,v in phase.items()},'mapping_conflicts':len(bad),'verified_RR_trials':verified,'report':str(OUT/'feasibility_report.md')},indent=2))
if __name__=='__main__':run()
