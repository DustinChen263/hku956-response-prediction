from pathlib import Path
import json,re,hashlib,wave
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/bcmi_feasibility';S=OUT/'sources';D=ROOT/'data/raw/bcmi_feasibility'

def edf_header(path):
 b=Path(path).read_bytes()[:16384];n=int(b[252:256]);pos=256
 h={'header_bytes':int(b[184:192]),'n_records':int(b[236:244]),'record_duration_s':float(b[244:252]),'n_channels':n,'reserved':b[192:236].decode().strip()}
 for key,width,typ in [('labels',16,str),('transducers',80,str),('units',8,str),('physical_min',8,float),('physical_max',8,float),('digital_min',8,int),('digital_max',8,int),('prefilter',80,str),('samples_per_record',8,int),('channel_reserved',32,str)]:
  h[key]=[typ(b[pos+i*width:pos+(i+1)*width].decode().strip()) for i in range(n)];pos+=n*width
 h['duration_s']=h['n_records']*h['record_duration_s'];h['expected_bytes']=h['header_bytes']+2*h['n_records']*sum(h['samples_per_record'])
 return h

def load_edf_channels(path,names=('ECG','VA1','VA2','VAtarg')):
 h=edf_header(path);a=np.memmap(path,dtype='<i2',mode='r',offset=h['header_bytes'],shape=(h['n_records'],sum(h['samples_per_record'])))
 result={};offsets=np.r_[0,np.cumsum(h['samples_per_record'])]
 for name in names:
  i=h['labels'].index(name);digital=np.array(a[:,offsets[i]:offsets[i+1]]).reshape(-1)
  physical=(digital.astype(float)-h['digital_min'][i])*(h['physical_max'][i]-h['physical_min'][i])/(h['digital_max'][i]-h['digital_min'][i])+h['physical_min'][i]
  result[name]={'digital':digital,'values':physical,'fs':h['samples_per_record'][i]/h['record_duration_s'],'unit':h['units'][i]}
 return h,result

def run():
 inv=[];aud=[];runs=[];trials=[];summ={}
 for ds in ['ds002724','ds002722']:
  root=S/ds;tree=json.loads((S/f'{ds}_tree.json').read_text());files=[x for x in tree['tree'] if x['type']=='blob'];wavs={}
  for x in files:
   path=x['path'];p=root/path;pointer=root/(path+'.annex_pointer');size=x['size'];md5='';status='metadata_downloaded'
   if pointer.exists():
    m=re.search(r'MD5E-s(\d+)--([0-9a-f]{32})',pointer.read_text());size=int(m[1]) if m else None;md5=m[2] if m else '';status='annex_pointer_only'
   actualwav=path.startswith('stimuli/') and path.endswith('.wav') and '__MACOSX' not in path
   header=root/(path+'.header')
   if header.exists():status='binary_header_verified'
   if (D/ds/path).exists():status='full_file_downloaded_MD5_verified'
   person=re.search(r'(sub-\d+)',path);session=re.search(r'(ses-\d+)',path);runid=re.search(r'(task-run\d+)',path)
   inv.append({'dataset':ds,'git_tree_sha':tree['sha'],'path':path,'participant_id':person[1] if person else '',
    'session_id':session[1] if session else ('implicit_calibration_session' if person and ds=='ds002722' else ''),
    'run_id':runid[1] if runid else '', 'file_kind':'audio_waveform' if actualwav else 'metadata_resource_fork' if '__MACOSX' in path else Path(path).suffix,
    'bytes':size,'manifest_md5':md5,'status':status,'source_url':f'https://github.com/OpenNeuroDatasets/{ds}/blob/{tree["sha"]}/{path}',
    'download_url':f'https://s3.amazonaws.com/openneuro.org/{ds}/{path}' if pointer.exists() else f'https://raw.githubusercontent.com/OpenNeuroDatasets/{ds}/{tree["sha"]}/{path}'})
   if actualwav:
    with wave.open(str(header)) as w:info={'sample_rate_hz':w.getframerate(),'channels':w.getnchannels(),'sample_width_bytes':w.getsampwidth(),'frames':w.getnframes(),'duration_s':w.getnframes()/w.getframerate()}
    wavs[Path(path).name]=info
    aud.append({'dataset':ds,'path':path,'bytes':size,'manifest_md5':md5,'status':status,**info,
     'exact_waveform_file_present':True,'deterministic_reconstruction_needed':False,'trial_mapping_status':'explicit_event_code_lookup' if ds=='ds002722' else 'ambiguous_three_variants_per_transition',
     'source_url':f'https://s3.amazonaws.com/openneuro.org/{ds}/{path}'})
  for f in sorted(root.rglob('*_events.tsv')):
   path=str(f.relative_to(root));edf=path.replace('_events.tsv','_eeg.edf');h=edf_header(root/(edf+'.header'));ev=pd.read_csv(f,sep='\t')
   who=path.split('/')[0];session=re.search(r'(ses-\d+)',path);session=session[1] if session else 'implicit_calibration_session';runid=re.search(r'(task-run\d+)',path)[1]
   ch=pd.read_csv(root/path.replace('_events.tsv','_channels.tsv'),sep='\t');meta=json.loads((root/path.replace('_events.tsv','_events.json')).read_text())
   if ds=='ds002722':
    text=meta.get('101-209',{}).get('Description','');lookup=re.findall(r"'([^']+\.wav)'",text)
    # The narrative repeats hvha12 before the actual list; take the final 108 entries.
    lookup=lookup[-108:];assert len(lookup)==108 and len(set(lookup))==108
    windows=[]
    # Preserve original event order, including corrupt duplicate zero timestamps.
    for idx,row in ev.iterrows():
     if not 1<=row.trial_type<=9:continue
     target=int(row.trial_type);onset=float(row.onset)
     nxt=ev.iloc[idx+1] if idx+1<len(ev) else None
     has_code=nxt is not None and nxt.trial_type>100 and nxt.onset==onset
     code=int(nxt.trial_type) if has_code else None
     name=lookup[code-101] if code is not None and 101<=code<=208 else None
     name_ok=name in wavs if name else False
     time_ok=onset>0 and int(((ev.trial_type.between(1,9))&(ev.onset==onset)).sum())==1
     prefix={'lvla':1,'nvla':2,'hvla':3,'lvna':4,'nvna':5,'hvna':6,'lvha':7,'nvha':8,'hvha':9}
     target_match=prefix.get(name[:4])==target if name else None
     windows.append({'onset_s':onset,'analysis_duration_s':21.,'target_a':target,'target_b':None,'stimulus_code':code,
       'audio_file':f'stimuli/{name}' if name_ok else 'not_verified','audio_candidates':f'stimuli/{name}' if name_ok else 'not_verified',
       'documented_trial_audio_lookup':bool(name_ok),'event_timing_unambiguous':bool(time_ok),
       'audio_filename_target_agreement':target_match,
       'waveform_duration_s':wavs[name]['duration_s'] if name_ok else None,'duration_status':'event_duration_21s; waveform_duration_separate'})
   else:
    assert len(ev)==36 and np.allclose(ev.onset.to_numpy()[1::2]-ev.onset.to_numpy()[::2],20)
    windows=[]
    for i in range(0,len(ev),2):
     a=int(ev.trial_type.iloc[i]);b=int(ev.trial_type.iloc[i+1]);candidates=[name for name in wavs if name.startswith(f'{a}-{b}_')]
     assert len(candidates)==3
     windows.append({'onset_s':float(ev.onset.iloc[i]),'analysis_duration_s':40.,'target_a':a,'target_b':b,'stimulus_code':None,'audio_file':'not_verified',
       'audio_candidates':';'.join('stimuli/'+n for n in sorted(candidates)),'documented_trial_audio_lookup':False,'event_timing_unambiguous':True,'audio_filename_target_agreement':None,'waveform_duration_s':None,'duration_status':'nominal40s; two21s events overlap1s; exact offset not_verified'})
   ci=h['labels'].index('ECG');rate=h['samples_per_record'][ci]/h['record_duration_s']
   ptr=(root/(edf+'.annex_pointer')).read_text();size=int(re.search(r'MD5E-s(\d+)',ptr)[1]);assert size==h['expected_bytes']
   runs.append({'dataset':ds,'participant_id':who,'session_id':session,'run_id':runid,'edf_file':edf,'trials':len(windows),'event_rows':len(ev),'edf_duration_s':h['duration_s'],
    'sampling_hz':rate,'ecg_channel':'ECG','ecg_unit':h['units'][ci],'channels':h['n_channels'],'expected_bytes':size,'header_size_consistent':True,
    'relative_times_available':True,'absolute_date_validated':False,'full_recording_downloaded':(D/ds/edf).exists()})
   for j,w in enumerate(windows,1):
    trials.append({'dataset':ds,'participant_id':who,'session_id':session,'run_id':runid,'trial_index':j,
     'trial_id':f'{ds}_{who}_{session}_{runid}_{j:02d}','edf_file':edf,'events_file':path,'sampling_hz':rate,
     'end_s':w['onset_s']+w['analysis_duration_s'],'within_edf_duration':w['onset_s']+w['analysis_duration_s']<=h['duration_s'],
     'valence_channel':'VA1','arousal_channel':'VA2','rating_type':'continuous_listener_felt_emotion_FEELTRACE',
     'scale_semantics':'0-9; neutral4.5; microV header is generic, not rating unit',
     'rating_samples_status':'downloaded_pending_numerical_audit' if (D/ds/edf).exists() else 'not_verified',
     'post_trial_SAM_labels':'not_verified','source_url':f'https://github.com/OpenNeuroDatasets/{ds}/blob/{tree["sha"]}/{path}',**w})
  selected=[r for r in inv if r['dataset']==ds];rr=[r for r in runs if r['dataset']==ds];tt=[r for r in trials if r['dataset']==ds];aa=[r for r in aud if r['dataset']==ds]
  summ[ds]={'git_tree_sha':tree['sha'],'version_doi':json.loads((root/'dataset_description.json').read_text())['DatasetDOI'],
   'participants':len({r['participant_id'] for r in rr}),'participant_ids':sorted({r['participant_id'] for r in rr}),
   'participant_sessions':len({(r['participant_id'],r['session_id']) for r in rr}),'runs':len(rr),'trials':len(tt),'event_rows':sum(r['event_rows'] for r in rr),
   'waveform_files':len(aa),'unique_waveform_manifest_hashes':len({r['manifest_md5'] for r in aa}),'used_unique_waveforms':len({r['audio_file'] for r in tt if r['documented_trial_audio_lookup']}),
   'documented_audio_mapped_trials':sum(r['documented_trial_audio_lookup'] for r in tt),
   'audio_and_timing_mapped_trials':sum(r['documented_trial_audio_lookup'] and r['event_timing_unambiguous'] for r in tt),
   'ambiguous_timing_trials':sum(not r['event_timing_unambiguous'] for r in tt),
   'audio_filename_target_mismatch_trials':sum(r['audio_filename_target_agreement'] is False for r in tt),'edf_bytes':sum(r['expected_bytes'] for r in rr),'audio_bytes':sum(r['bytes'] for r in aa),
   'repository_content_bytes':sum(r['bytes'] or 0 for r in selected),'sampling_rates':sorted({r['sampling_hz'] for r in rr}),
   'waveform_duration_range_s':[min(r['duration_s'] for r in aa),max(r['duration_s'] for r in aa)],
   'license_metadata':'CC0','license_README':'CC BY 4.0','license_status':'conflicting_statements; retain attribution and clarify redistribution'}
 pd.DataFrame(inv).to_csv(OUT/'dataset_inventory.csv',index=False);pd.DataFrame(aud).to_csv(OUT/'audio_availability_audit.csv',index=False)
 pd.DataFrame(runs).to_csv(OUT/'run_inventory.csv',index=False);pd.DataFrame(trials).to_csv(OUT/'emotion_label_audit.csv',index=False)
 pd.DataFrame(trials).groupby(['dataset','participant_id','session_id']).size().rename('trials').reset_index().to_csv(OUT/'participant_session_counts.csv',index=False)
 (OUT/'phase1_summary.json').write_text(json.dumps(summ,indent=2));print(json.dumps(summ,indent=2))
if __name__=='__main__':run()
