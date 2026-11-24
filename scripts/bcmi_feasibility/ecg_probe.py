"""Representative ECG feasibility only. No interval interpolation, no ML training."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import scipy.signal as sig
import neurokit2 as nk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from inventory import ROOT,OUT,D,edf_header,load_edf_channels
POLICIES=[dict(policy='exploratory_permissive',coverage=.70,gap=3.,artifact=.20,pairs=15,agreement=.70),dict(policy='exploratory_intermediate',coverage=.80,gap=2.,artifact=.10,pairs=20,agreement=.80),dict(policy='exploratory_stringent',coverage=.90,gap=1.5,artifact=.05,pairs=25,agreement=.90)]

def maxrun(mask):
 z=np.r_[False,mask,False];a=np.where(np.diff(z.astype(int))==1)[0];b=np.where(np.diff(z.astype(int))==-1)[0]
 return int((b-a).max()) if len(a) else 0

def greedy_matches(a,b,tolerance):
 matched=np.zeros(len(a),bool);i=j=0
 while i<len(a) and j<len(b):
  if abs(a[i]-b[j])<=tolerance:matched[i]=True;i+=1;j+=1
  elif a[i]<b[j]:i+=1
  else:j+=1
 return matched

def constant_spans(digital,fs,min_seconds=.5):
 mask=np.diff(digital)==0;z=np.r_[False,mask,False];a=np.where(np.diff(z.astype(int))==1)[0];b=np.where(np.diff(z.astype(int))==-1)[0]+1
 return [(start/fs,end/fs) for start,end in zip(a,b) if (end-start)/fs>=min_seconds]

def rrmetrics(peaks,onset,duration,fs,relative=.20,bad_spans=()):
 p=peaks[(peaks/fs>=onset)&(peaks/fs<onset+duration)]/fs-onset
 rr=np.diff(p);good=(rr>=.3)&(rr<=2.)
    # Do not compute NN differences through flagged constant signal sections.
 for start,end in bad_spans:
  good &= ~((p[:-1]+onset<end)&(p[1:]+onset>start))
 abrupt=good[:-1]&good[1:]&(np.abs(np.diff(rr))/np.maximum(rr[:-1],1e-12)>relative)
 flags=np.zeros(len(rr),bool)
 if len(rr)>1:flags[:-1]|=abrupt;flags[1:]|=abrupt
 good&=~flags
 pairs=good[:-1]&good[1:]
 cursor=0.;gaps=[]
 for a,b in zip(p[:-1][good],p[1:][good]):
  if a>cursor:gaps.append(a-cursor)
  cursor=b
 gaps.append(duration-cursor)
 return {'r_peaks':len(p),'rr_count':len(rr),'rr_min_ms':rr.min()*1000 if len(rr) else np.nan,'rr_max_ms':rr.max()*1000 if len(rr) else np.nan,
  'rr_outside_300_2000ms':int(((rr<.3)|(rr>2)).sum()),'candidate_nn_count':int(good.sum()),'rejected_intervals':int((~good).sum()),
  'rejected_fraction':float((~good).mean()) if len(rr) else 1.,'candidate_nn_coverage':float(rr[good].sum()/duration),
  'valid_consecutive_nn_pairs':int(pairs.sum()),'max_unsupported_gap_s':max(gaps),
  'rmssd_candidate_ms':float(np.sqrt(np.mean(np.diff(rr*1000)[pairs]**2))) if pairs.any() else np.nan},p,rr,good,pairs

def main():
 trials=pd.read_csv(OUT/'emotion_label_audit.csv');quality=[];intervals=[];peaks_out=[];runrows=[];plotcache={};sensitivity=[]
 for file in sorted(D.rglob('*.edf')):
  ds=file.relative_to(D).parts[0];rel=str(file.relative_to(D/ds));h,ch=load_edf_channels(file);x=ch['ECG']['values'];digital=ch['ECG']['digital'];fs=int(ch['ECG']['fs'])
  assert len(x)==int(h['duration_s']*fs) and np.isfinite(x).all()
  oriented,inverted=nk.ecg_invert(x,sampling_rate=fs)
  clean=nk.ecg_clean(oriented,sampling_rate=fs,method='neurokit',powerline=50)
  _,primary=nk.ecg_peaks(clean,sampling_rate=fs,method='neurokit',correct_artifacts=False)
  secondary_clean=nk.ecg_clean(oriented,sampling_rate=fs,method='hamilton2002')
  _,secondary=nk.ecg_peaks(secondary_clean,sampling_rate=fs,method='hamilton2002',correct_artifacts=False)
  p=np.asarray(primary['ECG_R_Peaks']);q=np.asarray(secondary['ECG_R_Peaks'])
  # Compare detector event timing allowing fixed method-specific QRS localization delays.
  match=greedy_matches(p,q,.15*fs)
  for idx,val in enumerate(p):peaks_out.append({'dataset':ds,'edf_file':rel,'sample_index':int(val),'time_s':val/fs,'primary_method':'neurokit','hamilton_match_within150ms':bool(match[idx])})
  runrows.append({'dataset':ds,'edf_file':rel,'duration_s':h['duration_s'],'sampling_hz':fs,'samples':len(x),'structural_missing_samples':int(h['duration_s']*fs)-len(x),
   'nonfinite_samples':int((~np.isfinite(x)).sum()),'longest_identical_run_s':(maxrun(np.diff(digital)==0)+1)/fs,
   'minimum_uV':x.min(),'maximum_uV':x.max(),'polarity_inverted':bool(inverted),'primary_peaks':len(p),'secondary_peaks':len(q),
   'detector_match_fraction':float(match.mean()),'source_discontinuities':'not_verified; plain EDF has regular grid but no original dropped-sample log'})
  bad_spans=constant_spans(digital,fs)
  subset=trials[(trials.dataset==ds)&(trials.edf_file==rel)]
  for idx,r in subset.iterrows():
   if not r.event_timing_unambiguous:continue
   a=int(round(r.onset_s*fs));b=int(round(r.end_s*fs));raw=x[a:b];cl=clean[a:b];dg=digital[a:b]
   m,t,rr,good,pair=rrmetrics(p,r.onset_s,r.analysis_duration_s,fs,bad_spans=bad_spans)
   selected=(p>=a)&(p<b);sec=(q>=a)&(q<b);nmatch=int(greedy_matches(p[selected],q[sec],.15*fs).sum());agreement=2*nmatch/(int(selected.sum())+int(sec.sum())) if selected.sum()+sec.sum() else 0.
   # EDF extrema can occur legitimately; flag repetition, not every single endpoint value.
   ci=h['labels'].index('ECG');extreme=(dg==h['digital_min'][ci])|(dg==h['digital_max'][ci]);flat=(maxrun(np.diff(dg)==0)+1)/fs
   sat=maxrun(extreme)/fs
   residual=raw-(cl if not inverted else -cl);noise_ratio=np.std(residual)/max(np.std(raw),1e-12)
   row={'dataset':ds,'participant_id':r.participant_id,'session_id':r.session_id,'run_id':r.run_id,'trial_id':r.trial_id,'trial_index':r.trial_index,
    'edf_file':rel,'onset_s':r.onset_s,'end_s':r.end_s,'duration_s':r.analysis_duration_s,'sampling_hz':fs,
    'nonfinite_fraction':float((~np.isfinite(raw)).mean()),'flatline_max_s':flat,'digital_extreme_fraction':float(extreme.mean()),'digital_extreme_max_run_s':sat,
    'raw_ecg_peak_to_peak_uV':float(np.ptp(raw)),'filter_removed_std_ratio_diagnostic':noise_ratio,
    'detector_agreement_F1_150ms':agreement,'polarity_inverted':bool(inverted),'manual_validation':'not_verified; selected high/low examples visually reviewed separately',
    'audio_lookup_available':r.documented_trial_audio_lookup,'audio_filename_target_agreement':r.audio_filename_target_agreement,
    'rating_valence_mean':float(ch['VA1']['values'][a:b].mean()),'rating_arousal_mean':float(ch['VA2']['values'][a:b].mean()),**m}
   for channel,name in [('VA1','valence'),('VA2','arousal')]:
    vals=ch[channel]['values'][a:b]
    for suffix,value in [('mean',vals.mean()),('minimum',vals.min()),('maximum',vals.max()),('fraction_outside_0_9',np.mean((vals<-.0001)|(vals>9.0001))),('unique_values',len(np.unique(vals)))]:
     trials.loc[idx,f'{name}_{suffix}']=value
   trials.loc[idx,'rating_samples_status']='verified_actual_continuous_channel_samples'
   trials.loc[idx,'numeric_values_are']='window means for feasibility illustration; aggregation not adopted as final labels'
   for policy in POLICIES:
    conditions={'coverage':m['candidate_nn_coverage']>=policy['coverage'],'gap':m['max_unsupported_gap_s']<=policy['gap'],
      'artifact':m['rejected_fraction']<=policy['artifact'],'pairs':m['valid_consecutive_nn_pairs']>=policy['pairs'],
      'detector_agreement':agreement>=policy['agreement'],'flatline':flat<.5,'sustained_extreme':sat<.05}
    row[policy['policy']+'_pass']=all(conditions.values());row[policy['policy']+'_failures']=';'.join(k for k,v in conditions.items() if not v)
   for relchange in [.1,.2,.3]:
    z=rrmetrics(p,r.onset_s,r.analysis_duration_s,fs,relchange,bad_spans=bad_spans)[0];po=POLICIES[1]
    ok=z['candidate_nn_coverage']>=po['coverage'] and z['max_unsupported_gap_s']<=po['gap'] and z['rejected_fraction']<=po['artifact'] and z['valid_consecutive_nn_pairs']>=po['pairs'] and agreement>=po['agreement'] and flat<.5 and sat<.05
    sensitivity.append({'trial_id':r.trial_id,'dataset':ds,'relative_change_limit':relchange,'intermediate_pass':ok})
   for k,interval in enumerate(rr):intervals.append({'trial_id':r.trial_id,'rr_index':k,'start_relative_s':t[k],'end_relative_s':t[k+1],'rr_ms':interval*1000,'candidate_NN':bool(good[k]),'accepted_pair_with_next':bool(pair[k]) if k<len(pair) else False})
   quality.append(row);plotcache[r.trial_id]=(raw,cl,p[selected]/fs-r.onset_s,q[sec]/fs-r.onset_s,fs,r.onset_s)
  print(ds,rel,'processed',len(subset),'trials',flush=True)
 out=pd.DataFrame(quality);out.to_csv(OUT/'ecg_quality_summary.csv',index=False);trials.to_csv(OUT/'emotion_label_audit.csv',index=False)
 pd.DataFrame(intervals).to_csv(OUT/'representative_rr_intervals.csv',index=False);pd.DataFrame(peaks_out).to_csv(OUT/'representative_rpeaks.csv',index=False)
 pd.DataFrame(runrows).to_csv(OUT/'representative_ecg_runs.csv',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'ecg_threshold_sensitivity.csv',index=False)
 summaries=[]
 for ds,g in out.groupby('dataset'):
  for po in POLICIES:
   keep=g[g[po['policy']+'_pass']]
   assert ((g.detector_agreement_F1_150ms>=0)&(g.detector_agreement_F1_150ms<=1)).all()
   summaries.append({'dataset':ds,'policy':po['policy'],'sampled_trials':len(g),'passing_trials':len(keep),'fraction':len(keep)/len(g),'sampled_participants':g.participant_id.nunique(),'passing_participants':keep.participant_id.nunique(),'fraction_not_population_estimate':True})
 pd.DataFrame(summaries).to_csv(OUT/'ecg_retention_summary.csv',index=False)
 (OUT/'ecg_probe_protocol.json').write_text(json.dumps({'neurokit_version':nk.__version__,'primary':'neurokit','secondary':'hamilton2002','detector_tolerance_ms':150,'correct_artifacts':False,'imputed_intervals':0,'rr_bounds_ms':[300,2000],'relative_change_limit':.2,'flags':'both intervals around abrupt adjacent change; RR overlapping identical-sample spans >=0.5s excluded; no concatenation','policies':POLICIES,'sampling':'first and last participant/run in lexical manifest order, selected before ECG inspection; not random representative population sampling'},indent=2))
 # Highest and lowest intermediate quality ranking per dataset; do not cherry-pick only clean traces.
 selected=[]
 for ds,g in out.groupby('dataset'):
  score=g.candidate_nn_coverage+g.detector_agreement_F1_150ms-g.rejected_fraction
  selected.extend([(ds,'lower-quality',g.loc[score.idxmin()]),(ds,'higher-quality',g.loc[score.idxmax()])])
 fig,axes=plt.subplots(4,2,figsize=(14,12))
 for i,(ds,label,row) in enumerate(selected):
  raw,cl,p,q,fs,onset=plotcache[row.trial_id];dur=len(raw)/fs
  axes[i,0].plot(np.arange(len(raw))/fs,raw,lw=.5,color='#607d8b');axes[i,0].set(title=f'{ds} {label}: {row.participant_id} trial {row.trial_index}\ncoverage={row.candidate_nn_coverage:.2f}, detector F1={row.detector_agreement_F1_150ms:.2f}',xlabel='Trial time (s)',ylabel='Raw ECG (uV)')
  # Densest detector disagreement 5-second section for the lower-quality example.
  starts=np.arange(1,max(2,int(dur)-5));scores=[abs(np.sum((p>=x)&(p<x+5))-np.sum((q>=x)&(q<x+5))) for x in starts];start=int(starts[np.argmax(scores)]) if label=='lower-quality' else 5
  end=min(start+5,dur);a=int(start*fs);b=int(end*fs);axes[i,1].plot(np.arange(a,b)/fs,cl[a:b],lw=.8,label='Cleaned ECG')
  for peaks,marker,color,name in [(p,'o','#c74735','NeuroKit'),(q,'x','#3c9d65','Hamilton')]:
   z=peaks[(peaks>=start)&(peaks<end)];iz=np.clip(np.round(z*fs).astype(int),0,len(cl)-1);axes[i,1].scatter(z,cl[iz],s=22,marker=marker,color=color,label=name)
  axes[i,1].set(xlabel='Trial time (s)',ylabel='Oriented/filtered ECG (uV)',title='Detected QRS events: visual validation required');axes[i,1].legend(fontsize=8)
 fig.tight_layout();fig.savefig(OUT/'representative_ecg_peaks.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12,7))
 for ax,(ds,label,row) in zip(axes.flat,selected):
  z=pd.DataFrame(intervals);z=z[z.trial_id==row.trial_id];ax.plot(z.end_relative_s,z.rr_ms,'.-',lw=.6);bad=z[~z.candidate_NN];ax.scatter(bad.end_relative_s,bad.rr_ms,color='red',marker='x',label='Flagged RR')
  ax.set(title=f'{ds} {label}: {row.participant_id} trial {row.trial_index}',xlabel='Trial time (s)',ylabel='Detected RR (ms)');ax.legend()
 fig.tight_layout();fig.savefig(OUT/'representative_rr_quality.png',dpi=160);plt.close(fig)
 print(pd.DataFrame(summaries).to_string(index=False))
if __name__=='__main__':main()
