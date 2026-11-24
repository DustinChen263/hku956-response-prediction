"""Inventory all channels; numerical signal inspection uses development people only.
No A/V outcomes, models, thresholds, or old experiment artifacts are changed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/paper_reassessment_20261101'
RATE={'HR':1,'BVP':64,'EDA':4,'TEMP':4}


def run():
    records=pd.read_csv(ROOT/'data/processed/author_v2/record_alignment.csv',dtype={'song_id':str})
    base=ROOT/'data/raw/HKU956/HKU956/1. physiological_signals'
    rows=[];inventory=[]
    for r in records.itertuples():
        for signal,hz in RATE.items():
            p=base/r.participant_id/signal/f'{r.song_no}_{r.song_id}.csv'
            info={'record_id':r.record_id,'participant_id':r.participant_id,'partition':r.partition,
                  'signal':signal,'exists':p.is_file(),'nonempty':p.is_file() and p.stat().st_size>0}
            inventory.append(info)
            if r.partition=='test' or not info['nonempty']:continue
            x=np.loadtxt(p,ndmin=1);finite=np.isfinite(x);valid=x[finite]
            rows.append({**info,'sampling_hz':hz,'wall_duration_s':r.published_duration_s,
                'samples':len(x),'sample_duration_s':len(x)/hz,
                'sample_duration_minus_wall_s':len(x)/hz-r.published_duration_s,
                'finite_fraction':finite.mean(),'constant_signal':bool(len(valid) and np.ptp(valid)==0),
                'zero_fraction':float(np.mean(x==0)),'negative_fraction':float(np.mean(x<0)),
                'minimum':float(np.min(valid)) if len(valid) else np.nan,
                'maximum':float(np.max(valid)) if len(valid) else np.nan,
                'mean':float(np.mean(valid)) if len(valid) else np.nan,
                'std':float(np.std(valid)) if len(valid) else np.nan})
    inventory=pd.DataFrame(inventory);df=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    inventory.to_csv(OUT/'signal_file_inventory.csv',index=False)
    df.to_csv(OUT/'development_signal_audit.csv',index=False)
    summary={}
    for signal,part in df.groupby('signal'):
        summary[signal]={'development_records':len(part),'participants':part.participant_id.nunique(),
            'nonfinite_records':int((part.finite_fraction<1).sum()),'constant_records':int(part.constant_signal.sum()),
            'records_with_negative_samples':int((part.negative_fraction>0).sum()),
            'duration_mismatch_gt_1s':int((part.sample_duration_minus_wall_s.abs()>1).sum()),
            'minimum_across_records':float(part.minimum.min()),'maximum_across_records':float(part.maximum.max()),
            'record_mean_quantiles':part['mean'].quantile([0,.25,.5,.75,1]).tolist()}
    result={'numerical_scope':'original train+validation only; no emotion values loaded',
        'all_record_inventory':inventory.groupby('signal')[['exists','nonempty']].sum().to_dict(),
        'signal_summary':summary,'these_are_diagnostics_not_validity_labels':True}
    (OUT/'signal_audit_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
