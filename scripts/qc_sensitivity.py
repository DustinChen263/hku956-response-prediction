"""Predeclared training-only label sensitivity; no performance-based threshold selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.common import parser,load_config,read_manifest,folder,write_json
from src.physiology import read_signal,mean_hr_qc
import json


def run(cfg):
    groups=json.loads((folder(cfg,'splits')/'participants.json').read_text())['groups']
    metadata=pd.read_csv(folder(cfg,'processed')/'manifest.csv',usecols=['record_id','participant_id','duration_s','hr_path'])
    train=metadata[metadata.participant_id.isin(groups['train'])]
    rows=[]
    for r in train.itertuples():
        signal=read_signal(r.hr_path,'hr')
        for window in ['first30','full']:
            duration=30 if window=='first30' else r.duration_s
            for rule,lo,hi,coverage in [('default',30,220,.8),('stricter_coverage',30,220,.9),('narrower_range',40,180,.8)]:
                settings={**cfg['physiology'],'min_hr_bpm':lo,'max_hr_bpm':hi,'min_coverage':coverage}
                result=mean_hr_qc(signal,duration,settings) if r.duration_s>=duration else {'mean_hr':np.nan,'hr_quality':'short_record','hr_coverage':0}
                rows.append({'record_id':r.record_id,'participant_id':r.participant_id,'window':window,'rule':rule,**result})
    frame=pd.DataFrame(rows)
    out=ROOT/'results/development';out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'qc_sensitivity.csv',index=False)
    summary=frame.groupby(['window','rule']).agg(valid=('mean_hr','count'),mean=('mean_hr','mean'),min_coverage=('hr_coverage','min')).reset_index()
    summary.to_csv(out/'qc_sensitivity_summary.csv',index=False)
    base=frame[frame.rule=='default'].pivot(index='record_id',columns='window',values='mean_hr').dropna()
    delta=(base.first30-base.full).abs()
    info={'population':'training participants only','records_compared':len(base),'mean_absolute_window_difference_bpm':float(delta.mean()),
          'median_absolute_window_difference_bpm':float(delta.median()),'thresholds_selected_from_results':False}
    write_json(out/'qc_window_comparison.json',info)
    print(summary.to_string(index=False));print(info)
    return summary

if __name__=='__main__':
    args=parser(__doc__).parse_args();run(load_config(args.config))
