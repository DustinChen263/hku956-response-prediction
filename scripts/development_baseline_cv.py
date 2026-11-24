"""Mean and song-mean diagnostics on five development-only participant folds."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.development_analysis import load_development,TARGETS
from src.common import parser,load_config,folder
from src.evaluate import summarize


def run(cfg):
    data,groups=load_development(cfg)
    data=data.dropna(subset=TARGETS)
    definitions=json.loads((folder(cfg,'splits')/'participants.json').read_text())['cv']
    predictions=[];heldout=[]
    for definition in definitions:
        split=definition['groups'];fold=definition['fold']
        assert not set(sum(split.values(),[]))&set(groups['test'])
        # Same inner training people as neural CV; inner validation is not refit here.
        fit=data[data.participant_id.isin(split['train'])]
        evaluate=data[data.participant_id.isin(split['test'])]
        heldout+=split['test']
        for target in TARGETS:
            mean=fit[target].mean();songs=fit.groupby('song_id')[target].mean()
            for model in ['mean','song_mean']:
                predicted=np.repeat(mean,len(evaluate)) if model=='mean' else evaluate.song_id.map(songs).fillna(mean).to_numpy()
                for value,(_,row) in zip(predicted,evaluate.iterrows()):
                    predictions.append({'record_id':row.record_id,'participant_id':row.participant_id,'song_id':row.song_id,
                        'model':model,'seed':0,'target':target,'actual':row[target],'predicted':value,
                        'split':'development_cv','fold':fold,'synthetic':False})
    assert len(heldout)==len(set(heldout))==26
    predictions=pd.DataFrame(predictions)
    metrics,people=summarize(predictions)
    fold_tables=[]
    for fold,sub in predictions.groupby('fold'):
        table,_=summarize(sub);table['fold']=fold;fold_tables.append(table)
    out=ROOT/'results/development'
    predictions.to_csv(out/'baseline_cv_predictions.csv',index=False)
    metrics.to_csv(out/'baseline_cv_metrics.csv',index=False)
    people.to_csv(out/'baseline_cv_participant_metrics.csv',index=False)
    pd.concat(fold_tables).to_csv(out/'baseline_cv_fold_metrics.csv',index=False)
    print(metrics[['model','target','participants','n','macro_mae','rmse']].to_string(index=False))
    print('Five outer folds cover development participants only; final holdout excluded.')
    return metrics

if __name__=='__main__':
    args=parser(__doc__).parse_args();run(load_config(args.config))
