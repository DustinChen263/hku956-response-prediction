import json
import numpy as np
import pandas as pd
from scripts.development_analysis import load_development
from scripts.resume_jamendo import pending_ids


def test_development_loader_never_parses_test_labels(tmp_path):
    processed=tmp_path/'processed';processed.mkdir()
    splits=tmp_path/'splits';splits.mkdir()
    pd.DataFrame({'record_id':['r1','r2','r3'],'participant_id':['train_person','val_person','test_person'],
        'song_id':['s1','s1','s2'],'duration_s':[40,40,40],
        'arousal':['1','2','DO_NOT_PARSE'],'valence':['1','2','DO_NOT_PARSE']}).to_csv(processed/'manifest.csv',index=False)
    (processed/'labels.csv').write_text('record_id,mean_hr,arousal,valence\nr1,70,1,1\nr2,80,2,2\nr3,FORBIDDEN,FORBIDDEN,FORBIDDEN\n')
    (splits/'participants.json').write_text(json.dumps({'groups':{'train':['train_person'],'validation':['val_person'],'test':['test_person']}}))
    data,_=load_development({'paths':{'processed_dir':str(processed),'splits_dir':str(splits)}})
    assert set(data.record_id)=={'r1','r2'}
    assert data.mean_hr.tolist()==[70,80]


def test_resume_does_not_redownload_completed_or_disallowed():
    df=pd.DataFrame({'song_id':['1','2','3','4','5'],'status':['downloaded_candidate','download_not_allowed','matched_id_only','pending_recheck','not_found']})
    assert pending_ids(df)==['3','4']


def test_inconsistent_lookup_remains_eligible_for_a_later_resume():
    df=pd.DataFrame({'song_id':['1'],'status':['lookup_inconsistent']})
    assert pending_ids(df)==['1']
