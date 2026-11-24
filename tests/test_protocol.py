import copy
import csv
import numpy as np
import pandas as pd
import pytest
from src.common import assert_splits, validate_manifest
from src.experiment import load_labels
from src.make_splits import make_splits
from src.evaluate import summarize, paired_bootstrap, compute_metrics
from src.recommend import rank_candidates


def test_participant_overlap_rejected():
    with pytest.raises(ValueError,match='leakage'):
        assert_splits({'train':['a'],'validation':['b'],'test':['a']})


def test_manifest_duplicate_record_rejected():
    df=pd.DataFrame({'record_id':['a','a'],'participant_id':['p','q'],'song_id':['s','s'],
                     'audio_path':['',''],'duration_s':[30,30],'arousal':[0,1],'valence':[0,1]})
    with pytest.raises(ValueError,match='unique'):
        validate_manifest(df)


def test_no_numeric_test_label_access(tmp_path):
    cfg={'paths':{'processed_dir':str(tmp_path)}}
    (tmp_path/'labels.csv').write_text('record_id,arousal\ntrain,1.5\ntest,THIS_TEST_LABEL_MUST_NOT_BE_PARSED\n')
    assert load_labels(cfg,['train'],['arousal'])[0,0] == 1.5


def test_split_cv_excludes_final_holdout(tmp_path):
    processed=tmp_path/'processed'; processed.mkdir()
    pd.DataFrame({'record_id':[f'r{i}' for i in range(30)],'participant_id':[f'p{i}' for i in range(30)],
                  'song_id':['shared']*30}).to_csv(processed/'manifest.csv',index=False)
    cfg={'paths':{'processed_dir':str(processed),'splits_dir':str(tmp_path/'splits')},
         'split':{'seed':17,'counts':[22,4,4],'cv_folds':5,'song_disjoint':False}}
    split=make_splits(cfg)
    assert [len(split['groups'][k]) for k in ['train','validation','test']] == [22,4,4]
    for fold in split['cv']:
        all_ids=set(sum(fold['groups'].values(),[]))
        assert not all_ids & set(split['groups']['test'])
    assert make_splits(cfg)==split


def test_macro_mae_does_not_weight_frequent_participant():
    p=pd.DataFrame({'model':['a']*11,'seed':[17]*11,'target':['arousal']*11,'split':['test']*11,
                    'participant_id':['one']*10+['two'],'actual':[0]*11,'predicted':[1]*10+[9]})
    table,_=summarize(p)
    assert table.iloc[0].macro_mae == 5
    assert np.isclose(table.iloc[0].mae,19/11)


def test_constant_pearson_undefined():
    assert np.isnan(compute_metrics(np.array([1.,2.]),np.array([1.,1.]))['pearson'])


def test_goal_ranking_uses_training_scale():
    pred=np.array([[100,1,3],[20,-3,3]])
    score=rank_candidates(pred,['rmssd_ms','arousal','valence'],{'arousal':-3,'valence':3},np.array([10,2,2]))
    assert score[1] > score[0]
    assert score[1] == 0  # Physiology ignored by default.
