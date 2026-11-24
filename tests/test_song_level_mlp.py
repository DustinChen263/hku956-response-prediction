import numpy as np
import pandas as pd
import pytest
from scripts.song_level_mlp import average_seed_predictions
from src.train import candidates


def rows():
    common=dict(record_id='r',participant_id='p',song_id='s',target='arousal',split='development_cv',actual=2.,synthetic=False)
    return pd.DataFrame([{**common,'model':'multi_mlp','seed':s,'predicted':p} for s,p in zip([17,42,93],[1.,2.,6.])]+[{**common,'model':'mean','seed':17,'predicted':0.}])


def test_seed_aggregation_is_prediction_ensemble_not_extra_participants():
    result=average_seed_predictions(rows())
    assert len(result)==2 and result.participant_id.nunique()==1
    assert result.loc[result.model=='multi_mlp','predicted'].iloc[0]==3.
    assert result.loc[result.model=='multi_mlp','seed_count'].iloc[0]==3
    assert result.seed.eq(0).all()


def test_seed_aggregation_rejects_missing_seed_or_mismatched_truth():
    f=rows()
    with pytest.raises(ValueError,match='Incomplete'):average_seed_predictions(f[f.seed!=93])
    f.loc[0,'actual']=8.
    with pytest.raises(ValueError,match='ground truth'):average_seed_predictions(f)


def test_fixed_search_grid_and_parameter_budget_levels():
    cfg=dict(learning_rate=.003,epochs=500,patience=40,min_delta=1e-5,pca_components=None,hidden_sizes=[16,32],l2_values=[.001,.01])
    independent=candidates('single_mlp',cfg,37,3)
    matched=candidates('matched_multi_mlp',cfg,37,3)
    assert len(independent)==len(matched)==4
    for a,b in zip(independent,matched):
        single=3*(39*a['hidden']+1)
        shared=41*b['hidden']+3
        assert abs(shared-single)/single<.01
