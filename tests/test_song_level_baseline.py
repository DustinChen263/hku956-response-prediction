"""Research boundaries for the separate whole-track experiment."""
import csv
import pytest
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.common import load_config
from src.experiment import load_design,split_data

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not (ROOT/"data/processed/song_level_v1/manifest.csv").exists(), reason="Requires locally prepared real dataset")
def test_whole_track_scope_keeps_unverified_exposure_and_missing_hrv():
    cfg=yaml.safe_load((ROOT/'configs/song_level_v1.yaml').read_text())
    assert cfg['audio']['representation_scope']=='whole_track'
    assert cfg['training']['targets']==['mean_hr','arousal','valence']
    assert cfg['research_scope']['hrv_required'] is False
    assert cfg['research_scope']['final_holdout_evaluation'] is False
    m=pd.read_csv(ROOT/cfg['paths']['processed_dir']/'manifest.csv')
    assert not m.alignment_verified.any()
    assert (m.loc[m.audio_available,'audio_start_s']==0).all()
    assert (m.physiology_window_s==30).all()
    assert 'arousal' not in m and 'valence' not in m
    labels=pd.read_csv(ROOT/cfg['paths']['processed_dir']/'labels.csv')
    selected=m[m.main_eligible_before_features].merge(labels,on='record_id')
    assert selected.rmssd_ms.isna().any()  # Auxiliary absence does not filter the main cohort.
    assert selected[['mean_hr','arousal','valence']].notna().all().all()


@pytest.mark.skipif(not (ROOT/"data/splits/participants.json").exists(), reason="Requires local participant split")
def test_cv_never_contains_final_holdout_and_test_labels_not_selected():
    split=json.loads((ROOT/'data/splits/participants.json').read_text())
    final=set(split['groups']['test']);outer=[]
    for fold in split['cv']:
        groups=fold['groups']
        assert not set(sum(groups.values(),[]))&final
        assert not set(groups['train'])&set(groups['validation'])
        assert not set(groups['train'])&set(groups['test'])
        outer+=groups['test']
    assert len(outer)==len(set(outer))==26


def test_numeric_labels_only_loaded_for_selected_partition(tmp_path):
    from src.experiment import load_labels
    # A corrupt holdout numeric target must never be parsed for a train request.
    p=tmp_path/'labels.csv'
    p.write_text('record_id,mean_hr,arousal,valence\ntrain,70,1,2\nholdout,DO_NOT_PARSE,DO_NOT_PARSE,DO_NOT_PARSE\n')
    cfg={'paths':{'processed_dir':str(tmp_path)}}
    np.testing.assert_array_equal(load_labels(cfg,['train'],['mean_hr','arousal','valence']),[[70,1,2]])
