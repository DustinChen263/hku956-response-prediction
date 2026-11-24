import pytest
from scripts.recover_jamendo import validate_response, download_allowed, metadata_row, JamendoError


def test_suspended_app_not_mistaken_for_missing_track():
    with pytest.raises(JamendoError, match='11'):
        validate_response({'headers':{'status':'failed','code':11},'results':[]},'370177')


def test_empty_success_is_missing_track():
    assert validate_response({'headers':{'status':'success','code':0},'results':[]},'370177') is None


def test_unexpected_song_id_rejected():
    with pytest.raises(JamendoError, match='requested song ID'):
        validate_response({'headers':{'status':'success','code':0},'results':[{'id':'other'}]},'370177')


def test_download_permission_required():
    assert not download_allowed({'audiodownload_allowed':False,'audiodownload':'https://example.com/audio'})
    assert not download_allowed({'audiodownload':'https://example.com/audio'})
    assert download_allowed({'audiodownload_allowed':True,'audiodownload':'https://example.com/audio'})


def test_recovered_metadata_does_not_claim_verified_original():
    row=metadata_row('370177',{'id':'370177','name':'Example','audio':'secret url','audiodownload':'another url'})
    assert not row['recording_identity_verified']
    assert not row['playback_alignment_verified']
    assert 'audio' not in row and 'audiodownload' not in row


def test_batch_rejects_extra_track():
    from scripts.jamendo_batch import validate_batch
    with pytest.raises(JamendoError,match='unexpected'):
        validate_batch({'headers':{'status':'success','code':0},'results':[{'id':'999'}]},['370177'])


def test_batch_can_return_subset_without_faking_matches():
    from scripts.jamendo_batch import validate_batch
    result=validate_batch({'headers':{'status':'success','code':0},'results':[{'id':'370177'}]},['370177','77933'])
    assert set(result)=={'370177'}


def test_batch_omissions_are_rechecked_individually(monkeypatch):
    import scripts.jamendo_batch as batch
    calls=[]
    def fake(ids, client_id):
        calls.append(list(ids))
        return {sid:{'id':sid} for sid in ids} if len(ids)==1 else {}
    monkeypatch.setattr(batch,'_lookup_once',fake)
    monkeypatch.setattr(batch.time,'sleep',lambda _:None)
    assert set(batch.lookup_batch(['1','2'],'test'))=={'1','2'}
    assert calls==[['1','2'],['1'],['2']]


def test_http_429_is_not_retried_aggressively(monkeypatch):
    import urllib.error
    import scripts.jamendo_batch as batch
    calls=[]
    def rate_limited(*args,**kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://api.jamendo.com',429,'Too Many Requests',{'Retry-After':'120'},None)
    monkeypatch.setattr(batch.urllib.request,'urlopen',rate_limited)
    with pytest.raises(JamendoError,match='Retry-After=120'):
        batch._lookup_once(['370177'],'not-a-real-key')
    assert len(calls)==1
