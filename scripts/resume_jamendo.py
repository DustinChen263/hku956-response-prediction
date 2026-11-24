"""Resume candidate recovery in small, paced batches, preserving every completed item."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import pandas as pd
import scripts.jamendo_batch as batch
from scripts.recover_jamendo import JamendoError, metadata_row, download_allowed
from src.common import folder,load_config,write_json


def pending_ids(table):
    states=['matched_id_only','pending_recheck','audio_download_or_decode_failed','lookup_inconsistent']
    return table.loc[table.status.isin(states),'song_id'].drop_duplicates().astype(str).tolist()


def run(cfg,key,gap=3,workers=2):
    if not key:
        raise JamendoError('Set JAMENDO_CLIENT_ID')
    reports=folder(cfg,'reports')
    catalog=pd.read_csv(reports/'jamendo_recovery_all.csv',dtype={'song_id':str})
    log=reports/'jamendo_resume.csv'
    previous=pd.read_csv(log,dtype={'song_id':str}) if log.exists() else pd.DataFrame()
    if not previous.empty:
        observed=set(catalog.loc[catalog.status=='lookup_inconsistent','song_id'])
        catalog=pd.concat([catalog,previous],ignore_index=True).drop_duplicates('song_id',keep='last').copy()
        catalog.loc[(catalog.status=='not_found') & catalog.song_id.isin(observed),'status']='lookup_inconsistent'
    ids=pending_ids(catalog)
    rows={str(r['song_id']):r for r in previous.to_dict('records')} if not previous.empty else {}
    directory=folder(cfg,'audio')/'jamendo_candidates'
    directory.mkdir(parents=True,exist_ok=True)
    original=batch._lookup_once
    last=[0.0]
    def paced(requested,client_id):
        time.sleep(max(0,gap-(time.monotonic()-last[0])))
        try:
            return original(requested,client_id)
        finally:
            last[0]=time.monotonic()
    batch._lookup_once=paced
    completed=0
    try:
        for start in range(0,len(ids),10):
            group=ids[start:start+10]
            tracks=batch.lookup_batch(group,key)
            for sid in group:
                rows[sid]=metadata_row(sid,tracks.get(sid))
                if sid in tracks and not download_allowed(tracks[sid]):
                    rows[sid]['status']='download_not_allowed'
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs=[pool.submit(batch.download_candidate,t,directory) for t in tracks.values() if download_allowed(t)]
                for job in as_completed(jobs):
                    row=job.result(); rows[row['song_id']]=row
                    pd.DataFrame(rows.values()).to_csv(log,index=False)
            pd.DataFrame(rows.values()).to_csv(log,index=False)
            completed+=len(group)
            write_json(reports/'jamendo_resume_status.json',{'state':'running','requested':len(ids),'processed_this_run':completed,
                'downloaded_in_resume_log':sum(r['status']=='downloaded_candidate' for r in rows.values()),'minimum_query_gap_s':gap})
            print(f'Resume {completed}/{len(ids)} IDs; downloaded {sum(r["status"]=="downloaded_candidate" for r in rows.values())}',flush=True)
    except JamendoError as exc:
        write_json(reports/'jamendo_resume_status.json',{'state':'blocked','requested':len(ids),'processed_this_run':completed,
            'reason':str(exc),'minimum_query_gap_s':gap})
        raise
    finally:
        batch._lookup_once=original
    write_json(reports/'jamendo_resume_status.json',{'state':'complete','requested':len(ids),'processed_this_run':completed,
        'downloaded_in_resume_log':sum(r['status']=='downloaded_candidate' for r in rows.values()),'minimum_query_gap_s':gap})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/default.yaml'))
    p.add_argument('--gap',type=float,default=3)
    args=p.parse_args()
    if args.gap<3:
        p.error('Minimum query gap is 3 seconds')
    try:
        run(load_config(args.config),os.environ.get('JAMENDO_CLIENT_ID',''),args.gap)
    except JamendoError as exc:
        p.exit(2,str(exc)+'\n')
