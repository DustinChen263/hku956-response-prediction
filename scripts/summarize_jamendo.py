"""Merge initial and omission-recheck logs and verify local candidate checksums."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import pandas as pd
import soundfile as sf
from src.common import parser,load_config,folder,read_manifest,digest,write_json


def summarize(cfg):
    reports=folder(cfg,'reports')
    logs=[]
    for name in ('jamendo_recovery.csv','jamendo_recheck_catalog.csv','jamendo_recheck_recovery.csv','jamendo_resume.csv'):
        p=reports/name
        if p.exists():
            logs.append(pd.read_csv(p,dtype={'song_id':str}))
    if not logs:
        raise ValueError('No recovery logs')
    history=pd.concat(logs,ignore_index=True)
    observed_matches=set(history.loc[history.status.isin(['matched_id_only','downloaded_candidate','download_not_allowed','audio_download_or_decode_failed']),'song_id'])
    rows=history.drop_duplicates('song_id',keep='last').copy()
    # A later empty response does not erase an earlier verified ID match.
    inconsistent=(rows.status=='not_found') & rows.song_id.isin(observed_matches)
    rows.loc[inconsistent,'status']='lookup_inconsistent'
    recheck_file=reports/'jamendo_recheck_catalog.csv'
    checked=set(pd.read_csv(recheck_file,dtype={'song_id':str}).song_id) if recheck_file.exists() else set()
    resume_file=reports/'jamendo_resume.csv'
    if resume_file.exists():
        checked |= set(pd.read_csv(resume_file,dtype={'song_id':str}).song_id)
    # Initial large-batch omissions are unresolved until a small/single-ID recheck completes.
    unresolved=(rows.status=='not_found') & ~rows.song_id.isin(checked)
    rows.loc[unresolved,'status']='pending_recheck'
    local=folder(cfg,'audio')/'jamendo_candidates'
    verified=[]
    for i,row in rows.iterrows():
        if row.status!='downloaded_candidate':
            continue
        path=local/f'{row.song_id}.mp3'
        if not path.is_file() or digest(path)!=row.sha256:
            raise ValueError(f'Candidate checksum mismatch: {row.song_id}')
        info=sf.info(path)
        rows.loc[i,'decoded_duration_s']=info.duration
        rows.loc[i,'file_bytes']=path.stat().st_size
        verified.append(row.song_id)
    manifest=read_manifest(cfg)
    subset=manifest[manifest.song_id.isin(verified)].copy()
    durations=rows.set_index('song_id').decoded_duration_s.dropna().to_dict()
    subset['candidate_duration_s']=subset.song_id.map(durations)
    subset['length_sufficient_if_start_zero']=subset.candidate_duration_s+.05>=subset.duration_s
    subset[['record_id','participant_id','song_id','duration_s','candidate_duration_s','length_sufficient_if_start_zero']].to_csv(reports/'jamendo_record_coverage.csv',index=False)
    summary={'dataset_songs':int(manifest.song_id.nunique()),'dataset_records':len(manifest),
             'statuses':{str(k):int(v) for k,v in rows.status.value_counts().items()},
             'downloaded_candidates':len(verified),'matched_id_observed_any_run':len(observed_matches),'matched_ids_including_not_downloaded':int(rows.status.isin(['downloaded_candidate','matched_id_only','download_not_allowed','audio_download_or_decode_failed']).sum()),'covered_records_by_id':len(subset),
             'covered_participants_by_id':int(subset.participant_id.nunique()),
             'records_shorter_than_exposure_even_at_zero':int((~subset.length_sufficient_if_start_zero).sum()),
             'file_bytes':int(rows.loc[rows.status=='downloaded_candidate','file_bytes'].sum()),
             'recording_identity_verified':False,'playback_alignment_verified':False}
    rows.to_csv(reports/'jamendo_recovery_all.csv',index=False)
    write_json(reports/'jamendo_recovery_summary.json',summary)
    subset.groupby('participant_id').agg(covered_records=('record_id','size')).to_csv(reports/'jamendo_participant_coverage.csv')
    text=['# Jamendo 音频恢复结果','',f"下载并校验 {len(verified)} 首候选录音，覆盖 {len(subset)} 条聆听记录、{subset.participant_id.nunique()} 位参与者（仅按 ID 对应）。",'',
          f"文件合计 {summary['file_bytes']/1e9:.3f} GB。状态计数：{summary['statuses']}。",'',
          'API 批量查询曾漏报有效 ID；未完成复查的标为 pending_recheck，先前匹配但后来未返回的标为 lookup_inconsistent。not_found 仅表示本次单 ID 查询未返回曲目，不证明永久下架。',
          '下载仅使用 API 明确允许下载的链接；许可和作者归属记录在 CSV 与逐曲 JSON，不假设所有歌曲仍是 CC-BY。',
          '文件保留在 data/raw/audio/jamendo_candidates/，未加入真实训练。',
          f"即使假定播放起点为零，仍有 {summary['records_shorter_than_exposure_even_at_zero']} 条记录的播放时长超过候选文件长度，需要核查。",'',
          '**当前仍未核实与实验当年的录音版本一致性和实际播放起点；这些文件不是已完成对齐的研究输入。**',
          '真实模型训练未启动。没有使用 client_secret，也未将 client_id 写入配置。']
    (reports/'jamendo_recovery_report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(summary)
    return summary

if __name__=='__main__':
    args=parser(__doc__).parse_args()
    summarize(load_config(args.config))
