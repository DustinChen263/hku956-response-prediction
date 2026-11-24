"""Audit official HKU956 files or a documented normalized manifest."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import parser, load_config, folder, digest, write_json, validate_manifest


def hku_manifest(cfg):
    root = Path(cfg['paths']['raw_dir'])
    ratings = pd.read_csv(root / '3. AV_ratings.csv', dtype={'song_id': str, 'participant_id': str})
    links = pd.read_csv(root / '2. original_song_audio.csv', dtype={'song_id': str})
    if links.song_id.duplicated().any():
        raise ValueError('Duplicate song ids in audio-link table')
    ratings = ratings.merge(links, on='song_id', how='left', validate='many_to_one')
    signal_root = root / '1. physiological_signals'
    segments_path = root / 'audio_segments.csv'
    segments = None
    if segments_path.exists():
        segments = pd.read_csv(segments_path, dtype={'record_id': str}).set_index('record_id', verify_integrity=True)
    records = []
    for r in ratings.itertuples():
        rid = f'{r.participant_id}_{r.song_no}_{r.song_id}'
        stem = f'{r.song_no}_{r.song_id}.csv'
        audio_dir = Path(cfg['paths']['audio_dir'])
        audio = next((p for ext in ('mp3', 'wav', 'flac') if (p := audio_dir / f'{r.song_id}.{ext}').is_file()), None)
        start = np.nan
        verified = False
        if segments is not None and rid in segments.index:
            seg = segments.loc[rid]
            start = float(seg['audio_start_s'])
            verified = str(seg.get('alignment_verified', '')).lower() in ('true', '1')
        records.append(dict(record_id=rid, participant_id=r.participant_id, song_id=r.song_id,
                            song_no=r.song_no, audio_path=str(audio) if audio else '', audio_url=r.link,
                            audio_start_s=start, audio_end_s=start + r.play_duration,
                            alignment_verified=verified, listen_start='', listen_end='',
                            ibi_path=str(signal_root / r.participant_id / 'IBI' / stem),
                            ibi_format='untimed_values',
                            hr_path=str(signal_root / r.participant_id / 'HR' / stem),
                            arousal=r.arousal_rating, valence=r.valence_rating, duration_s=r.play_duration,
                            synthetic=False))
    return pd.DataFrame(records)


def audit(cfg):
    raw = Path(cfg['paths']['raw_dir'])
    if cfg.get('adapter') == 'hku956':
        df = hku_manifest(cfg)
    else:
        df = pd.read_csv(cfg['manifest'], dtype={'record_id': str, 'participant_id': str, 'song_id': str})
        for col in ('audio_path', 'ibi_path', 'hr_path'):
            if col in df:
                df[col] = df[col].map(lambda s: str((Path(cfg['manifest']).parent / str(s)).resolve()) if pd.notna(s) and not Path(str(s)).is_absolute() else s)
    validate_manifest(df)
    if 'synthetic' in df:
        markers = df.synthetic.astype(str).str.lower().isin(['true', '1'])
        if not (markers == bool(cfg.get('synthetic'))).all():
            raise ValueError('Synthetic marker mismatch: real and synthetic records must be isolated')
    df['synthetic'] = bool(cfg.get('synthetic'))
    df['audio_available'] = df.audio_path.fillna('').map(lambda x: bool(x) and Path(x).is_file())
    df['ibi_available'] = df.get('ibi_path', pd.Series('', index=df.index)).fillna('').map(lambda x: bool(x) and Path(x).is_file())
    df['availability_flags'] = [f'audio={a};ibi={i}' for a, i in zip(df.audio_available, df.ibi_available)]
    processed, reports = folder(cfg, 'processed'), folder(cfg, 'reports')
    df.to_csv(processed / 'manifest.csv', index=False)
    files = []
    if raw.exists():
        for p in sorted(raw.rglob('*')):
            if p.is_file():
                files.append({'path': str(p.relative_to(raw)), 'bytes': p.stat().st_size, 'sha256': digest(p)})
    pd.DataFrame(files, columns=['path', 'bytes', 'sha256']).to_csv(processed / 'raw_inventory.csv', index=False)
    df.groupby('participant_id').agg(records=('record_id', 'size'), audio_available=('audio_available', 'sum'), ibi_available=('ibi_available', 'sum')).to_csv(processed / 'participant_counts.csv')
    aligned = df.get('alignment_verified', pd.Series(False, index=df.index)).astype(str).str.lower().isin(['true', '1'])
    summary = dict(synthetic=bool(cfg.get('synthetic')), records=len(df), participants=df.participant_id.nunique(),
                   songs=df.song_id.nunique(), local_audio_records=int(df.audio_available.sum()),
                   verified_alignment_records=int(aligned.sum()), ibi_records=int(df.ibi_available.sum()),
                   missing_arousal=int(df.arousal.isna().sum()), missing_valence=int(df.valence.isna().sum()),
                   repeated_participant_song_records=int(df.duplicated(['participant_id', 'song_id'], keep=False).sum()),
                   duration_min=float(df.duration_s.min()), duration_mean=float(df.duration_s.mean()), duration_max=float(df.duration_s.max()))
    write_json(reports / 'audit_summary.json', summary)
    issues = []
    if not df.audio_available.all():
        issues.append('音频未全部本地可用；官方包仅提供链接。缺失音频不得用其他版本替代。')
    if not aligned.all():
        issues.append('部分或全部音频起点尚未核验；禁止默认用整首歌开头冒充实际聆听片段。')
    if cfg.get('adapter') == 'hku956':
        issues.append('官方 IBI CSV 无逐拍时间戳；严格模式不生成可用于主实验的 RMSSD 标签。mean_hr 是显式备选目标。')
    status = '可提取音频特征' if (df.audio_available & aligned).any() else '真实音频训练尚受资源或片段对齐阻碍'
    lines = ['# 数据审计', '', f"合成数据：{summary['synthetic']}。路线：{status}。", '', '```json', __import__('json').dumps(summary, indent=2, ensure_ascii=False), '```', '', '## 数据约束', '']
    lines += ['- ' + i for i in issues] or ['- 未发现阻断性缺失。']
    lines += ['', '重复 participant/song 是不同聆听事件时保留；不直接删除重复歌曲。', '数据许可：官方 HKU956 元数据为 CC BY-NC 4.0；音频还需保留原作者与曲目归属。', '官方说明的播放内对齐不提供播放前基线，不能估计真实情绪或 HRV 改变量。']
    (reports / 'data_audit.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    (reports / 'data_dictionary.md').write_text('# Manifest 数据字典\n\n每行是一次聆听事件。record_id 唯一；participant_id 和 song_id 为字符串。\n'
        'duration_s 为播放秒数；arousal/valence 为原始 [-10,10] 听后评分；audio_start_s 为原音频中的真实播放起点（秒）。\n'
        'alignment_verified 必须为 true 才能提取特征。ibi_path 指向间期文件；带时间戳的标准格式为 time_s,ibi_ms。\n'
        'time_s 为相对播放起点的间期结束时刻；hr_path 指向 1 Hz HR 数值或 time_s,hr_bpm 文件。\n'
        'audio_segments.csv 为用户核验的 record_id,audio_start_s,alignment_verified 表；不能自行猜测。\n', encoding='utf8')
    print(f"Audit: {len(df)} records, {summary['participants']} participants; {status}")
    return df


def main():
    args = parser(__doc__).parse_args()
    audit(load_config(args.config))

if __name__ == '__main__':
    main()
