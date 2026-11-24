from pathlib import Path
import numpy as np
import pandas as pd
from .common import parser, load_config, folder, read_manifest, write_json
from .physiology import read_signal, rmssd_qc, mean_hr_qc


def build_labels(cfg):
    df = read_manifest(cfg)
    rows = []
    pcfg = cfg['physiology']
    for row in df.to_dict('records'):
        window = pcfg.get('window_s')
        duration = min(float(row['duration_s']), float(window)) if window else float(row['duration_s'])
        result = {'record_id': row['record_id'], 'arousal': row['arousal'], 'valence': row['valence'],
                  'label_window_s': duration, 'synthetic': bool(cfg.get('synthetic'))}
        if window and row['duration_s'] < window:
            result.update(rmssd_ms=np.nan, rmssd_quality='fail', rmssd_reason='short_record', mean_hr=np.nan, hr_quality='short_record')
        else:
            try:
                ibi = read_signal(str(row.get('ibi_path', '')), 'ibi')
                if ibi is None:
                    result.update(rmssd_ms=np.nan, rmssd_quality='missing', rmssd_reason='missing_ibi')
                else:
                    values = ibi['values']
                    # Named ibi_ms files are already in ms; untimed HKU values are seconds.
                    if ibi['times'] is None and pcfg['ibi_unit'] == 'seconds':
                        values = values * 1000
                    result.update(rmssd_qc(values, ibi['times'], duration, pcfg))
                hr = read_signal(str(row.get('hr_path', '')), 'hr')
                result.update(mean_hr_qc(hr, duration, pcfg))
            except (ValueError, OSError, pd.errors.EmptyDataError) as exc:
                raise ValueError(f"Signal parsing failed for {row['record_id']}: {exc}") from exc
        rows.append(result)
    labels = pd.DataFrame(rows)
    labels.to_csv(folder(cfg, 'processed') / 'labels.csv', index=False)
    counts = {t: int(labels[t].notna().sum()) for t in ('rmssd_ms', 'mean_hr', 'arousal', 'valence')}
    report = folder(cfg, 'reports')
    write_json(report / 'label_summary.json', counts)
    reasons = labels.rmssd_reason.fillna('pass').value_counts().to_dict()
    (report / 'qc_report.md').write_text('# 生理标签质量报告\n\n'
        f"合成数据：{bool(cfg.get('synthetic'))}\n\n有效标签：{counts}\n\nRMSSD 原因计数：{reasons}\n\n"
        f"当前窗口：{pcfg.get('window_s')} 秒；None 表示完整播放。阈值为配置中的暂定工程参数，须经文献与数据审计后锁定。\n"
        'untimed IBI 仅生成诊断候选数值，rmssd_ms 保持缺失；没有做时间戳推算或缺口拼接。\n'
        'mean_hr 单位 bpm，使用播放开头的固定窗口，与 RMSSD 是不同目标。\n', encoding='utf8')
    print('Valid labels:', counts)
    return labels


def main():
    args = parser('Build physiological and emotion labels').parse_args()
    build_labels(load_config(args.config))

if __name__ == '__main__':
    main()
