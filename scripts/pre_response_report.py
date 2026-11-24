"""Assemble the four pre-response deliverables from completed artifacts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import json
import pandas as pd
from src.common import write_json,digest


def run():
    reports=ROOT/'reports';results=ROOT/'results/development'
    summary=json.loads((reports/'jamendo_recovery_summary.json').read_text())
    metrics=pd.read_csv(results/'baseline_metrics_validation.csv')
    cv=pd.read_csv(results/'baseline_cv_metrics.csv')
    protocol=json.loads((ROOT/'configs/protocol_v1.json').read_text())
    assert digest(reports/'protocol_v1.md')==protocol['prose_sha256']
    assert digest(results/'protocol.json')==protocol['baseline_protocol_sha256']
    matrix=pd.read_csv(reports/'literature_matrix.csv')
    lines=['# 作者回复前的四项工作','', '日期：2026-10-17。以下为真实 HKU956 数据工作；不包含合成烟测成绩。', '',
      '## 1. 标签与个体差异分析','',
      '已分析 22 位训练参与者的 666 条记录，以及 4 位验证参与者的 151 条记录。最终 4 位测试参与者未用于这些分析。',
      'mean-HR 的训练／验证平均值为 78.05／65.19 bpm，存在明显参与者分布偏移；保留原划分，不依据验证结果重抽。',
      '已输出分布、逐参与者统计、个体内中心化相关性、训练共享歌曲的听众间差异。',
      '[中文分析报告](development/development_report.md) · [英文初步结果](development/preliminary_results.md)', '',
      '## 2. 无需音频的诊断基线','',
      '|模型|目标|验证集参与者宏平均 MAE|开发集五折参与者宏平均 MAE|','|---|---|---:|---:|']
    for model in ['mean','song_mean']:
        for target in ['mean_hr','arousal','valence']:
            a=metrics[(metrics.model==model)&(metrics.target==target)&(metrics.group=='all')].iloc[0]
            b=cv[(cv.model==model)&(cv.target==target)].iloc[0]
            lines.append(f'|{model}|{target}|{a.macro_mae:.3f}|{b.macro_mae:.3f}|')
    lines += ['', 'HR 单位 bpm，A/V 为原始评分单位。歌曲均值仅为共享歌曲诊断，对未见歌曲回退训练全局均值。',
       '五折开发集预测覆盖 26 人和 817 条记录，每人仅作为一个外层评估折的对象；每折只在内层训练参与者拟合基线。',
       '没有运行正式音频模型，也没有打开最终测试评估。后续应在同一个已核验音频 cohort 上重算所有基线。','',
       '## 3. 文献与协议','',
       f'已整理 {len(matrix)} 篇正式同行评审文献，只有原始 HKU956 论文作为直接数据集研究；其余为音乐情绪、生理测量或建模方法依据。',
       '[文献综述](literature_review.md) · [证据矩阵](literature_matrix.csv) · [BibTeX](references.bib) · [协议 v1](protocol_v1.md)',
       '协议区分有文献支持的原则和暂定工程阈值。训练内固定窗／完整播放 mean-HR 的平均绝对差为 0.773 bpm；三组筛查选项均保留 666 条，不证明设备测量有效。',
       '协议为本地版本快照，不是公开预注册，也不声称验证标签尚未被查看。正式音频 cohort 须待作者对齐证据后锁定。','',
       '## 4. 剩余音频恢复','',
       f"现有 {summary['downloaded_candidates']} 首候选录音、{summary['file_bytes']/1e9:.3f} GB，按 ID 覆盖 {summary['covered_records_by_id']} 条记录、{summary['covered_participants_by_id']} 位参与者。",
       f"状态计数：{summary['statuses']}。",
       '逐文件保留 SHA-256、作者、许可、时长与解码信息。至少 3 秒查询间隔、小批量请求、两个下载连接；429 即停，不绕过限流。',
       'API 返回不稳定，先前匹配但后来未返回的状态保留为 lookup_inconsistent，不将它们认定为永久下架。',
       '[恢复报告](jamendo_recovery_report.md) · [合并明细](jamendo_recovery_all.csv) · [曝光覆盖](jamendo_record_coverage.csv)','',
       '## 仍需作者回复解决的问题','',
       '历史录音版本是否一致、实际播放起点以及暂停／续播／跳转的处理规则。当前 alignment_verified 保持 false；不自动将候选录音投入正式训练。']
    (reports/'pre_response_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write_json(reports/'pre_response_completion.json',{'development_analysis':'complete','diagnostic_baselines':'complete',
      'literature_and_protocol':'complete','audio_recovery':'attempted_available_items; see recovery statuses',
      'final_test_evaluated':False,'formal_audio_model_trained':False,'audio_summary':summary})
    print(reports/'pre_response_report.md')

if __name__=='__main__':run()
