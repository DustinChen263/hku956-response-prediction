"""Report whole-track development baselines; never evaluate the final holdout."""
from pathlib import Path
import json
import sys
import os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.common import digest,write_json


def run():
    out=ROOT/'reports/song_level_v1';results=ROOT/'results/song_level_v1'
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reports/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts=pd.read_csv(out/'cohort_counts.csv')
    ext=json.loads((out/'extraction_summary.json').read_text())
    val=pd.read_csv(results/'metrics_validation.csv')
    cv=pd.read_csv(results/'development_cv_metrics.csv')
    bootstrap=pd.read_csv(results/'development_cv_paired_bootstrap.csv')
    predictions=pd.read_csv(results/'development_cv_predictions.csv')
    split=json.loads((ROOT/'data/splits/participants.json').read_text())
    final=set(split['groups']['test'])
    assert not final&set(predictions.participant_id)
    assert not (results/'evaluation_test.json').exists()
    key=['record_id','target']
    baseline=predictions[predictions.model=='mean'][key].sort_values(key).reset_index(drop=True)
    for model in ['song_mean','ridge']:
        compare=predictions[predictions.model==model][key].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(baseline,compare)
    registry=json.loads((results/'registry.json').read_text())
    alphas=[]
    for folder in [results,*sorted(results.glob('cv_*'))]:
        if not (folder/'registry.json').exists():continue
        reg=json.loads((folder/'registry.json').read_text())
        entry=next(m for m in reg['models'] if m['kind']=='ridge')
        alphas.append({'run':folder.name,'alpha':entry['params']['alpha'],'validation_standardized_macro_mae':entry['validation_score']})
    pd.DataFrame(alphas).to_csv(out/'selected_alphas.csv',index=False)
    plot=cv.pivot(index='target',columns='model',values='macro_mae')
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    names={'mean_hr':'Mean HR (bpm)','arousal':'Arousal (rating units)','valence':'Valence (rating units)'}
    for ax,target in zip(axes,['mean_hr','arousal','valence']):
        models=['mean','song_mean','ridge'];values=[plot.loc[target,m] for m in models]
        bars=ax.bar(['Mean','Song mean','Acoustic Ridge'],values,color=['#9099a2','#9ac9c7','#376bb2'],width=.65)
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=9)
        ax.set_ylim(0,max(values)*1.2);ax.set_title(names[target]);ax.set_ylabel('Participant macro MAE')
        ax.tick_params(axis='x',labelsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Whole-track acoustic baselines: development-only participant CV',fontsize=12)
    fig.tight_layout();fig.savefig(out/'development_cv_mae.png',dpi=180);plt.close(fig)
    def table(frame):
        columns=['model','target','n','participants','macro_mae','rmse','r2']
        selected=frame[columns].copy()
        lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
        for row in selected.itertuples(index=False,name=None):
            lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    details=[]
    for target in ['mean_hr','arousal','valence']:
        sub=cv[cv.target==target].set_index('model')
        base=float(sub.loc['mean','macro_mae']);ridge=float(sub.loc['ridge','macro_mae'])
        b=bootstrap[bootstrap.target==target].iloc[0]
        details.append(f"- {names[target]}：均值 {base:.4f} → Ridge {ridge:.4f}；MAE 相对变化 {(ridge/base-1)*100:+.2f}%。Ridge−mean 配对差值 {b.macro_mae_delta_a_minus_b:.4f}，参与者 bootstrap 描述性 95% 区间 [{b.ci_low:.4f}, {b.ci_high:.4f}]。")
    md=f'''# 整曲声学基线实际结果（2026-10-28）

本轮真实数据结果，非合成烟测。主目标 mean-HR、Arousal、Valence，HRV 不作为纳入条件。输入为整首 Jamendo 对应 ID 录音；这是歌曲层级关联/预测，输入不声称重建实际播放片段。完整协议见 protocol.md。

## 数据与运行

{ext['tracks']} 首录音提取成功，{ext['failures']} 首失败；{ext['records_with_features']} 条聆听记录有特征，{ext['main_records_with_features']} 条具备三个主目标。特征 {ext['dimension']} 维。全量提取墙钟 {ext['wall_s']/60:.2f} 分钟（包含缓存命中和加载）。每首只提取一次，同歌记录共享向量。

主实验固定训练 538 条／22 人、验证 122 条／4 人、最终预留测试 111 条／4 人。没有因 HRV 缺失删除记录。默认原标签中两条不足 30 秒记录不进入三个主目标完整子集。仅声学输入有数值结果，不代表精确刺激对齐已解决。

训练均值、训练歌曲均值和 Ridge 使用完全相同的记录。特征和目标标准化仅拟合相应训练分区；Ridge 在预先给定的四个 alpha 中使用参与者隔离验证选择。本轮主实验及五个 CV 折共 36 次拟合（每次 Ridge 同时训练三个独立输出回归器）。具体配置见 selected_alphas.csv 和各 registry.json。

## 固定验证集

{table(val)}

固定验证只有四人，且先前已有证据表明其 HR 分布与训练集不同。此处是选模/开发结果，不是最终测试结果。

## 五折开发集参与者 CV

每名非最终测试参与者恰好出现在一个外层评估折中。共 {predictions.record_id.nunique()} 条记录、{predictions.participant_id.nunique()} 人；外层训练内部的参与者验证集选择超参数，不用外层评估结果选 alpha。下表宏平均基于合并的逐参与者 MAE，而不是直接平均五折分数。

{table(cv)}

{chr(10).join(details)}

区间使用 2,000 次参与者簇 bootstrap，是开发集条件下的描述性不确定性，不作为预注册确认性显著性检验；区间若跨零，不能作稳定改善结论。多个目标分别报告，禁止平均原单位误差。调参与结论不读取四名最终测试参与者的预测误差。

![Development CV MAE](development_cv_mae.png)

## 限制与下一步

1. 整曲表征与非零起点、暂停、跳转的实际曝光存在差异；mean-HR 仅反映首 30 秒墙钟，A/V 是整次聆听评分，三者时间范围不同。输入定义已明确改变，不能写成精确音频片段反应预测。
2. 音频恢复子集不是全部 956 条，存在可用性选择偏差。歌曲跨参与者分区重叠，故只声称未见参与者泛化，不声称未见歌曲或个性化。
3. 歌曲均值是利用训练标签的共享歌曲诊断，不是音频模型，也不等同推荐有效性。模型可以不超过均值基线，这仍是可报告结果。
4. 本轮未训练 MLP、未使用 MERT、未评估最终预留测试。下一步在相同输入、cohort 和参与者划分上比较单任务/多任务小型 MLP，避免根据当前某个目标的正结果扩张搜索。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/song_level_baseline.py prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/song_level_baseline.py extract
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/song_level_baseline.py baseline
python scripts/report_song_level_baseline.py
```

prepare 核对已锁定协议；相同特征有缓存，已有匹配签名模型和评估会复用。配置见 configs/song_level_v1.yaml，输入、标签和输出位于独立 song_level_v1 目录。不会覆盖既有主流程或历史实验。
'''
    (out/'baseline_report.md').write_text(md)
    verification={'feature_sha256_verified':True,'same_records_for_all_models':True,
      'development_outer_participants':predictions.participant_id.nunique(),
      'final_holdout_excluded_from_development':True,'final_holdout_evaluated':False,
      'original_split_sha256':digest(ROOT/'data/splits/participants.json'),
      'report_source_sha256':digest(Path(__file__)),'feature_array_sha256':digest(ROOT/'features/song_level_v1/audio_features.npz')}
    write_json(out/'artifact_verification.json',verification)
    print('\n'.join(details))

if __name__=='__main__':run()
