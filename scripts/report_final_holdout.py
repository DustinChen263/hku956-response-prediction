"""Summarize the frozen final holdout, keeping development evidence separate."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common import write_json,digest
from scripts.final_holdout import verify_lock


def table(frame,cols):
    lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for row in frame[cols].itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def run():
    lock=verify_lock();out=ROOT/'reports/final_holdout_v1';results=ROOT/'results/final_holdout_v1'
    metrics=pd.read_csv(results/'ensemble_metrics.csv')
    people=pd.read_csv(results/'participant_metrics.csv')
    bootstrap=pd.read_csv(results/'paired_bootstrap.csv')
    delta=pd.read_csv(results/'primary_comparison_participants.csv')
    dev=pd.read_csv(ROOT/'results/song_level_mlp_v1/development_cv_ensemble_metrics.csv')
    comparison=dev[['model','target','macro_mae']].rename(columns={'macro_mae':'development_cv_macro_mae'}).merge(
        metrics[['model','target','macro_mae']],on=['model','target'],validate='one_to_one').rename(columns={'macro_mae':'final_test_macro_mae'})
    comparison.to_csv(out/'development_vs_final.csv',index=False)
    primary=bootstrap[(bootstrap.model_a=='multi_mlp')&(bootstrap.model_b=='single_mlp')]
    seed=pd.read_csv(results/'individual_seed_metrics.csv')
    ranges=seed.groupby(['model','target']).macro_mae.agg(['min','mean','max']).reset_index()
    ranges.to_csv(out/'individual_seed_mae_ranges.csv',index=False)
    sample=people[people.model=='mean'][['target','participant_id','n']]
    sample=sample[sample.target=='mean_hr'].drop(columns='target')
    assert sample.n.sum()==111 and len(sample)==4
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reports/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.8))
    pids=lock['test_participants']
    for ax,t,label in zip(axes,['mean_hr','arousal','valence'],['Mean HR (bpm)','Arousal','Valence']):
        d=delta[delta.target==t].set_index('participant_id').loc[pids]
        ax.axvline(0,color='#777',linestyle='--',linewidth=1)
        ax.scatter(d.delta_multi_minus_single,range(4),s=55,color=['#347a75' if x<0 else '#b0664b' for x in d.delta_multi_minus_single],zorder=3)
        for i,x in enumerate(d.delta_multi_minus_single):ax.annotate(f'{x:+.3f}',(x,i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.set_yticks(range(4),pids);ax.set_ylim(3.7,-.7);ax.set_title(label)
        ax.set_xlabel('MAE difference: multi minus single')
        ax.margins(x=.25);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Final holdout: all four participants (negative favors multi-task)',fontsize=12)
    fig.tight_layout();fig.savefig(out/'participant_primary_comparison.png',dpi=180);plt.close(fig)
    md=f'''# 最终预留测试结果（2026-10-28）

**主要结论：常规共享三任务 MLP 相比单任务 MLP 的开发集优势未在最终测试保持。Arousal 的音频预测仍优于均值基线，但多任务收益取决于模型家族，不能宣称全面或稳定提升。**

## 方案与执行

用户授权后，先保存 protocol.json，锁定原参与者划分、111 条测试记录、15 个已训练模型文件及哈希、三个主目标、三种子平均方式、比较和指标，再访问最终测试预测误差。没有训练、train+validation 重训、阈值修改、测试后排除或挑选表现最好的种子。

保留所有既定家族：mean、song_mean、ridge、single_mlp、multi_mlp、matched_multi_mlp、av_mlp。神经模型按记录等权平均 seed17/42/93 的预测；同时报告每个种子。主比较为 multi_mlp − single_mlp，分别评价三个目标；其余为已冻结的补充比较。最终测试结果不用于再次调参或选择“获胜部署模型”。

历史训练配置中的 final_holdout_evaluation=false 是当时暂缓状态，原文件与签名未改。此次独立最终评估协议明确授权覆盖该暂缓状态；不通过修改旧配置绕过签名。评估缓存保存在原模型目录，综合报告和指标放在 final_holdout_v1 独立目录。

## 样本与输入

测试 4 人、111 条，来自原预留 139 条：113 条已有音频，其中两条不满足原 30 秒生理标签要求。仍使用原来已确定的主目标完整 cohort；HRV 不作为过滤条件。

{table(sample,['participant_id','n'])}

输入是 37 维整曲声学特征，不声称重建实际聆听片段；mean-HR 是首 30 秒墙钟平均心率（bpm），A/V 为整次聆听后的 [-10,10] 原始评分。暂停、非零起点和跳转仍是输入与曝光偏差来源。歌曲在训练和测试间有重叠，因此只评价未见参与者，不声称未见歌曲或个性化。

## 最终测试：所有家族

主要指标为参与者宏平均 MAE，四人等权；RMSE、R² 等为记录级辅助指标。神经模型表中为三种子预测集成。四人对应的原单位误差不能跨目标直接平均。

{table(metrics,['model','target','n','participants','macro_mae','rmse','r2'])}

## 主要比较：共享三任务 vs 单任务

{table(primary,['target','macro_mae_delta_a_minus_b','ci_low','ci_high'])}

- mean-HR：单任务 7.1547、多任务 7.7439 bpm，差 +0.5892。四人中 3 人多任务误差更高。单任务点估计低于均值 7.5528，但开发 CV 中该优势没有一致出现，生理目标的泛化证据有限。
- Arousal：单任务 3.1949、多任务 3.2858，差 +0.0908；四人多任务误差均较高。开发 CV 中多任务相对单任务的小幅改善没有复现。
- Valence：单任务 3.9461、多任务 4.0493，差 +0.1032；两人变好、两人变差，与零差异相容。单任务与均值 3.9472 基本持平，不能认为音频模型取得稳定实质收益。

仅四个参与者簇，bootstrap 95% 区间非常离散且不稳定，**只作描述性不确定性，不作确认性显著性声明**。111 条重复记录不是 111 个独立受试者，也不能通过增加 bootstrap 次数弥补人数不足。

![All participants](participant_primary_comparison.png)

{table(delta,['participant_id','target','records','single_mae','multi_mae','delta_multi_minus_single'])}

## 补充结果的解释

- Arousal 上 Ridge 3.2846、常规多任务 3.2858，都低于均值 4.0274；整曲声学特征有一定预测信息，但不意味着多任务优于简单模型。
- 预算匹配候选家族的 Arousal 为 3.1247，是本测试已列音频模型中的较低点估计；其 HR 7.8960、Valence 4.0622 没有全面收益。该家族最终选中 1889 参数，而单任务为 3747 参数，不能称为严格等参控制，也不依据其测试名次宣布新的最优模型。
- A/V 双任务 Arousal 3.1659，优于常规三任务 3.2858；这与开发 CV 中两者的方向不同。其 Valence 3.9781 与三任务 4.0493 的差异不稳定，不能据此确定 HR 任务必然产生负迁移。
- song_mean 在 Valence 为 3.7924，但依赖共享歌曲的训练标签，是诊断性参照；不能把它当作纯音频泛化模型或推荐有效性。

所有预定补充比较及区间保存在 results/final_holdout_v1/paired_bootstrap.csv，未按是否有利筛选结果。

## 开发集与最终测试分开报告

{table(comparison,['model','target','development_cv_macro_mae','final_test_macro_mae'])}

两种分数来自不同参与者和训练方式，不能简单把分数降低解释为性能进步；应重点看每个分区内对相同基线的差异与模型排序是否保持。开发 CV 是模型开发证据，最终四人提供独立但小样本的评估，两者不合并为一个新的调参指标。

## 各种子结果范围

{table(ranges,['model','target','min','mean','max'])}

这里是每个单种子模型的测试 MAE 范围及均值，区别于先平均预测后的集成 MAE。没有根据最终测试选种子或更改集成权重。

## 可用于论文的结论

在 HKU956 的可恢复音频子集上，使用整曲声学特征预测未见参与者的聆听反应，Arousal 相比训练均值呈现较一致的改善；mean-HR 与 Valence 的改善有限且依赖划分和模型。共享多任务网络没有显示对三个目标的稳定全面优势。结果支持对任务依赖性、输入与曝光不一致、个体生理差异、测量噪声和小样本的讨论，不能转述为因果作用或已验证的音乐干预效果。

HRV 仍为独立辅助／探索性方向，本轮未增加其模型结果。MERT、严格曝光片段和更充分参数控制尚未实施，不声称原长期研究计划已全部完成。

## 交付与复现

- protocol.json：最终评估前冻结的模型、记录、配置和代码哈希。
- results/final_holdout_v1/：逐记录、各随机种子、集成、逐参与者指标及比较区间。
- development_vs_final.csv：开发 CV 与最终测试分开列示。
- completion.json、artifact_verification.json：完成与独立校验记录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/final_holdout.py evaluate
python scripts/report_final_holdout.py
```

相同签名模型的后续执行读取缓存，核对预测校验值，不会重新拟合或重新选择。最终测试已开封；若以后新增特征、阈值或模型，应明确标为后续探索，不再声称这些新增选择拥有未使用过的本测试集。

下一步是整理论文方法、结果和限制，并在单独协议中明确是否实施 HRV 或精确曝光子集补充；保留本轮冻结结果作为主线证据。
'''
    (out/'final_report.md').write_text(md)
    print('Final holdout report saved.')

if __name__=='__main__':run()
