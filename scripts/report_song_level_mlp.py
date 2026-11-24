"""Report frozen-budget neural comparisons without final holdout access."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common import digest,write_json


def markdown(frame,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in frame[columns].itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def run():
    out=ROOT/'reports/song_level_mlp_v1';results=ROOT/'results/song_level_mlp_v1'
    metrics=pd.read_csv(results/'development_cv_ensemble_metrics.csv')
    seed=pd.read_csv(results/'development_cv_individual_seed_metrics.csv')
    boot=pd.read_csv(results/'development_cv_ensemble_paired_bootstrap.csv')
    validation=pd.read_csv(results/'validation_ensemble_metrics.csv')
    selected=pd.read_csv(out/'selected_models.csv')
    params=selected.groupby(['run','model']).parameters.first().unstack()
    params['selected_matched_to_single_ratio']=params.matched_multi_mlp/params.single_mlp
    params.to_csv(out/'actual_parameter_comparison.csv')
    cols=['model','target','macro_mae','rmse','r2']
    ranges=seed.groupby(['model','target']).macro_mae.agg(['min','mean','max']).reset_index()
    ranges.to_csv(out/'individual_seed_mae_ranges.csv',index=False)
    delta=boot[(boot.model_a=='multi_mlp')&(boot.model_b=='single_mlp')]
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reports/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    models=['mean','ridge','single_mlp','multi_mlp']
    pivot=metrics.pivot(index='target',columns='model',values='macro_mae')
    for ax,t,label in zip(axes,['mean_hr','arousal','valence'],['Mean HR (bpm)','Arousal','Valence']):
        vals=[pivot.loc[t,m] for m in models]
        bars=ax.bar(['Mean','Ridge','Single MLP','Multi MLP'],vals,color=['#a5a9ad','#70a8c8','#d69b64','#427f70'])
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=8)
        ax.set_ylim(0,max(vals)*1.2);ax.set_title(label);ax.set_ylabel('Participant macro MAE')
        ax.tick_params(axis='x',labelsize=8);ax.spines[['right','top']].set_visible(False)
    fig.suptitle('Development-only participant CV: MLP predictions averaged across 3 seeds',fontsize=11)
    fig.tight_layout();fig.savefig(out/'development_cv_mlp_mae.png',dpi=180);plt.close(fig)
    md=f'''# 单任务与多任务 MLP 比较（2026-10-28）

本轮已完成真实数据训练。输入、标签、cohort 与整曲声学基线相同：37 维整曲特征，mean-HR／Arousal／Valence 三主目标，训练 538 条／22 人、验证 122 条／4 人，开发 CV 合并外层评估 660 条／26 人。HRV 不参与筛选或损失；原四名最终预留测试参与者未评估。

## 主要结果

三种子预测先在逐记录层面平均，再计算参与者宏平均 MAE；均值和 Ridge 为确定性单次预测。各目标分别比较，不混合原始单位。此表评估的是三种子集成模型，不是单次神经网络训练的期望误差；单种子结果另报。

{markdown(metrics,cols)}

多任务相对单任务的主要比较（差值 = multi − single，负数更好）：

{markdown(delta,['target','macro_mae_delta_a_minus_b','ci_low','ci_high','participants','paired_records'])}

- Arousal：多任务 3.1088 对单任务 3.1689，小幅改善约 1.90%；描述性参与者 bootstrap 区间位于零以下，但仍未超过 Ridge 的 3.0708。不能据此宣称多任务是最优方法。
- mean-HR：多任务 10.1878 对单任务 10.0440，点估计较差；与零差异相容。均值基线 9.9720 仍低于本轮各神经模型，没有获得可靠的音频预测 HR 优势。
- Valence：多任务 3.4912 对单任务 3.5310，小幅改善但区间跨零；也未超过均值 3.4672，尚无稳定收益。

所有区间为开发数据条件下 2,000 次参与者簇 bootstrap 的描述性结果，未作正式多重检验或确认性显著性声明。种子不是独立参与者，外层折不是五次独立受试者实验。

![MLP comparison](development_cv_mlp_mae.png)

## 训练与搜索

使用现有 NumPy 单隐藏层 ReLU MLP、全批 Adam，最多 500 epochs，patience=40，learning_rate=0.003，min_delta=1e-5。三主目标用训练分区标签尺度标准化，损失为等权 MSE；早停和配置选择使用标准化的参与者宏平均 MAE。所有特征 scaler 仅拟合相应训练分区。

每个家族 4 个候选，先用 seed=17 选择，再用同一选定配置复跑 seed=42、93；不按其他种子结果重选配置。单任务是三个独立网络，但为保持家族调参预算一致，共用所选 hidden/l2 配置，每个目标独立早停。

- single_mlp：三个独立网络，hidden 16／32，l2 0.001／0.01。
- multi_mlp：三输出共享网络，同样的 hidden/l2 网格。
- matched_multi_mlp：共享候选 hidden 46／91，分别近似匹配单任务 hidden 16／32 的三个网络总参数，l2 网格不变。
- av_mlp：只有 A/V 双输出的共享模型，同一主 cohort、同样四候选；与三任务模型比较的是任务集合变化，不声称单独隔离所有结构差异。

固定主划分和五个开发外层折，共 144 次 ResponseModel.fit，对应 216 次网络训练；另有 1 次事先指定配置的估时试跑。最终保存 72 个三种子候选赢家的模型文件（单任务文件内含三个网络）。具体参数、早停 epoch 见 selected_models.csv。

### 参数量控制的限制

预算匹配家族在候选层级的总参数误差小于 1%，但它与单任务家族独立选模后可能选择不同容量等级。因此选出的模型不构成每折严格相等参数量的控制，不能把它的结果解释为已经完全排除参数量影响。实际参数见 actual_parameter_comparison.csv；例如主划分单任务为 3747 参数，而被选中的匹配家族为 1889 参数。后者的三个主目标也没有形成稳定全面优势。

## 单种子结果

以下是每个种子先合并外层预测、再算参与者宏平均 MAE 后的最小／均值／最大。三种子均值与预测集成后的误差是不同统计量，不能混用。

{markdown(ranges,['model','target','min','mean','max'])}

## 固定验证结果

{markdown(validation,cols)}

固定验证用于开发，不等于最终测试。外层 CV 的各折配置仅由其内层验证选择；最终四名参与者完全排除于此。

## 协议、修复与边界

protocol.json 在估时和正式 MLP 拟合前冻结配置、输入、代码和划分哈希；记录了 Ridge 结果此前已经看过，因此不是公开预注册。没有根据此次外层分数扩展搜索。原基线配置和模型未覆盖，src 源文件未修改。

全部训练和评估完成后，汇总脚本错误地将 cv_metrics.csv 当成 CV 目录。只修复报告枚举路径并重跑汇总，没有重训或改变模型、标签、预测和选择。原驱动脚本已归档，修复记录见 protocol_amendment_01.json；不得声称修复后的脚本哈希与最初锁定值相同。

整曲输入仍存在实际曝光偏差；歌曲跨分区重叠，只能解释为歌曲层级、未见参与者预测。没有建立因果或推荐效果证据；生理和 Valence 未超过简单基线的结果应如实纳入论文。

## 交付与后续

本轮已有固定验证和开发 CV 的逐记录／逐参与者／各种子／集成结果、配对差值区间及模型文件。尚未评估最终测试，也未进行 MERT 或精确播放片段补充分析。下一步应先冻结最终模型和评估方案，再一次性评估预留测试；可保留全部预先指定家族作为报告对象，不能按测试结果进一步调参。

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/song_level_mlp.py run
python scripts/report_song_level_mlp.py
```

匹配签名的训练和评估会复用缓存。最初 prepare 锁针对归档的训练驱动版本；当前汇总修复由单独 amendment 记录，重跑不覆盖初始锁。
'''
    (out/'mlp_report.md').write_text(md)
    folds=[results,*[p.parent for p in sorted(results.glob('cv_*/registry.json'))]]
    fit_count=sum(json.loads((p/'registry.json').read_text())['fit_count'] for p in folds)
    assert fit_count==144
    preds=pd.read_csv(results/'development_cv_ensemble_predictions.csv')
    final=set(json.loads((ROOT/'data/splits/participants.json').read_text())['groups']['test'])
    assert not final&set(preds.participant_id)
    assert preds.participant_id.nunique()==26 and preds.record_id.nunique()==660
    assert not (results/'evaluation_test.json').exists()
    write_json(out/'completion.json',{'formal_response_model_fits':fit_count,'formal_network_fits':216,
       'selected_model_files':sum(len(list(p.glob('*.pkl'))) for p in folds),
       'development_records':660,'development_participants':26,'final_holdout_evaluated':False,
       'report_source_sha256':digest(Path(__file__)),'ensemble_metrics_sha256':digest(results/'development_cv_ensemble_metrics.csv')})
    print('MLP report and completion checks saved.')

if __name__=='__main__':run()
