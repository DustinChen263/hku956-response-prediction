"""Export scientific result figures for insertion in slides. No model fitting."""
from pathlib import Path
import json,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];O=R/'reports/midterm_visuals_20261103'
O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.titlesize':21,'axes.labelsize':17,'xtick.labelsize':15,'ytick.labelsize':16,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
navy='#17324D';teal='#168C91';blue='#3974B9';orange='#D98932';gray='#99A5B1';red='#B55A59'
models=['mean','ridge','single_mlp','multi_mlp'];names=['Mean baseline','Ridge','Single-task MLP','Three-task MLP'];colors=[gray,blue,teal,orange]
d=pd.read_csv(R/'reports/final_holdout_v1/development_vs_final.csv')
ret=pd.read_csv(R/'reports/quality_audit_20261101/retention_summary.csv')
pred=pd.read_csv(R/'reports/hr_diagnostics_20261103/record_predictions_diagnostic.csv')
variance=pd.read_csv(R/'reports/hr_diagnostics_20261103/variance_decomposition.csv')
pdf=PdfPages(O/'all_figures.pdf');files=[]
def base(title,subtitle):
 f=plt.figure(figsize=(16,9));f.text(.06,.93,title,fontsize=28,weight='bold',color=navy);f.text(.06,.875,subtitle,fontsize=16,color='#526373');return f

def save(f,name,foot):
 f.text(.06,.045,foot,fontsize=12,color='#526373',va='bottom')
 f.savefig(O/f'{name}.png',dpi=200);f.savefig(O/f'{name}.svg');pdf.savefig(f);files.append(name);plt.close(f)

f=base('Study data and evaluation','Whole-song audio features predict first-30-second mean HR and post-listening arousal / valence')
ax=f.add_axes([.27,.25,.62,.54]);labs=['All listening records','Records with recovered audio','Model cohort (at least 30 s)','Development evaluation','Final held-out evaluation'];nums=[956,773,771,660,111]
ax.barh(range(5),nums,color=[gray,blue,teal,teal,orange],height=.6);ax.set_yticks(range(5),labs);ax.invert_yaxis();ax.set_xlim(0,1100);ax.set_xlabel('Listening records');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for i,n in enumerate(nums):ax.text(n+15,i,str(n),va='center',weight='bold',color=navy)
f.text(.06,.14,'30 participants in total     26 in development     4 in the final test',fontsize=20,color=navy)
save(f,'01_data_and_evaluation','Evaluation sets separate participants. Songs may overlap. Development uses five-fold participant CV.\nAudio covers 460 of 592 songs. The final test was already opened on 28 October 2026.')

for field,tag,title,subtitle in [('final_test_macro_mae','02_final_results','Arousal improves, while HR and valence show limited gains','Final test: 111 records from 4 unseen participants'),('development_cv_macro_mae','03_development_results','Arousal shows the clearest gains in development','Development: 660 records from 26 participants, five-fold participant CV')]:
 f=base(title,subtitle)
 for i,(target,lab) in enumerate([('mean_hr','Mean HR (bpm)'),('arousal','Arousal (rating units)'),('valence','Valence (rating units)')]):
  ax=f.add_axes([.07+i*.315,.32,.265,.45]);vals=d[d.target.eq(target)].set_index('model').loc[models,field]
  ax.bar(range(4),vals,color=colors,width=.65);ax.set_xticks(range(4),['Mean','Ridge','Single','Three-task'],rotation=20);ax.set_title(lab,pad=18);ax.set_ylim(0,vals.max()*1.25);ax.set_ylabel('Participant-macro MAE' if i==0 else '');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
  for j,v in enumerate(vals):ax.text(j,v+.025*vals.max(),f'{v:.2f}',ha='center',fontsize=18,weight='bold',color=navy)
 f.text(.07,.18,'Lower error is better. Arousal and valence ratings range from −10 to 10.',fontsize=19,color=navy)
 save(f,tag,'Main comparisons shown. Neural results average predictions from three seeds. Each participant has equal metric weight.\nSeparate axes retain the original units. These descriptive results do not establish causal music effects.')

f=base('HR gains do not repeat across evaluations','Error reduction relative to the training-mean baseline within each evaluation')
ax=f.add_axes([.1,.29,.8,.47]);xx=np.arange(3);targets=['mean_hr','arousal','valence']
for j,(col,label,c) in enumerate([('development_cv_macro_mae','Development CV',blue),('final_test_macro_mae','Final test',teal)]):
 vals=[]
 for t in targets:
  q=d[d.target.eq(t)].set_index('model')[col];vals.append(100*(q['mean']-q['single_mlp'])/q['mean'])
 ax.bar(xx+(j-.5)*.28,vals,width=.28,label=label,color=c)
 for x,v in zip(xx+(j-.5)*.28,vals):ax.text(x,v+(1 if v>=0 else -1.2),f'{v:+.1f}%',ha='center',va='bottom' if v>=0 else 'top',fontsize=20,weight='bold')
ax.axhline(0,color=navy,lw=1);ax.set_xticks(xx,['Mean HR','Arousal','Valence']);ax.set_ylabel('MAE reduction (%)');ax.set_ylim(-6,29);ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
f.text(.1,.15,'Single-task MLP vs mean baseline. Positive values indicate lower error.',fontsize=19,color=navy)
save(f,'04_gains_vs_baseline','Development and final results involve different participants. Compare each model with its own evaluation baseline.\nRounded 0.0% for final valence reflects a negligible improvement, not a meaningful gain.')

f=base('HR predictions stay close to the average','Frozen single-task ensemble. Post-hoc diagnostics, with no new training')
for i,(scope,label,n) in enumerate([('development_cv','Development: 660 records',660),('final_test','Final test: 111 records',111)]):
 g=pred[(pred.model=='single_mlp')&(pred.evaluation==scope)];assert len(g)==n
 ax=f.add_axes([.09+i*.46,.27,.36,.49]);ax.scatter(g.actual,g.predicted,s=28,alpha=.55,color=blue if i==0 else teal,edgecolors='none');ax.plot([50,150],[50,150],'--',color=gray,label='Perfect prediction');ax.set(xlim=(50,150),ylim=(50,150),xlabel='Actual mean HR (bpm)',ylabel='Predicted mean HR (bpm)',title=label);ax.set_aspect('equal');ax.legend(frameon=False,fontsize=12)
 f.text(.09+i*.46,.14,f'Actual SD: {g.actual.std():.2f} bpm\nPredicted SD: {g.predicted.std():.2f} bpm',fontsize=18,color=navy)
save(f,'05_hr_prediction_scatter','Prediction spread is only 28–29% of actual spread. SD = standard deviation.\nDevelopment points come from different outer-fold models. The scatter is descriptive, not an estimate of a causal effect.')

f=base('HR varies widely between participants','Descriptive variance partition across 771 model records from 30 participants')
q=variance[variance.scope.eq('model_cohort_descriptive_only')].iloc[0];a=100*q.between_person_SS_fraction;b=100*q.within_person_SS_fraction
ax=f.add_axes([.08,.57,.84,.16]);ax.barh([0],[a],color=blue,height=.7);ax.barh([0],[b],left=[a],color=teal,height=.7);ax.text(a/2,0,f'{a:.1f}%\nBetween participant means',ha='center',va='center',color='white',fontsize=22,weight='bold');ax.text(a+b/2,0,f'{b:.1f}%\nWithin participants',ha='center',va='center',color='white',fontsize=18);ax.set_xlim(0,100);ax.axis('off')
f.text(.08,.40,'Within-person prediction correlation is also weak',fontsize=23,weight='bold',color=navy)
f.text(.08,.30,'Development: r = 0.052       Final test: r = 0.096',fontsize=24,color=teal)
f.text(.08,.19,'Removing personal means does not reveal strong tracking of record-to-record HR changes.',fontsize=17,color=navy)
save(f,'06_hr_individual_differences','Exact, record-weighted sums-of-squares partition, not causal attribution or an ICC. Song choice and context can contribute.\nWithin-person correlation subtracts each participant’s observed and predicted means for diagnosis only. No prediction calibration.')

f=base('IBI gaps limit the available RMSSD sample','First-30-second window. Illustrative E3 quality screening, not a universal validity rule')
ax=f.add_axes([.31,.32,.57,.45]);nums=[956,937,150,122];labs=['All listening records','Records with any IBI','Pass first-30-second E3 screening','Pass E3 and have audio'];ax.barh(range(4),nums,color=[gray,blue,teal,orange],height=.55);ax.set_yticks(range(4),labs);ax.invert_yaxis();ax.set_xlim(0,1100);ax.set_xlabel('Listening records');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for i,n in enumerate(nums):ax.text(n+15,i,str(n),va='center',weight='bold',fontsize=20)
f.text(.07,.18,'719 records have internal gaps. The final subset includes 22 people.',fontsize=20,color=navy)
save(f,'07_rmssd_availability','E3: ≥80% coverage, ≤3 s maximum gap, ≥20 valid adjacent pairs, ≤5% flagged intervals, and ≥30 s listening.\nNo missing-beat recovery, gap imputation or RMSSD training. Available audio does not verify playback alignment.')

# Native tabular source in Markdown, rendered table graphic for direct insertion.
f=base('Final test results: all model families','Participant-macro MAE. Final test: 111 records from 4 participants. Lower is better.')
order=['mean','song_mean','ridge','single_mlp','multi_mlp','matched_multi_mlp','av_mlp']
labelsmap=dict(zip(order,['Mean baseline','Song-mean reference','Ridge','Single-task MLP','Three-task MLP','Capacity comparison MLP','A/V two-task MLP']))
tab=d.pivot(index='model',columns='target',values='final_test_macro_mae').reindex(order)[['mean_hr','arousal','valence']]
ax=f.add_axes([.07,.24,.86,.55]);ax.axis('off');cells=[[labelsmap[k]]+[f'{v:.2f}' if pd.notna(v) else '—' for v in row] for k,row in tab.iterrows()]
t=ax.table(cellText=cells,colLabels=['Model','HR MAE (bpm)','Arousal MAE','Valence MAE'],cellLoc='center',colWidths=[.43,.19,.19,.19],bbox=[0,0,1,1]);t.auto_set_font_size(False);t.set_fontsize(18)
for (r,c),cell in t.get_celld().items():
 cell.set_edgecolor('white')
 if r==0:cell.set_facecolor(navy);cell.set_text_props(color='white',weight='bold')
 else:cell.set_facecolor('#EDF3F7' if r%2 else '#FFFFFF');cell.set_text_props(color=navy)
save(f,'08_full_results_table','Song-mean uses training labels for shared songs. The capacity comparison is not an exact parameter-matched control.\nThree-seed prediction ensembles for neural models. Test rankings do not justify selecting a new winner after viewing the test.')
pdf.close()
md='''# Midterm results: figures and speaking notes

图表使用英文，适合直接插入现有 PPT。PNG 为 3200×1800；SVG 可缩放。all_figures.pdf 可预览全部 8 页。

建议正文使用 01、02、04、05、07；03、06、08 可作备份页。所有图均读取已保存结果，没有重新训练。

| 图 | 中文讲解提示 | 英文讲解句 |
|---|---|---|
| 01 数据与评估 | 771 条模型数据；开发 26 人、最终 4 人，按人隔离但歌曲可重叠。 | We evaluated predictions for unseen participants, with possible song overlap. |
| 02 最终模型结果 | Arousal 有收益；HR 和 Valence 缺乏稳定改善。 | Audio features helped predict arousal, while gains for HR and valence were limited. |
| 03 开发集结果 | 简单 Ridge 对 Arousal 有效，神经网络没有稳定优势。 | A simple linear model was competitive with the neural models. |
| 04 相对基线改善 | 单任务 HR 最终好 5.3%，开发却略差；Arousal 两边都好。 | HR improvements did not repeat across the two evaluations. |
| 05 HR 散点 | 理想情况贴近虚线，实际预测集中在平均值附近。 | The model predicts a narrow HR range and misses high and low values. |
| 06 个体差异 | 79% 是描述性组间变异，不是个人基线的因果比例；组内相关也弱。 | Participant differences matter, but they do not fully explain the weak predictions. |
| 07 HRV 可用性 | 采用一套探索规则，仅 122 条同时通过且有音频；尚未训练 HRV。 | IBI gaps leave a small sample for exploratory RMSSD prediction. |
| 08 全模型表 | 备份回答其他模型、不能测试后挑最优、容量控制不严格。 | Additional models did not show consistent gains across all targets. |

## 最终结果表（复制到 PPT 后可编辑）

'''
md+=tab.rename(index=labelsmap,columns={'mean_hr':'HR MAE (bpm)','arousal':'Arousal MAE','valence':'Valence MAE'}).to_markdown(floatfmt='.4f')
md+='''

## 数据表

| Item | Records | Participants |
|---|---:|---:|
| Full dataset | 956 | 30 |
| Model cohort | 771 | 30 |
| Fixed training set | 538 | 22 |
| Fixed validation set | 122 | 4 |
| Development CV (training + validation participants) | 660 | 26 |
| Final test | 111 | 4 |
| First-30-second E3 RMSSD subset | 150 | 23 |
| E3 RMSSD subset with audio | 122 | 22 |

## 必须保留的解释

- 输入：37 维整曲声学特征。输出：开始听歌后最初 30 秒的绝对 mean-HR，以及听完后的 A/V。
- 这不是 HR 作为输入的情绪识别，也不是音乐造成 HR 变化的因果检验。
- MAE 对参与者等权；A/V 原范围 −10 至 10。不能把 bpm 与评分单位直接相加。
- 神经网络结果是三种子预测平均后的集成结果。
- 最终测试只有 4 人，不能把 111 条重复记录当作 111 名独立受试者。图中没有展示推断性显著性。
- 组间变异 79% 不等于静息基线解释了 79%；选歌、环境等也可能贡献。
- 没有发现可用听歌前基线。连续原始信号、播放位置与 IBI 缺口来源仍待作者确认。
- E3 不是临床通用阈值。122 条有音频也不意味着实际播放片段完全对齐。

## Sources in this project

- reports/final_holdout_v1/development_vs_final.csv
- reports/hr_diagnostics_20261103/record_predictions_diagnostic.csv
- reports/hr_diagnostics_20261103/variance_decomposition.csv
- reports/quality_audit_20261101/retention_summary.csv
- reports/quality_audit_20261101/summary.json

Rebuild: `MPLCONFIGDIR=/private/tmp/hku956_mpl python scripts/create_midterm_visuals.py`
'''
(O/'README_speaking_notes.md').write_text(md)
# Validate all displayed retention figures against source data.
x=ret[(ret.window=='first30')&(ret.policy=='E3_stringent')&(ret.scope=='all')].iloc[0]
assert (x.retained_records,x.with_audio_records,x.with_audio_participants)==(150,122,22)
s=json.loads((R/'reports/quality_audit_20261101/summary.json').read_text());assert (s['records'],s['empty_ibi_records'],s['raw_gap_records'])==(956,19,719)
with zipfile.ZipFile(O/'midterm_figures_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(O.iterdir()):
  if p.suffix!='.zip':z.write(p,p.name)
print('Created',len(files),'figures with PNG/SVG, PDF, speaking notes and ZIP.')
