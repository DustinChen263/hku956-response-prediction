"""Post-hoc descriptive diagnostics of frozen predictions; no fitting or correction."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/hr_diagnostics_20261103'; OUT.mkdir(exist_ok=True)
paths=['results/song_level_mlp_v1/development_cv_ensemble_predictions.csv','results/final_holdout_v1/ensemble_predictions.csv','data/processed/author_v2/record_alignment.csv','data/processed/song_level_v1/labels.csv','data/processed/author_v2/exposure_intervals.csv']
provenance=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in paths]
a=pd.read_csv(ROOT/paths[2]); labels=pd.read_csv(ROOT/paths[3])
parts=[]
for path,scope in zip(paths[:2],['development_cv','final_test']):
 d=pd.read_csv(ROOT/path);d=d[d.target.eq('mean_hr')].copy();d['evaluation']=scope;parts.append(d)
p=pd.concat(parts,ignore_index=True)
assert not p.duplicated(['evaluation','model','record_id']).any()
p=p.merge(a,on=['record_id','participant_id','song_id'],validate='many_to_one')
p['error']=p.predicted-p.actual;p['abs_error']=p.error.abs()
p['pause_any']=np.where(p.paused_wall_s>0,'yes','no')
p['seek_status']=np.where(p.exposure_status.eq('zero_start_consistent_under_unverified_assumption'),'zero_start_consistent_unverified','other_ambiguous_or_conflict')
p['duration_bin']=pd.cut(p.published_duration_s,[0,45,60,np.inf],labels=['30-45s','>45-60s','>60s']).astype(str)
iv=pd.read_csv(ROOT/paths[4]).merge(a[['record_id','wall_start_unix_s']],on='record_id',suffixes=('','_record'),validate='many_to_one')
iv['overlap_first30']=np.maximum(0,np.minimum(iv.wall_end_unix_s,iv.wall_start_unix_s_record+30)-np.maximum(iv.wall_start_unix_s,iv.wall_start_unix_s_record))
paused=iv[iv.state.eq('paused')].groupby('record_id').overlap_first30.sum()
p['pause_first30_s']=p.record_id.map(paused).fillna(0)
p['pause_first30']=np.where(p.pause_first30_s>0,'yes','no')
p.to_csv(OUT/'record_predictions_diagnostic.csv',index=False)
# Descriptive variance partition: exact sums of squares, not causal variance components or ICC.
def decomposition(d,name):
 means=d.groupby('participant_id').actual.transform('mean');grand=d.actual.mean()
 total=float(((d.actual-grand)**2).sum());within=float(((d.actual-means)**2).sum());between=float(((means-grand)**2).sum())
 assert np.isclose(total,within+between)
 return dict(scope=name,records=len(d),participants=d.participant_id.nunique(),total_sd=d.actual.std(),between_person_SS_fraction=between/total,within_person_SS_fraction=within/total,within_person_rms=np.sqrt(within/len(d)),participant_mean_min=means.min(),participant_mean_max=means.max())
unique=p[p.model.eq('single_mlp')].copy();assert len(unique)==771
all_lab=labels.merge(a[['record_id','participant_id']],on='record_id',validate='one_to_one')
all_lab=all_lab[all_lab.mean_hr.notna()].rename(columns={'mean_hr':'actual'})
decs=[decomposition(all_lab,'all_available_HR_labels'),decomposition(unique,'model_cohort_descriptive_only')]
decs += [decomposition(d,s) for s,d in unique.groupby('evaluation')]
pd.DataFrame(decs).to_csv(OUT/'variance_decomposition.csv',index=False)
pr=[];mr=[]
for (scope,model),d in p.groupby(['evaluation','model']):
 ac=d.actual-d.groupby('participant_id').actual.transform('mean');pc=d.predicted-d.groupby('participant_id').predicted.transform('mean')
 corr=lambda x,y: float(x.corr(y)) if x.std()>1e-10 and y.std()>1e-10 else np.nan
 mr.append(dict(evaluation=scope,model=model,n=len(d),participants=d.participant_id.nunique(),actual_sd=d.actual.std(),predicted_sd=d.predicted.std(),sd_ratio=d.predicted.std()/d.actual.std(),correlation=corr(d.actual,d.predicted),within_person_correlation=corr(ac,pc),within_person_sd_ratio=pc.std()/ac.std(),macro_mae=d.groupby('participant_id').abs_error.mean().mean(),record_mae=d.abs_error.mean()))
 for pid,g in d.groupby('participant_id'):
  pr.append(dict(evaluation=scope,model=model,participant_id=pid,n=len(g),actual_mean=g.actual.mean(),actual_sd=g.actual.std(),predicted_mean=g.predicted.mean(),predicted_sd=g.predicted.std(),bias=g.error.mean(),mae=g.abs_error.mean(),mse=(g.error**2).mean(),bias_squared=g.error.mean()**2,error_variance=(g.error**2).mean()-g.error.mean()**2,within_person_correlation=corr(g.actual,g.predicted)))
pm=pd.DataFrame(pr);pm.to_csv(OUT/'participant_diagnostics.csv',index=False)
mm=pd.DataFrame(mr);mm.to_csv(OUT/'prediction_diagnostics.csv',index=False)
# Exposure associations: both pooled and participant-macro, plus within-person paired subgroup contrasts.
group_rows=[];contrasts=[]
for (scope,model),d in p.groupby(['evaluation','model']):
 for feature in ['pause_any','pause_first30','seek_status','duration_bin']:
  for level,g in d.groupby(feature):
   group_rows.append(dict(evaluation=scope,model=model,feature=feature,level=level,n=len(g),participants=g.participant_id.nunique(),mean_actual=g.actual.mean(),record_mae=g.abs_error.mean(),participant_macro_mae=g.groupby('participant_id').abs_error.mean().mean()))
  if feature!='duration_bin':
   low='no' if feature.startswith('pause') else 'zero_start_consistent_unverified';high='yes' if feature.startswith('pause') else 'other_ambiguous_or_conflict'
   tab=d.groupby(['participant_id',feature]).abs_error.mean().unstack()
   if low in tab and high in tab:
    pair=tab[[low,high]].dropna();delta=pair[high]-pair[low]
    contrasts.append(dict(evaluation=scope,model=model,feature=feature,contrast=f'{high} minus {low}',paired_participants=len(pair),mean_within_person_mae_difference=delta.mean(),median_difference=delta.median(),participants_positive=int((delta>0).sum())))
pd.DataFrame(group_rows).to_csv(OUT/'exposure_subgroups.csv',index=False)
pd.DataFrame(contrasts).to_csv(OUT/'exposure_within_participant_contrasts.csv',index=False)
# Direct inspection of all raw HR files and their correspondence to logged windows.
raw=ROOT/'data/raw/HKU956/HKU956/1. physiological_signals';hrrows=[]
for row in a.itertuples():
 path=raw/row.participant_id/'HR'/f'{row.song_no}_{row.song_id}.csv'
 v=np.loadtxt(path,ndmin=1)
 hrrows.append(dict(record_id=row.record_id,path=str(path.relative_to(ROOT)),samples=len(v),log_duration_s=row.log_duration_s,extra_samples_beyond_listening_window=len(v)-row.log_duration_s,finite=bool(np.isfinite(v).all()),raw_first30_mean=float(v[:30].mean()) if len(v)>=30 else np.nan,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
hr=pd.DataFrame(hrrows);check=unique[['record_id','actual']].merge(hr,on='record_id',validate='one_to_one');assert np.allclose(check.actual,check.raw_first30_mean,atol=1e-10)
hr.to_csv(OUT/'raw_hr_baseline_inventory.csv',index=False)
files=[]
for root in [ROOT/'data/raw/HKU956/HKU956',ROOT/'data/raw/author_supplement_20261028']:
 for path in sorted(root.rglob('*')):
  if path.is_file(): files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,suffix=path.suffix))
pd.DataFrame(files).to_csv(OUT/'source_file_inventory.csv',index=False)
# Session gaps exist in event times, but are not additional HR samples.
ordered=a.sort_values(['participant_id','session_id','wall_start_unix_s']).copy()
ordered['previous_record_end']=ordered.groupby(['participant_id','session_id']).wall_end_unix_s.shift()
ordered['gap_from_previous_record_s']=ordered.wall_start_unix_s-ordered.previous_record_end
ordered[['record_id','participant_id','session_id','wall_start_unix_s','previous_record_end','gap_from_previous_record_s']].to_csv(OUT/'between_record_time_gaps.csv',index=False)
events=pd.read_csv(ROOT/'data/processed/author_v2/sanitized_events.csv');events.event.value_counts().rename_axis('event').reset_index(name='count').to_csv(OUT/'log_event_inventory.csv',index=False)
# Plots: use only frozen single-task predictions, keep evaluations separate.
fig,axes=plt.subplots(2,3,figsize=(15,9))
for i,(scope,d) in enumerate(unique.groupby('evaluation',sort=False)):
 ax=axes[i,0];ax.scatter(d.actual,d.predicted,s=12,alpha=.45);lo=min(d.actual.min(),d.predicted.min());hi=max(d.actual.max(),d.predicted.max());ax.plot([lo,hi],[lo,hi],'k--');ax.set(xlabel='Actual mean HR (bpm)',ylabel='Predicted mean HR (bpm)',title=scope)
 ax=axes[i,1];ax.hist(d.actual,bins=18,alpha=.5,label='Actual');ax.hist(d.predicted,bins=18,alpha=.5,label='Predicted');ax.legend();ax.set(xlabel='Mean HR (bpm)',ylabel='Records',title='Distribution spread')
 g=pm[(pm.evaluation==scope)&(pm.model=='single_mlp')].sort_values('actual_mean');ax=axes[i,2];x=np.arange(len(g));ax.errorbar(x,g.actual_mean,yerr=g.actual_sd,fmt='o',label='Actual mean ± within-person SD');ax.scatter(x,g.predicted_mean,marker='x',label='Predicted mean');ax.set_xticks(x,g.participant_id.str.replace('hku',''),rotation=90);ax.legend(fontsize=7);ax.set(ylabel='Mean HR (bpm)',title='Participants sorted by actual mean')
fig.suptitle('Frozen single-task ensemble: post-hoc HR diagnostics');fig.tight_layout();fig.savefig(OUT/'prediction_diagnostics.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,(scope,g) in zip(axes,pm[pm.model.eq('single_mlp')].groupby('evaluation',sort=False)):
 g=g.sort_values('mae');ax.bar(g.participant_id,g.mae);ax.tick_params(axis='x',rotation=90);ax.set(title=scope,ylabel='Participant MAE (bpm)')
fig.tight_layout();fig.savefig(OUT/'participant_errors.png',dpi=170);plt.close(fig)
summary=dict(training_performed=False,predictions_modified=False,test_set_previously_opened=True,descriptive_only=True,records_verified_against_raw_HR=len(check),raw_hr_files=len(hr),raw_hr_files_matching_logged_duration=int((hr.extra_samples_beyond_listening_window==0).sum()),raw_hr_files_with_extra_samples=int((hr.extra_samples_beyond_listening_window>0).sum()),records_with_positive_between_record_gap=int((ordered.gap_from_previous_record_s>0).sum()),source_files_inventoried=len(files),provenance=provenance)
(OUT/'verification.json').write_text(json.dumps(summary,indent=2))
print(pd.DataFrame(decs).to_string(index=False));print(mm[mm.model.isin(['mean','ridge','single_mlp','multi_mlp'])].to_string(index=False));print(pd.DataFrame(contrasts).query("model=='single_mlp'").to_string(index=False));print(json.dumps(summary,indent=2))
# Human-readable report, regenerated with every execution.
def table(df):
 return df.to_markdown(index=False,floatfmt='.3f')
sel=mm[mm.model.isin(['mean','ridge','single_mlp','multi_mlp'])]
sub=pd.DataFrame(group_rows);sub=sub[sub.model.eq('single_mlp')]
con=pd.DataFrame(contrasts);con=con[con.model.eq('single_mlp')]
report='''# HKU956 心率预测诊断报告

日期：2026-11-03。性质：已开封测试集之后的探索性、描述性分析。

## 结论

当前证据更支持“缺少个体信息、预测收缩及任务/曝光定义限制共同影响结果”，而不是单纯归因于模型太弱。不能由预测失败推断音乐没有生理影响。

在 771 条实际模型记录中，约 79.0% 的 HR 总离差平方和位于参与者平均水平之间。单任务模型预测的标准差仅为真实值的约 28%–29%。但去掉每个人的均值后，预测与真实值的相关性仍很弱（开发集 0.052、最终测试 0.096）：个体均值差异并非唯一问题，现有模型也未能较好追踪同一个人不同记录的 HR 波动。

现有发布文件没有发现可直接用于每条记录的听歌前 HR 基线。建议保留现有 HR 结果作为有边界的探索性结果，以 Arousal 作为目前预测证据最强的目标；若继续以生理变化为主目标，应先争取连续原始记录、独立听歌前基线及准确播放位置。

## 1. 范围、复核与信息边界

- 只读取冻结的开发集五折外层预测（660 条、26 人）及最终测试预测（111 条、4 人），分别报告，未训练或选择模型。
- 单任务/多任务神经模型为既有三个种子的预测集成；本次未改变预测或集成方式。
- 逐条重新读取 956 个原始 HR 文件；模型 cohort 的 771 个标签均与对应原始前 30 个 1 Hz 样本均值相符。
- 模型输入是 37 维整曲音频特征，不包括个人基线；标签是听歌最初 30 秒的绝对平均 HR，不是音乐导致的 HR 变化。
- 对参与者均值进行的分析仅用于描述；没有用测试标签校准预测。下述相关性、方差分解不是因果效应或新的模型成绩。
- 源文件哈希保存在 verification.json；所有新产物位于独立报告目录。

## 2. 个体差异有多大

'''+table(pd.DataFrame(decs))+'''

`between_person_SS_fraction` 是按记录加权的描述性平方和分解，且总平方和严格等于组间加组内平方和。不是随机效应 ICC、遗传/生理差异的因果比例，也不是音频解释率。每个人听的歌曲和情境不同，组间部分可能同时包含选歌与环境差异。

全部可用的 954 条 30 秒 HR 标签中，该比例为 80.7%；模型 cohort 为 79.0%。cohort 各参与者均值范围为 57.37–95.50 bpm，组内均方根偏离为 5.15 bpm。分布确实存在显著的描述性个体差别，但不能把全部组间差异都归因于静息心率。

## 3. 模型是否主要预测均值附近

'''+table(sel[['evaluation','model','actual_sd','predicted_sd','sd_ratio','within_person_correlation','within_person_sd_ratio','macro_mae']])+'''

单任务开发集：真实标准差 11.57 bpm，预测标准差 3.21 bpm；最终测试：8.39 对 2.44 bpm。预测明显收缩，难以覆盖较低或较高的个人平均水平。

在每个人内部对真实值与预测值分别去均值后，单任务预测标准差约为真实组内标准差的 51%–55%，相关性仍很弱。因此即使给出事后个人均值，现有预测的记录间排序也没有呈现强关系。这不能模拟合法的基线校准，也不能用来声称已经训练了 ΔHR 模型。

开发集预测来自不同外层折的不同模型，整体相关性还受折间预测水平影响，不能单凭整体负相关宣称音乐和 HR 负相关；主要解释应结合分折结构与组内结果。均值基线在开发集也有非零标准差，因为每折训练均值不同。

逐人误差分解显示，参与者宏平均 MSE 中平均偏差平方占比约为开发集 80.0%、最终测试 75.9%。这是误差描述，不代表能在部署时消除这一比例。最终测试仅 4 人，单人高误差会明显影响汇总。

![预测与真实分布](prediction_diagnostics.png)

![各参与者误差](participant_errors.png)

## 4. 暂停、对齐疑点和时长与误差

本节以冻结的单任务预测为主要诊断对象。区分“整次聆听中任意暂停”和“标签前 30 秒中暂停”；后者通过日志区间与前 30 秒取交集计算。对齐分组仅是日志证据分类，零起点相容组也没有得到精确播放位置验证。

'''+table(sub[['evaluation','feature','level','n','participants','record_mae','participant_macro_mae']])+'''

不同分组包含的人不同，直接比较总体均值可能被人员构成影响。下面只纳入同一参与者同时拥有两类记录的人，先求每个人两组 MAE 差，再对人等权平均；仍未控制选歌、活动、时序或时长等混杂因素。

'''+table(con[['evaluation','feature','paired_participants','mean_within_person_mae_difference','participants_positive']])+'''

主要观察：

- 开发集有 95/660 条记录发生过暂停；24 名有两类记录的参与者中，19 人暂停组误差更高，平均差 +2.36 bpm。
- 最终测试有 21/111 条暂停记录；仅 3 人可做组内比较，三人方向均为暂停组更差，平均差 +2.91 bpm。注意按记录汇总时暂停组反而误差较低，说明人员构成足以改变表面方向，不能只报有利数字。
- 真正与前 30 秒标签重叠的暂停：开发集仅 35 条/13 人，最终测试仅 2 条/2 人。后者不能支持稳健结论。
- 对齐疑点组在开发集组内平均差仅 +0.11 bpm；最终只有 3 条疑点记录、1 人。无法确定跳转/对齐疑点是主要误差来源。
- 时长组是预先在本次描述分析中固定的 30–45、>45–60、>60 秒；误差没有稳定单调关系。时长为整次听歌时长，目标始终是前 30 秒。

暂停关联是值得后续验证的线索，不是因果结论；也不能据此事后删除记录并把更好的结果称为新的独立测试成绩。即使无暂停且零起点相容，也不代表实际曝光位置已验证。

## 5. 能否获得可信的听歌前 HR 基线

检查范围：本地原始 HKU956 全部文件与作者补充目录，共 4,788 个文件；包含 956×5 个分段生理文件、数据说明/标签/个性资料、原始日志和时间戳 IBI 附件及下载元数据。

- 所有 956 个 HR 文件的样本数都恰好等于对应日志听歌窗口的秒数；未发现窗口以外的额外 HR 样本。
- 原始 HR 是单列数值，不含可用于重新定位到完整连续会话的独立时间戳。当前对齐依赖作者说明和已匹配的窗口日志。
- 会话内有 925 条记录的开始晚于上一条记录结束，日志证明存在间隔，但这些间隔没有对应的已发布 HR 样本。不能把日志时间当作生理信号。
- 时间戳 IBI 仍然是同一套按听歌记录分段的附件，已确认没有新增心搏；不能补成连续的听歌前 HR。
- 当前目录中没有发现会话级连续 E4 导出、明确静息基线或每条听歌前窗口。结论仅限现有本地材料，不能证明作者没有保留额外原始数据。

因此，当前无法可靠构建“听歌期间 HR 减听歌前未播放音乐的 HR”。不应使用本次听歌最初几秒、上一首歌结束值或同一人的全数据均值替代独立基线；这些做法回答不同问题，且后者若用于测试校准会泄漏标签。

## 6. 决策与下一步

1. 先向作者确认是否还有连续原始 E4 HR/BVP/IBI、设备绝对起始时间、听歌前未播放音乐区间、完整播放/跳转位置日志。仅准备问题，本次未发送邮件。
2. 若获得这些数据，先验证时间同步、基线覆盖和信号质量，再制定 ΔHR 的新协议。基线必须在预测时可获得；若作为输入，需明确研究已变为“音频加个人基线”的条件预测。
3. 若无法补齐，把 Arousal 作为现有证据最强的预测结果；保留 HR 与 Valence 的有限/不稳定收益，讨论输入信息与任务定义限制。HRV 继续限定子集探索，不能默认换目标即可改善结果。
4. 新音频表征或模型可在开发集内部进行有界探索，但不能承诺提高 HR 表现，也不能再将已经查看过的最终四人作为未使用过的确认性测试。若研究要主张新方法有效，需要新的独立确认设计。

停止条件：没有独立基线和可靠对齐，就不把当前数据中的绝对 HR 改名为音乐诱发变化；没有新独立验证，就不靠反复调整当前测试集选出“最佳模型”。

## 7. 复现及文件

运行：`MPLCONFIGDIR=/private/tmp/hku956_mpl python scripts/diagnose_hr_20261103.py`。

- variance_decomposition.csv：总体/组间/组内描述。
- prediction_diagnostics.csv、participant_diagnostics.csv：预测收缩、相关性和逐人误差。
- exposure_subgroups.csv、exposure_within_participant_contrasts.csv：全部模型的曝光分组结果。
- raw_hr_baseline_inventory.csv、source_file_inventory.csv、between_record_time_gaps.csv、log_event_inventory.csv：基线检索证据。
- record_predictions_diagnostic.csv：冻结预测与日志属性的连接结果。
- verification.json：范围、标签复核及输入哈希。

无新增模型训练、无 HRV 推算、无长缺口插补、无预测校准、无原始结果覆盖。
'''
(OUT/'diagnostic_report.md').write_text(report)
# Verify sources remain unchanged after all computations.
assert all(hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in provenance)
