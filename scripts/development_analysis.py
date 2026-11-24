"""Training/development-only EDA and mean/song-mean diagnostics; no test label parsing."""
from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.common import load_config,folder,write_json,digest,assert_splits
from src.experiment import load_labels
from src.evaluate import compute_metrics,paired_bootstrap
TARGETS=['mean_hr','arousal','valence']


def load_development(cfg):
    base=folder(cfg,'processed')
    manifest=pd.read_csv(base/'manifest.csv',usecols=['record_id','participant_id','song_id','duration_s'],
                         dtype={'record_id':str,'participant_id':str,'song_id':str})
    split=json.loads((folder(cfg,'splits')/'participants.json').read_text())['groups']
    assert_splits(split)
    dev=manifest[manifest.participant_id.isin(split['train']+split['validation'])].copy()
    assert not set(dev.participant_id)&set(split['test'])
    dev['partition']=np.where(dev.participant_id.isin(split['train']),'train','validation')
    y=load_labels(cfg,dev.record_id.tolist(),TARGETS)
    for j,t in enumerate(TARGETS):
        dev[t]=y[:,j]
    return dev,split


def analyze(cfg):
    data,split=load_development(cfg)
    out=ROOT/'results/development'
    reports=ROOT/'reports/development'
    out.mkdir(parents=True,exist_ok=True);reports.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(reports/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data.to_csv(out/'development_labels.csv',index=False)
    descriptions=[]; people=[];correlations=[]; song_rows=[]
    for part,frame in data.groupby('partition'):
        for target in TARGETS:
            valid=frame[target].dropna()
            descriptions.append({'partition':part,'target':target,'records':len(frame),'valid':len(valid),'missing':int(frame[target].isna().sum()),
                'participants':int(frame.loc[frame[target].notna(),'participant_id'].nunique()),'mean':valid.mean(),'sd':valid.std(),
                'min':valid.min(),'q25':valid.quantile(.25),'median':valid.median(),'q75':valid.quantile(.75),'max':valid.max()})
            for pid,sub in frame.groupby('participant_id'):
                vals=sub[target].dropna()
                people.append({'partition':part,'participant_id':pid,'target':target,'n':len(vals),'mean':vals.mean(),'sd':vals.std(),'median':vals.median()})
        for a,b in itertools.combinations(TARGETS,2):
            sub=frame[['participant_id',a,b]].dropna()
            within=sub[[a,b]]-sub.groupby('participant_id')[[a,b]].transform('mean')
            correlations.append({'partition':part,'target_a':a,'target_b':b,'records':len(sub),
                'pooled_pearson':sub[a].corr(sub[b]),'pooled_spearman':sub[a].corr(sub[b],method='spearman'),
                'within_person_centered_pearson':within[a].corr(within[b]),'interpretation':'descriptive; repeated observations; no independent-row p-values'})
    train=data[data.partition=='train']
    for target in TARGETS:
        # Average repeated plays within participant/song before assessing listener differences.
        grouped=train.groupby(['song_id','participant_id'])[target].mean().dropna().reset_index()
        for song,sub in grouped.groupby('song_id'):
            if sub.participant_id.nunique()>=2:
                song_rows.append({'song_id':song,'target':target,'listeners':sub.participant_id.nunique(),
                    'mean':sub[target].mean(),'between_listener_sd':sub[target].std(),'range':sub[target].max()-sub[target].min()})
    pd.DataFrame(descriptions).to_csv(out/'label_summary.csv',index=False)
    pd.DataFrame(people).to_csv(out/'participant_summary.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'correlations.csv',index=False)
    pd.DataFrame(song_rows).to_csv(out/'shared_song_listener_variability.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,part in enumerate(['train','validation']):
        for ax,target in zip(axes[row],TARGETS):
            edges=np.histogram_bin_edges(data[target].dropna(),bins=25)
            ax.hist(data.loc[data.partition==part,target].dropna(),bins=edges)
            ax.set_xlim(edges[0],edges[-1])
            ax.set(title=f'{part}: {target}',ylabel='Listening records',xlabel='bpm' if target=='mean_hr' else 'rating [-10, 10]')
    fig.suptitle('HKU956 development labels — test excluded');fig.tight_layout();fig.savefig(reports/'label_distributions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,target in zip(axes,TARGETS):
        records=[s[target].dropna().to_numpy() for _,s in train.groupby('participant_id')]
        names=[pid for pid,_ in train.groupby('participant_id')]
        ax.boxplot(records,tick_labels=names,showfliers=False)
        ax.tick_params(axis='x',labelrotation=90);ax.set(title=f'Training listeners: {target}')
    fig.suptitle('Individual differences — descriptive, not causal');fig.tight_layout();fig.savefig(reports/'participant_distributions.png',dpi=160);plt.close(fig)
    # Complete-label cohort shared by both baselines, independent of audio recovery.
    complete=data.dropna(subset=TARGETS)
    fit=complete[complete.partition=='train'];val=complete[complete.partition=='validation']
    predictions=[];metrics=[];by_person=[]; means={}
    for target in TARGETS:
        global_mean=float(fit[target].mean());means[target]=global_mean
        song_means=fit.groupby('song_id')[target].mean()
        seen=val.song_id.isin(song_means.index)
        for name in ['mean','song_mean']:
            pred=np.repeat(global_mean,len(val)) if name=='mean' else val.song_id.map(song_means).fillna(global_mean).to_numpy()
            actual=val[target].to_numpy()
            for i,(_,row) in enumerate(val.iterrows()):
                predictions.append({'record_id':row.record_id,'participant_id':row.participant_id,'song_id':row.song_id,
                    'model':name,'seed':0,'target':target,'actual':actual[i],'predicted':pred[i],'split':'validation',
                    'song_seen_in_train':bool(seen.iloc[i]),'synthetic':False})
            for group,mask in [('all',np.ones(len(val),dtype=bool)),('seen_song',seen.to_numpy()),('unseen_song',~seen.to_numpy())]:
                if not mask.any():
                    continue
                sub=val.loc[mask].copy();sub['prediction']=pred[mask]
                person_errors=[]
                for pid,p in sub.groupby('participant_id'):
                    stat=compute_metrics(p[target].to_numpy(),p.prediction.to_numpy())
                    person_errors.append(stat['mae'])
                    by_person.append({'model':name,'target':target,'group':group,'participant_id':pid,**stat})
                metrics.append({'model':name,'target':target,'group':group,'macro_mae':float(np.mean(person_errors)),
                    'participants':len(person_errors),**compute_metrics(actual[mask],pred[mask])})
    predictions=pd.DataFrame(predictions);metrics=pd.DataFrame(metrics)
    predictions.to_csv(out/'baseline_predictions_validation.csv',index=False)
    metrics.to_csv(out/'baseline_metrics_validation.csv',index=False)
    pd.DataFrame(by_person).to_csv(out/'baseline_participant_metrics.csv',index=False)
    # Pair the diagnostics with four participant clusters, only as descriptive uncertainty.
    from src.evaluate import paired_bootstrap
    renamed=predictions.copy();renamed['model']=renamed.model.replace({'song_mean':'ridge'})
    boot=paired_bootstrap(renamed,repeats=2000,seed=17)
    boot=boot[(boot.model_a=='ridge') & (boot.model_b=='mean')].copy()
    boot['model_a']='song_mean';boot['comparison_note']='diagnostic song mean minus global mean; four validation clusters only'
    boot.to_csv(out/'baseline_paired_bootstrap.csv',index=False)
    manifest={'test_labels_used':False,'train_people':split['train'],'validation_people':split['validation'],
        'targets':TARGETS,'cohort':'complete labels, all songs regardless of audio availability','train_records':fit.record_id.tolist(),
        'validation_records':val.record_id.tolist(),'training_global_means':means,
        'validation_seen_song_records':int(val.song_id.isin(fit.song_id).sum()),'validation_records_count':len(val),
        'labels_file_sha256':digest(folder(cfg,'processed')/'labels.csv')}
    write_json(out/'protocol.json',manifest)
    report=['# 训练／验证集初步分析','', '**真实 HKU956 标签；测试集未分析。音频模型未训练。**','',
        f'完整标签基线：训练 {len(fit)} 条／{fit.participant_id.nunique()} 人，验证 {len(val)} 条／{val.participant_id.nunique()} 人。',
        f'验证集中 {manifest["validation_seen_song_records"]} 条记录的歌曲在训练集中出现；其余回退训练全局均值。','',
        '## 验证基线','', '|模型|目标|参与者宏平均 MAE|记录级 RMSE|','|---|---|---:|---:|']
    for r in metrics[metrics.group=='all'].itertuples():
        report.append(f'|{r.model}|{r.target}|{r.macro_mae:.4f}|{r.rmse:.4f}|')
    report+=['','mean_hr 单位 bpm，A/V 为原始评分单位；不能跨目标直接比较原始误差。',
        '这些结果使用全部完整标签记录，后续音频模型可能只有子集；正式比较必须在相同 cohort 上重跑基线。',
        '歌曲均值仅为共享歌曲诊断，不能推广为未见歌曲预测模型。',
        '四位验证参与者的 bootstrap 区间仅作描述性参考，不据此证明显著性。','',
        '## 个体差异与相关性','', '同歌曲差异先对同一参与者的重复播放取均值，再计算不同听众间标准差。',
        '相关性同时汇报记录级和个体内中心化版本，不做把重复记录当作独立样本的显著性检验。',
        '同歌曲间差异包含个人基线、上下文、测量噪声与刺激差异，不能视为纯音乐效应或理论预测上限。','',
        '数据表和运行协议位于 results/development/；图表位于 reports/development/。']
    tmean=float(train.mean_hr.mean()); vmean=float(data.loc[data.partition=='validation','mean_hr'].mean())
    report += ['', '## 本次数据中的初步发现', '',
        f'训练／验证集平均 mean-HR 为 {tmean:.3f} / {vmean:.3f} bpm，提示参与者人群偏移；不据此重抽划分。',
        f'验证集 {len(val)} 条中 {manifest["validation_seen_song_records"]} 条属于训练已见歌曲，其余为未见歌曲。',
        '歌曲均值相对全局均值的改善仅见 Arousal；HR 和 Valence 未改善。该结论仅描述当前验证诊断，不是正式音频模型结论。',
        '训练参与者内 HR 与 A/V 的中心化相关性均很弱，但不能推断所有音频模型都无预测能力。']
    (reports/'development_report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    print(metrics[metrics.group=='all'].to_string(index=False));print('Training/validation only. Test labels not parsed.')
    return data,metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(ROOT/'configs/default.yaml'))
    analyze(load_config(p.parse_args().config))
