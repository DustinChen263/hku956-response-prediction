"""Render technical linkage/QC evidence without reading emotion labels or fitting models."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.common import digest,write_json


def run():
    report=ROOT/'reports/author_v2';processed=ROOT/'data/processed/author_v2'
    summary=json.loads((report/'summary.json').read_text())
    records=pd.read_csv(processed/'record_alignment.csv');intervals=pd.read_csv(processed/'exposure_intervals.csv')
    qc=pd.read_csv(processed/'hrv_qc_all_windows.csv');first=qc[qc.window=='wall_first30']
    people=pd.read_csv(report/'participant_hrv_coverage.csv')
    counts=first.assign(passed=first.rmssd_quality=='pass').groupby('partition').agg(records=('record_id','size'),passed=('passed','sum')).reindex(['train','validation','test'])
    valid_people=first[first.rmssd_quality=='pass'].groupby('partition').participant_id.nunique()
    available=set(records.loc[records.audio_available,'record_id'])
    joint=pd.read_csv(processed/'joint_evidence_and_qc.csv')
    conditional_audio=int((joint.conditional_media_consistency & joint.audio_available).sum())
    conditional_joint=int((joint.conditional_media_consistency & joint.audio_available & joint.physiology_qc_pass & joint.no_logged_pause_in_first30).sum())
    cohorts=[]
    for part,frame in joint.groupby('partition'):
        for name,mask in [('matched_records',pd.Series(True,index=frame.index)),('audio_available',frame.audio_available),
            ('hrv_first30_pass',frame.physiology_qc_pass),('audio_and_hrv',frame.audio_available & frame.physiology_qc_pass),
            ('conditional_media_audio',frame.conditional_media_consistency & frame.audio_available),
            ('conditional_media_audio_hrv_no_pause',frame.conditional_media_consistency & frame.audio_available & frame.physiology_qc_pass & frame.no_logged_pause_in_first30),
            ('strict_verified_audio',frame.strict_audio_training_eligible)]:
            sub=frame.loc[mask];cohorts.append({'partition':part,'cohort':name,'records':len(sub),'participants':sub.participant_id.nunique(),'training_enabled':False})
    pd.DataFrame(cohorts).to_csv(report/'conditional_cohort_counts.csv',index=False)
    anchored=qc[qc.window.str.startswith('anchored_bout')]
    os.environ.setdefault('MPLCONFIGDIR',str(report/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    x=np.arange(3)
    axes[0].bar(x,counts.records,color='#d7e0e9',label='Matched records')
    axes[0].bar(x,counts.passed,color='#237c9b',label='First-30s IBI QC pass')
    axes[0].set_xticks(x,counts.index);axes[0].set_ylabel('Listening records');axes[0].legend()
    for k,v in enumerate(counts.itertuples()):axes[0].text(k,v.passed+7,str(v.passed),ha='center')
    labels=['Initial-start consistent\n(unverified assumption)','Partial / seek ambiguity','State conflict','Two-anchor consistency\n(not continuous observation)']
    values=[summary['exposure_statuses'].get(k,0) for k in ['zero_start_consistent_under_unverified_assumption','partial_or_seek_ambiguous','state_sequence_conflict','two_anchor_consistent_not_continuously_observed']]
    axes[1].barh(labels,values,color=['#6f9eb3','#d49a59','#b96969','#6b9b8a']);axes[1].invert_yaxis();axes[1].set_xlabel('Listening records')
    for k,v in enumerate(values):axes[1].text(v+6,k,str(v),va='center')
    axes[1].set_xlim(0,max(values)*1.12)
    fig.suptitle('HKU956 author supplement: linkage is not exposure verification')
    fig.tight_layout();fig.savefig(report/'alignment_and_qc.png',dpi=160);plt.close(fig)
    # A metadata-only timeline example explaining the original length discrepancy.
    rid='hku1919_32_423152';sub=intervals[intervals.record_id==rid];origin=sub.wall_start_unix_s.min()
    fig,ax=plt.subplots(figsize=(10,2.4))
    for r in sub.itertuples():
        ax.barh(0,r.wall_duration_s,left=r.wall_start_unix_s-origin,height=.4,color='#237c9b' if r.state=='playing' else '#d49a59')
        if r.wall_duration_s>=6:ax.text((r.wall_start_unix_s+r.wall_end_unix_s)/2-origin,0,f'{r.state}\n{r.wall_duration_s:g}s',ha='center',va='center',color='white',fontsize=9)
    ax.set_xlim(0,122);ax.set_yticks([]);ax.set_xlabel('Elapsed wall-clock seconds from record start')
    ax.set_title('hku1919_32_423152: 122s elapsed includes 7s pause; audio is about 114s')
    fig.tight_layout();fig.savefig(report/'pause_timeline_example.png',dpi=160);plt.close(fig)
    lines=['# 作者补充数据：对齐与 HRV 质量审计','',
        '日期：2026-10-28。仅做时间匹配、曝光证据与生理 QC；没有读取主观 A/V 标签或进行模型拟合。','',
        '## 1. 记录匹配已经完成','',
        f"评分记录 {summary['records']} 条全部找到唯一日志区间；日志区间未重复使用。所有配对的日志墙钟长度与发布 play_duration 完全一致（秒级）。",
        '955 条以 song-start 开始；hku1927_0_226749 以新会话中的 song-play 续播作为起点，68 秒区间同时得到 IBI 时间范围支持，单列在 resume_boundary_cases.csv。',
        '日志按 Asia/Hong_Kong（2019 年 UTC+8）解释，IBI 时间为 Unix 秒。并非以首个检出心搏作为播放起点。',
        f"956 个新文件的 IBI 值与原发布文件逐值一致（包含 {summary['empty_ibi_records']} 个空文件）。相邻时间差与后一条 IBI 在 2ms 内一致的比例为 {100*summary['ibi_next_interval_match_fraction_2ms']:.2f}%，支持按间期结束时刻解释时间戳；正向不连续处保留为缺口，不插补。",'',
        '## 2. 曝光重建需要区分证据','',
        f"有 {summary['paused_records']} 条记录含可见的正时长暂停。122 秒／约 114 秒异常中的约 7 秒暂停已由日志直接解释，不能继续要求音频长度覆盖整个墙钟区间。",'',
        '|证据状态|记录数|含义|','|---|---:|---|']
    meanings={
        'zero_start_consistent_under_unverified_assumption':'端点与从零连续播放假设一致，但起点未直接记录；不能证明没有隐藏跳转',
        'partial_or_seek_ambiguous':'存在位置／时间不一致、初始位置不明、缺位置或可见跳变等',
        'state_sequence_conflict':'重复暂停等状态序列冲突，需人工核验',
        'two_anchor_consistent_not_continuously_observed':'两端位置与时间一致，但没有连续播放轨迹证明'}
    for name,count in summary['exposure_statuses'].items():lines.append(f'|{name}|{count}|{meanings[name]}|')
    lines += ['',
        '因此“日志匹配成功”不等于“实际音频逐样本重建成功”。所有 alignment_verified 保持 false；不将推测起点或播放路径偷偷写入旧 manifest。',
        f'并不是所有记录都失去研究价值：{conditional_audio} 条已有音频且与连续播放假设相容；其中 {conditional_joint} 条同时通过首 30 秒 HRV QC、该窗口无已记录暂停。它们可作为明确标注假设的候选子集，仍不是自动确认的训练样本。',
        f"有 {len(anchored)} 个至少 30 秒、两端位置一致的播放区间（来自 {anchored.record_id.nunique()} 条记录），可作条件重建候选；其中 {int((anchored.rmssd_quality=='pass').sum())} 个通过当前 IBI QC，{int(((anchored.rmssd_quality=='pass') & anchored.record_id.isin(available)).sum())} 个已有音频。端点一致仍依赖区间内无隐藏 seek、播放速率为 1 的假设。",'',
        '## 3. HRV 可计算性：仍不支持全员完整标签主任务','',
        '沿用 v1 预设工程阈值：间期 300–2000 ms，相邻变化 ≤20%，有效相邻对 ≥20，覆盖率 ≥80%，最长缺口 ≤3s，连续性容差 0.04s；窗口至少 30 秒。没有为提高通过率调参。',
        f"固定开始后 30 秒墙钟窗口：{summary['first30_hrv_pass']}/956 条通过（{summary['first30_hrv_pass']/956:.1%}）；完整墙钟窗口：{summary['full_hrv_pass']}/956 条通过。它们是不同的测量目标，不逐条挑选更容易通过的窗口。",'',
        '|分区|记录|前 30 秒 QC 通过|有通过记录的参与者|','|---|---:|---:|---:|']
    for part,r in counts.iterrows():lines.append(f'|{part}|{int(r.records)}|{int(r.passed)}|{int(valid_people.get(part,0))}|')
    lines += ['',
        f"前 30 秒通过记录中 {summary['first30_pass_with_audio']} 条已有候选音频，尚未叠加严格曝光验证。",
        f"首 30 秒有 {int((first.pause_overlap_s>0).sum())} 条含已记录暂停，其中 {int(((first.pause_overlap_s>0)&(first.rmssd_quality=='pass')).sum())} 条通过 IBI QC；生理质量通过不代表这 30 秒全在听音乐。",'',
        '验证集 51 条通过记录中的 43 条来自同一位参与者。HRV 缺失明显不均衡，不能当作随机缺失；不删除无合格记录的参与者后继续宣称原全员评估，也不以选择更好成绩为由重抽划分。',
        '以上仅表示通过当前技术筛查，不是 ECG 等价性或临床有效性验证。PPG 衍生间期的 RMSSD 需说明为脉搏间期变异指标。',
        '测试分区仅参与预设质量筛查和元数据配对；未查看预测成绩、调模型或汇报测试目标分布。','',
        '## 4. 下一版研究范围建议','',
        '保留 mean-HR／A/V 的主线；HRV 暂作有掩码的辅助目标或明确限定的探索性子研究。不能直接将 HRV 改回覆盖 30 人的完整标签主目标。',
        '音频模型有两个需明确区分的定义：歌曲整体特征预测自然聆听反应（承认曝光位置不同）；或在有充分播放证据的区间上做片段模型。前者不宣称重建真实刺激，后者不能用零起点猜测填充所有记录。',
        '协议 v2 草案记录了两者的门槛，当前仍未自动选择、启用或训练任何路线。',
        '[协议 v2 草案](protocol_v2_draft.md) · [机器摘要](summary.json) · [质量计数](hrv_qc_counts.csv) · [参与者覆盖](participant_hrv_coverage.csv)',
        '源文件与阈值／代码校验值均保存；新数据位于 data/processed/author_v2/，原始数据及 v1 实验保持不变。']
    (report/'alignment_qc_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(report/'alignment_qc_report.md')

if __name__=='__main__':run()
