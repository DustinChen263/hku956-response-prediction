from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; O=R/'reports/dataset_walkthrough_20261111'
pid='hku1919';song=370177;order=16
av=pd.read_csv(R/'data/raw/HKU956/HKU956/3. AV_ratings.csv')
r=av[(av.participant_id==pid)&(av.song_id==song)&(av.song_no==order)].iloc[0]
v=np.loadtxt(R/f'data/raw/HKU956/HKU956/1. physiological_signals/{pid}/HR/{order}_{song}.csv')
pd.DataFrame({'sample_number_1based':np.arange(1,len(v)+1),'nominal_elapsed_s':np.arange(len(v)),'hr_bpm':v}).to_csv(O/'example_hr_all_72_samples.csv',index=False)
summary={'record_id':f'{pid}_{order}_{song}','rating_row':r.to_dict(),'HR_samples':len(v),'first_hr_bpm':v[0],'last_hr_bpm':v[-1],'min_hr_bpm':v.min(),'max_hr_bpm':v.max(),'whole_window_mean_hr_bpm':v.mean(),'first30_mean_hr_bpm':v[:30].mean(),'last30_mean_hr_bpm':v[-30:].mean(),'last_minus_first_bpm':v[-1]-v[0],'audio_path':str(R/'data/raw/audio/jamendo_candidates/370177.mp3'),'audio_duration_s':276.8955555555556,'audio_position_verified':False,'time_axis':'nominal 1 Hz sample offsets, not independently timestamped HR samples','illustrative_example_not_representative':True}
(O/'example_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
f,ax=plt.subplots(figsize=(12,6));f.subplots_adjust(top=.78,bottom=.22,left=.1,right=.95)
ax.plot(np.arange(len(v)),v,color='#168C91',lw=2.5,label='Published HR readings (1 Hz)')
ax.axvspan(0,30,color='#3974B9',alpha=.08,label='First 30 s used for our model label')
ax.hlines(v[:30].mean(),0,30,color='#3974B9',ls='--',lw=2,label=f'First-30-s mean: {v[:30].mean():.2f} bpm')
ax.scatter([0,71],[v[0],v[-1]],color='#17324D');ax.annotate(f'{v[0]:.2f}',(0,v[0]),xytext=(7,5),textcoords='offset points');ax.annotate(f'{v[-1]:.2f}',(71,v[-1]),xytext=(-48,8),textcoords='offset points')
ax.set(xlabel='Elapsed listening time (s, nominal 1 Hz sample offsets)',ylabel='Heart rate (bpm)',xlim=(0,72),ylim=(89.5,96));ax.legend(loc='upper right',fontsize=10);ax.grid(alpha=.15)
f.text(.1,.93,'A real listening record: hku1919, song 370177',fontsize=22,weight='bold',color='#17324D')
f.text(.1,.85,'72 s listening window     Post-listening valence: +9.3     Arousal: +1.0',fontsize=14)
f.text(.1,.06,'A/V are one rating pair after listening, not second-by-second signals.\nThe HR change occurs during listening. No pre-song baseline or verified audio-position alignment is available.',fontsize=11,color='#526373')
f.savefig(O/'example_hr_timeline.png',dpi=200);f.savefig(O/'example_hr_timeline.svg');plt.close(f)
notes='''# HKU956 入门：一条真实记录

来源：原始 AV_ratings.csv、hku1919/HR/16_370177.csv、作者 raw_log 及 record_alignment.csv。
论文：https://www.mdpi.com/2076-3417/12/18/9354，重点 §3.1–3.2。

原始标签行：hku1919,16,370177,9.3,positive,1,positive,72。

participant_id 是匿名听众编号；song_no 为从 0 开始的听歌顺序编号，16 是编号，不是秒数；song_id 为歌曲标识。
valence_rating=9.3 是听完后的愉悦程度评分，arousal_rating=1 是听完后的激活/能量程度评分，范围均为 -10 到 +10。
valence/arousal 两个字符串列由数值符号派生：大于0 positive，小于0 negative，等于0 neutral。arousal=negative 不是不开心。
play_duration=72 是本次听歌墙钟窗口秒数，不是原始歌曲总长度。对应音频约276.90秒。

HR 文件只有一列，无表头，共72个数值。单位bpm，发布采样率1Hz。一个读数94.85表示当时设备估计心率94.85次/分钟，不是这一秒跳94.85次。数值有小数是估计值的表示，不等于对应精度已验证。
图上0–71秒由采样顺序推定；原始HR没有逐样本绝对时间戳，不声称精确到秒的音频同步。

原始前12个HR值：'''+', '.join(f'{x:.2f}' for x in v[:12])+f'''.

第一条{v[0]:.2f}，末条{v[-1]:.2f}，末减首{v[-1]-v[0]:.2f} bpm。
最低{v.min():.2f}，最高{v.max():.2f}，全窗口均值{v.mean():.5f}，前30秒均值{v[:30].mean():.5f} bpm。

本次模型目标：[mean_HR={v[:30].mean():.5f}, arousal=1.0, valence=9.3]。
这是一条解释数据结构的例子，不是总体效应或模型预测结果。

原始日志：2019-08-22 16:39:12 song-start，16:40:24 survey-popup（原日志香港本地时间），相隔72秒。
结束时记录媒体位置71.578358秒，起始位置为空。零起点假设与记录相容，但尚未完整验证。
窗口内还包含浏览推荐、搜索等交互事件，因此不能把变化仅归因于音乐。
提供完整歌曲试听，不裁切为所谓精确听歌片段。A/V没有连续时间轨迹，不能画成随每秒音乐变化的曲线。

同一听众曾在song_no=3再次对应同歌曲ID，记录124秒、V=9.2、A=-1.0。评分属于某一次听歌记录，不是歌曲永远固定的情绪标签。

## 对此前基线表述的澄清

论文§3.1明确提到，为稳定信号和采集基线，在实验前问卷开始前两分钟佩戴手环。不能说实验没有考虑/采集基线。
目前本地发布分段文件中没有找到可用于建模的听歌前基线，仍需作者确认是否保留并可分享这些连续原始数据。

原论文主要使用音频、生理特征或两者结合分类A/V正负；我们的实验使用音频预测连续HR/A/V，任务不同。
'''
(O/'walkthrough_notes_zh.md').write_text(notes)
print(json.dumps(summary,ensure_ascii=False,indent=2))
