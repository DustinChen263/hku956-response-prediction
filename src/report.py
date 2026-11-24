"""Generate transparent tables and static research figures from actual output files."""
import json
import os
import pandas as pd
import numpy as np
from .common import parser, load_config, folder


def report(cfg):
    out, dest = folder(cfg, 'results'), folder(cfg, 'reports')
    os.environ.setdefault('MPLCONFIGDIR', str(dest / '.matplotlib'))
    is_synthetic = bool(cfg.get('synthetic'))
    label = 'SYNTHETIC SMOKE TEST — NOT RESEARCH RESULTS' if is_synthetic else 'HKU956'
    metrics_file = out / 'metrics_test.csv'
    lines = ['# 实验报告', '', f'**{label}**', '']
    if metrics_file.exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        metrics = pd.read_csv(metrics_file)
        metrics.to_csv(out / 'metrics.csv', index=False)
        summary = metrics.groupby(['model','target']).macro_mae.agg(['mean','std','count']).reset_index()
        summary['synthetic'] = is_synthetic
        summary.to_csv(out / 'seed_summary.csv', index=False)
        lines += ['## 测试集参与者宏平均 MAE', '', '|模型|目标|跨种子均值|跨种子标准差|种子数|', '|---|---|---:|---:|---:|']
        for r in summary.itertuples():
            sd = f'{r.std:.4f}' if np.isfinite(r.std) else '—'
            lines.append(f'|{r.model}|{r.target}|{r.mean:.4f}|{sd}|{r.count}|')
        targets = list(metrics.target.unique())
        fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), squeeze=False)
        for ax, target in zip(axes[0], targets):
            sub = summary[summary.target == target]
            ax.bar(sub.model, sub['mean'])
            ax.tick_params(axis='x', labelrotation=65)
            ax.set(title=target, ylabel='Participant macro MAE')
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(dest / 'model_comparison.png', dpi=160)
        plt.close(fig)
        people = pd.read_csv(out / 'participant_metrics_test.csv')
        fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), squeeze=False)
        for ax, target in zip(axes[0], targets):
            sub = people[people.target == target].groupby(['participant_id','model']).mae.mean().unstack()
            sub.plot.bar(ax=ax, legend=False)
            ax.set(title=target, ylabel='MAE')
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(dest / 'participant_errors.png', dpi=160)
        plt.close(fig)
        predictions = pd.read_csv(out / 'predictions_test.csv')
        predictions['absolute_error'] = abs(predictions.actual - predictions.predicted)
        labels = pd.read_csv(folder(cfg,'processed') / 'labels.csv')
        diagnostics = predictions.merge(labels.drop(columns=['arousal','valence','synthetic'], errors='ignore'), on='record_id', how='left')
        diagnostics.to_csv(out / 'residual_diagnostics.csv', index=False)
        lines += ['', '跨种子标准差仅反映训练随机性，不能解释为受试者不确定性。',
                  '配对参与者 bootstrap 见 paired_bootstrap_test.csv；少量测试参与者下仅作描述性参考。',
                  '残差、时长和 QC 标志见 residual_diagnostics.csv。']
        cv = out / 'cv_metrics.csv'
        if cv.exists():
            cross = pd.read_csv(cv)
            cross.groupby(['model','target']).macro_mae.agg(['mean','std','count']).to_csv(out / 'cv_summary.csv')
            lines += ['', '参与者外层交叉验证已完成，见 cv_metrics.csv 和 cv_summary.csv；最终测试参与者未进入 CV。']
        else:
            lines += ['', '参与者交叉验证尚未运行，不能仅依赖一次四人测试作稳定性结论。']
    else:
        lines += ['尚无真实音频实验结果。请查看 data_audit.md 和 feature_failures.csv。',
                  '模型代码已实现不等于数据可行性或研究假设已成立。']
    lines += ['', '## 解释边界', '', '音频输入只能学习人群层面的关联模式；不是个性化响应模型。',
              '播放期间统计量不是前后变化量；离线排序不构成推荐干预效果证据。',
              '当前 QC 默认参数尚须短窗测量文献与人工审计支持后锁定。',
              '单任务与多任务主比较使用相同记录；参数量与训练次数见 registry.json。',
              '完整文献综述、真实音频起点核验、窗口敏感性需要单独完成；参数预算匹配结果以是否运行 matched_multi_mlp 为准。']
    if is_synthetic:
        lines += ['', '**以上数值全部来自人为构造样本，仅验证软件，不能写作 HKU956 论文结果。**']
    (dest / 'final_report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    # Label distribution is useful even while audio is blocked.
    labels_file = folder(cfg,'processed') / 'labels.csv'
    if labels_file.exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = pd.read_csv(labels_file)
        targets = cfg['training']['targets']
        fig, axes = plt.subplots(1,len(targets),figsize=(5*len(targets),3),squeeze=False)
        for ax,t in zip(axes[0],targets):
            ax.hist(labels[t].dropna(),bins=25)
            ax.set(title=t, ylabel='Listening records')
        fig.suptitle(label + ' — labels')
        fig.tight_layout()
        fig.savefig(dest/'label_distributions.png',dpi=160)
        plt.close(fig)
    print('Report:', dest / 'final_report.md')


def main():
    args = parser(__doc__).parse_args()
    report(load_config(args.config))

if __name__ == '__main__':
    main()
