from pathlib import Path
import csv,json

R=Path(__file__).resolve().parent
S=json.loads((R/'STABILITY_RESULTS.json').read_text())
def read(name):
    with (R/'tables'/name).open() as f:return list(csv.DictReader(f))
metrics=read('repeat_metrics.csv');contrasts=read('repeat_contrasts.csv')
ix={(int(r['seed']),r['method']):r for r in metrics}
cix={(int(r['seed']),r['comparison']):r for r in contrasts}
core=['adaptive','fixed_optimized','fixed_sequence_stop']
lab=['个体化选择','优化固定','固定序列加停止']
delta=S['contrasts']['adaptive_minus_fixed'];stop=S['contrasts']['fixed_sequence_stop_minus_fixed']
new=[r for r in contrasts if r['role']=='new_repeat' and r['comparison']=='adaptive_minus_fixed']
positive=sum(float(r['delta_ba_pp'])>0 for r in new)
table='| 划分 | 自适应 BA | 固定 BA | BA 差值（百分点） | 平均时间差（秒） | 自适应 / 固定覆盖率 |\n|---|---|---|---|---|---|\n'
for i,seed in enumerate(range(20260913,20260919)):
    a,f=ix[seed,'adaptive'],ix[seed,'fixed_optimized'];d=cix[seed,'adaptive_minus_fixed']
    name='原主要分析' if i==0 else f'新增 {i}：{seed}'
    table+=f"| {name} | {100*float(a['ba']):.2f}% | {100*float(f['ba']):.2f}% | {float(d['delta_ba_pp']):+.2f} | {float(d['delta_seconds']):+.2f} | {100*float(a['coverage']):.1f}% / {100*float(f['coverage']):.1f}% |\n"
stable='| 方法 | 二分类预测发生改变 | 三分类输出发生改变 | 动作集合发生改变 |\n|---|---|---|---|\n'
for m,l in zip(core,lab):
    v=S['participant_stability'][m]
    stable+='| '+l+' | '+' | '.join(f"{v[k]}/390（{100*v[k]/390:.1f}%）" for k in ['n_distinct_binary','n_distinct_actual_outputs','n_distinct_task_sets'])+' |\n'
report=f'''# PADS 审稿补强：第二轮结果

日期：2026-09-13。已落实你的选择：**仅使用公开数据，不需要新增临床采集或寻找合作方**。

## 已完成的补充实验

已在 5 组新随机划分下完成整个嵌套开发过程，共 25 个新外层测试折、103,600 次逻辑回归拟合。每个外层折均重新拟合标准化、任务 PCA、状态分类器、固定/自适应采集策略，以及用于选择最终二分类与拒判阈值的中层完整过程。没有沿用原模型对新划分简单打分。

原主要分析的代码、折文件与计划均保留原哈希。本轮是看到原主要结果后开展的探索性稳定性分析，不能追溯性变成新的预设主要研究。分析仍只有 **390 名独立参与者**，5 次重复不会变成 1,950 名独立患者，33,150 行全部情景预测也不是同等数量的临床样本。

## 完整重复的结果

BA 指全体参与者的基础二分类平衡准确率，包括最终被标记为“需进一步评估”的人；它不等同于三分类输出的正确率。覆盖率则是实际获得明确判断的人数比例。时间差为自适应减固定，两者受同一 50% 名义时间上限约束，实际平均耗时并未被强制匹配。

{table}

仅在 5 组新增划分中，自适应减固定 BA 的平均差为 **{delta['delta_ba_pp']['mean']:+.2f} 个百分点**，范围 **{delta['delta_ba_pp']['min']:+.2f} 至 {delta['delta_ba_pp']['max']:+.2f}**；其中 {positive}/5 组点估计为正。平均时间差为 **{delta['delta_seconds']['mean']:+.2f} 秒**，范围 **{delta['delta_seconds']['min']:+.2f} 至 {delta['delta_seconds']['max']:+.2f} 秒**。原主要差值仍为 −0.83 个百分点，原条件 95% 区间仍为 −5.97 至 +4.20。

这些重复使用同一批人，训练集也有重叠。因此这里只报告差值、平均数、中位数和范围，没有用 25 折做简单 t 检验，也没有把范围标成 95% 置信区间。原主要分析与新增重复的差值方向并不一致，目前不能确认自适应选择存在稳定优势。重复结果展示的是本队列内对数据划分和开发过程的敏感性，不能证明外部泛化、临床等效或非劣效。

在次要比较中，固定序列加停止相对优化固定方案的平均 BA 差为 {stop['delta_ba_pp']['mean']:+.2f} 个百分点（范围 {stop['delta_ba_pp']['min']:+.2f} 至 {stop['delta_ba_pp']['max']:+.2f}），平均名义采集时间差为 {stop['delta_seconds']['mean']:+.2f} 秒（范围 {stop['delta_seconds']['min']:+.2f} 至 {stop['delta_seconds']['max']:+.2f}）。这仍是次要发现，不能替代原来的主要假设。

## 同一个人的判断是否随开发数据改变

下表计算同一人在 5 组新增划分中是否出现过至少两种不同结果，没有把原主要分析混入这些比例。

{stable}

二分类判断和实际三分类输出的变化提示，单次划分的预测不宜被当作稳定的个体临床建议。动作集合变化还包含开发数据导致固定协议选择改变的影响；它本身不等于分类失败。这里没有测量患者病情随时间的变化。

## 公开数据兼容性

检查了 PADS 来源本身及四个外部候选。依据所读来源资料，本轮没有建立能够直接验证完整策略的独立兼容队列：

- [Gait in Parkinson’s Disease](https://physionet.org/content/gaitpdb/1.0.0/) 使用足底力与 PD/健康对照，缺少所需腕部动作和 DD 比较。
- [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) 的结局是冻结步态事件，不能替代 PD/DD 参考诊断。
- [REMAP](https://www.nature.com/articles/s41597-023-02663-5) 区分公开粗化骨架数据与受控的额外数据，研究 PD/健康对照和活动表现，不能直接重建本策略。
- [mPower 官方资料](https://www.synapse.org/Synapse:syn4993293/wiki/247859) 列出的手机任务与本研究不同，所需 DD 标签和完整输入未获得确认。

这些是基于已核查候选的兼容性判断，不是“世界上不存在其他数据集”的断言。没有因外部性能不理想而排除候选，因为本轮没有拟合这些外部数据。详细来源与可用性边界见 PUBLIC_DATA_FEASIBILITY.md。

## 新增科研图

| 图 | 内容 | 解释边界 |
|---|---|---|
| R6 | 原主要分析及 5 个完整新重复的 BA、时间、覆盖率 | 每个点代表整个重复，原主要分析单独标记 |
| R7 | 两个配对比较的准确性和时间差 | 展示方向变化，不把重复当独立队列 |
| R8 | 同一人的输出变化及各重复动作频率 | 区分判断变化和协议变化，不当作病情进展 |

每组图包含 350 dpi PNG、PDF 和 SVG。新增 8 份 CSV 保留所有重复预测、逐折指标、重复指标、配对差值、个体变化、动作频率、既定人口亚组及 DD 亚型。图 R6–R8 与第一轮的 10 组图共同构成当前 13 组科研图。

## 对 20 个审稿维度的推进

| 维度 | 本轮新增证据 | 仍需承认的限制 |
|---|---|---|
| 6 受试者隔离 | 25 个外层折的中层和状态层记录均核对 | 不证明源诊断评估的盲法 |
| 11 嵌套开发 | 每次重新选择协议与阈值 | 没有评价其他模型家族 |
| 12 配对与分母 | 每重复保留同一批 390 人 | 重复之间有依赖 |
| 14 拒判覆盖 | 每次由中层设阈值、外层实测覆盖 | 20% 目标不能保证外层精确达标 |
| 16 开发稳定性 | 5 组新划分的全流程拟合完成 | 不是总体抽样的全流程 bootstrap |
| 18 人群异质性 | 保存各重复的既定亚组和 DD 亚型 | 小亚组并未增加真实人数 |
| 19 外部适用性 | 明确公开候选不兼容的具体原因 | 没有完成外部验证或前瞻实验 |
| 20 可复现 | 模型、阈值、路径、划分和代码已保存 | 新分析保留探索性标记 |

## 下一轮建议

优先检验校准与拒判覆盖稳定性：主要目标保持你选择的 20% 拒判，并在开发层设定其他目标作敏感性分析；外层只评价实际覆盖、两类错误和三分类损失。其他有价值的路线是相同预算下的随机动作对照，或在开发层重新选择实际平均时间匹配的协议。下一轮通过单题问答确定优先顺序。
'''
(R/'PADS_round2_results_ZH.md').write_text(report)
engtable=table
for x,y in [('划分','Partition'),('自适应 / 固定覆盖率','Adaptive / fixed coverage'),('自适应 BA','Adaptive BA'),('固定 BA','Fixed BA'),('BA 差值（百分点）','BA difference (pp)'),('平均时间差（秒）','Mean time difference (s)'),('原主要分析','Original primary'),('新增','New repeat')]:engtable=engtable.replace(x,y)
text=f'''# Manuscript additions: repeated development

## Methods

Following the primary analysis, we evaluated development stability using five additional stratified five-fold outer partitions, with base seeds 20260914–20260918. All preprocessing, task-specific representations, state classifiers, acquisition-policy development and middle-layer threshold setting were refitted within every outer development set. The original input configuration, model family, hyperparameters, time constraints, task dependencies and 20% development rejection target were retained. Each new base seed governed outer partitioning and the original inner-seed offsets. The original primary partition remained unchanged.

For each repeat, its five outer folds were combined into one held-out prediction per participant per method. Binary balanced accuracy included all participants, including those subsequently assigned an indeterminate three-way output; clear-output coverage was reported separately. The two methods shared the same nominal motion-time cap, rather than matched realized mean times. We reported per-repeat metrics and paired contrasts, with descriptive mean, median and range across the five new repeats. Repeated predictions were not treated as additional independent participants. Because evaluation sets and development samples overlap across repeats, we did not apply a naive fold-level significance test or interpret the range as a confidence interval. These analyses assess development sensitivity within the source cohort rather than external validation or full population-sampling uncertainty.

We also counted participants with changes in underlying binary predictions, actual three-way outputs or acquired task sets across the five new repeats. Existing age, recorded-gender and DD-subtype definitions were retained for supplementary repeat-level summaries.

## Results

All 25 additional outer-fold evaluations completed, involving 103,600 logistic classifier fits and 33,150 held-out prediction rows across the 17 original scenarios. Each repeat included the same 390 participants (276 PD and 114 DD). Split ledgers confirmed participant separation at all three learning/evaluation levels, and all abbreviated paths satisfied the nominal acquisition constraints.

{engtable}

Across the five new repeats, the mean adaptive-minus-fixed BA difference was {delta['delta_ba_pp']['mean']:+.2f} percentage points (range {delta['delta_ba_pp']['min']:+.2f} to {delta['delta_ba_pp']['max']:+.2f}); {positive} of five point estimates favored adaptive selection. The mean difference in nominal acquisition time was {delta['delta_seconds']['mean']:+.2f} seconds (range {delta['delta_seconds']['min']:+.2f} to {delta['delta_seconds']['max']:+.2f}). The original primary contrast remained −0.83 percentage points, with its original conditional interval of −5.97 to 4.20.

Fixed-sequence stopping versus optimized fixed assessment had a mean BA difference of {stop['delta_ba_pp']['mean']:+.2f} points (range {stop['delta_ba_pp']['min']:+.2f} to {stop['delta_ba_pp']['max']:+.2f}) and a mean time difference of {stop['delta_seconds']['mean']:+.2f} seconds across the new repeats.

'''
for m,l in zip(core,['adaptive selection','optimized fixed assessment','fixed-sequence stopping']):
    s=S['participant_stability'][m]
    text+=f"For {l}, binary predictions changed across repeats in {s['n_distinct_binary']}/390 participants, actual three-way outputs changed in {s['n_distinct_actual_outputs']}/390, and task sets changed in {s['n_distinct_task_sets']}/390. "
text+='''

## Discussion

Repeated development broadens the evidence beyond a single partition while retaining the original primary result. The differing signs of the original and repeated contrasts do not establish a stable advantage for individualized task selection. The analysis also shows why the same participant can receive different outputs after the development sample changes. Such differences matter to the stability of an individualized policy, even when aggregate accuracy changes less markedly. They do not establish clinical unreliability in a new service, which requires independent evaluation, nor do they establish fairness or a disease-specific interaction in small subgroups.

A targeted review of public candidate datasets did not establish a directly compatible independent evaluation cohort. The [PhysioNet gait dataset](https://physionet.org/content/gaitpdb/1.0.0/) uses foot-force measurements and PD/healthy controls; [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) labels freezing-of-gait events; [REMAP](https://www.nature.com/articles/s41597-023-02663-5) addresses mobility activities with PD/healthy controls and separates open coarsened pose data from controlled additional signals; and the tasks in the [mPower portal](https://www.synapse.org/Synapse:syn4993293/wiki/247859) differ from the PADS protocol. These differences preclude claiming external evaluation of the complete frozen PD-versus-DD policy using those data as currently documented. They do not exclude future compatible data or separately specified component-level transfer studies.

This work therefore remains a retrospective internal evaluation. Repeated partitioning cannot substitute for independent diagnostic labels, new-site evaluation or direct measurement of a shortened clinical workflow. Calibration, transfer of coverage targets and separation of task selection from stopping remain appropriate next questions.

## Figure captions

Figure R6. Performance across the original primary partition and five new complete nested-development repeats. Panels show all-participant binary balanced accuracy, mean nominal motion time and actual clear-output coverage. Each point summarizes 390 outer-held-out participants in a complete repeat. The original primary row is shaded; no independence across repeats is assumed.

Figure R7. Paired accuracy and time contrasts. Adaptive selection and fixed-sequence stopping are each compared with optimized fixed assessment within the same repeat. P denotes the original primary analysis; 1–5 denote seeds 20260914–20260918. The plot displays observed differences without treating repeated partitions as independent cohorts.

Figure R8. Development sensitivity of outputs and tasks. A, participants with any change in binary prediction, three-way output or task set across five new repeats; all denominators are 390. B, task-acquisition frequency for adaptive and optimized fixed assessment in each repeat. Changes in fitted procedures do not represent longitudinal changes in patients.

Public-data source documentation: [PhysioNet gait](https://physionet.org/content/gaitpdb/1.0.0/), [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait), [REMAP original paper](https://www.nature.com/articles/s41597-023-02663-5), [mPower official portal](https://www.synapse.org/Synapse:syn4993293/wiki/247859).
'''
(R/'MANUSCRIPT_ADDITIONS_ROUND2.md').write_text(text)
print('Wrote second-round results and manuscript additions.')
