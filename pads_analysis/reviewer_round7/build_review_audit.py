"""Twenty source-informed research questions, with completed experimental answers."""
from pathlib import Path
import json,re
from assemble_round7 import ci

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'integrated'

def main():
    result=json.loads((ROOT/'ROUND7_RESULTS.json').read_text())
    refs=json.loads((ROOT/'REFERENCES_WITH_LOCATIONS.json').read_text())
    reference={r['id']:r for r in refs}
    mapping={int(k):v for k,v in json.loads((ROOT/'REFERENCE_NUMBER_MAP.json').read_text()).items()}
    d={(r['method'],r['comparator'],r['metric']):r for r in result['contrasts']}
    a=d['adaptive','fixed_optimized','balanced_accuracy'];t=d['adaptive','fixed_optimized','nominal_seconds']
    q=next(r for r in result['joint_accuracy_time'] if r['method']=='adaptive' and r['comparator']=='fixed_optimized')
    pdg=result['participant_diagnostics']['adaptive'];u=result['unique_per_resample']
    audit=(ROOT.parent/'reviewer_round6/integrated/PADS_20_dimension_review_ZH.md').read_text()
    audit=audit.split('## 引用文献逐条审核',1)[0]
    audit=re.sub(r'^# .*', '# PADS主文第7轮：20维审稿自查与完整重拟合证据',audit,count=1)
    audit=audit.replace('## 本轮新增的真实补充结果','## 保留的第6轮参数敏感性结果')
    audit=audit.replace('使用本轮','使用该轮').replace('其余行用本轮共享的','其余行用第6轮共享的')
    # Preserve earlier numerical evidence; update the questions that this round answers.
    updates={
    1:{'已写入':'引言第一个临床背景段之后新增“Study objectives and research questions”。RQ1明确人群、个体化动作选择、优化固定比较、共同名义时间上限及全受试者二分类BA；RQ2分解选动作与停止，RQ3研究概率/拒判后果，RQ4检验完整按人重拟合。摘要第一句同步表明主要问题。'},
    3:{'已写入':f'主分析始终390人；本轮500次每次抽390行，但不同身份中位数仅{u["median"]:.0f}人，范围{u["minimum"]:.0f}–{u["maximum"]:.0f}。主图7、补图S18和补充方法S9明确区分行数、人数与拟合次数。'},
    4:{'已写入':'保留官方临床诊断来源，同时引用历史诊断准确性综述及较新的病理对照研究，说明疾病阶段、病例谱和诊断经验会影响标签可靠性。没有用其他研究的误诊率“校正”PADS标签。'},
    6:{'答':'全部抽样以身份为单位，保留该人的临床变量、双腕和所有任务。重复副本在外层、中层和状态交叉拟合中均同组。500次全流程审计检查40,000个训练/验证边界，未发现身份交叉。',
       '已写入':'主文新增按人抽样方法与真实完成情况；逐次保存受试者身份台账。原有未来特征扰动检查保留。不同副本仅保留抽样权重，不能成为彼此的验证样本。'},
    8:{'已写入':'保留六组单因素完整重训与全部结果；本轮在原配置下另加500次受试者重抽样，每次重新拟合。文献补充小样本、惩罚估计、PCA及算法比较依据，避免把“拟合很多模型”当成样本量足够。',
       '下一步':'若继续研究模型容量，应另行固定学习曲线或不同模型家族比较；当前不从本轮分布中挑选表现最好的设置替代主分析。'},
    10:{'已写入':'三个随机控制均保留。新的500次重拟合包含随机下一动作策略，并重新选择其自己的输出阈值；两类依赖测试人群任务池的置换控制不在本轮重拟合内。讨论明确“固定顺序停止省时”部分由其只取固定序列前缀保证，重点应看准确率与未解决结果的代价。'},
    11:{'答':'除原有312/78外层、208/104中层与内部状态交叉拟合外，500次抽样重新生成成组的五折外层与两级三折内层。标准化、PCA、所有状态分类器、策略、固定子集、二分类阈值和拒判边界每次重建。',
        '已写入':'复算7,500组确定性策略二分类阈值/拒判边界；随机策略2,500个阈值按保存的候选BA重新核对。随机加权拒判边界由固定代码重估，但未保存全部中层终端状态数组供独立复算，已注明这一复现范围。'},
    12:{'答':f'原主要差值−0.83点及条件区间[−5.97,+4.20]保留。完整重拟合的BA差中位数为{ci(a,100,True)}点，时间差{ci(t,1,True)}秒；括号是500次的中央95%经验稳定区间。',
        '已写入':'将固定预测条件区间、分区范围、随机采集模拟范围、完整重拟合经验区间、个体间分布范围分别定义。新分布不作为更正后的主要置信区间，不将方向频率当P值。',
        '仍未解决':'本轮覆盖固定家族与四个核心采集策略；校准器、类别分配及置换任务池仍只有其既有条件分析。经验分位区间尚无原估计量名义覆盖率保证，亦无多重检验校正。',
        '下一步':'主要内部缺口已补做。保留新结果的探索性地位；进一步主张需新问题与独立证据，而不能事后改写原主要假设。'},
    13:{'已写入':'保留Brier、log loss、可靠性图及残余校准斜率。补充类不平衡处理与校准的文献，但说明相关研究中的重采样校正不同于本模型的类别权重，不能直接断言为本结果的原因。本轮不声称已完成校准器的500次重拟合。'},
    14:{'已写入':'保留图5、表S3及全部成本/目标结果；新表S6报告四个核心策略在500次重拟合中的PD/DD明确正确、明确错误和拒判。图S17呈现开发阈值分布及测试拒判率，开发20%目标不被写成测试保证。'},
    16:{'问':'重新抽取受试者并重复全部开发步骤之后，结论还稳定吗？',
        '答':f'500/500次完整重拟合均完成，共10,360,000次逻辑回归拟合；原首折复现另4,144次。个体化对固定BA差为{ci(a,100,True)}点，时间差{ci(t,1,True)}秒。准确率更高{q["higher_ba_count"]}/500次、用时更短{q["lower_time_count"]}/500次，二者同时满足{q["joint_count"]}/500次。因此，未获得稳定个体化优势的判断仍成立，但原先“个体化比优化固定更耗时”的排序并不稳定。',
        '已写入':f'主图7、表5、补图S15–S18、表S5–S8。按个体计算后，二分类与原预测不一致比例的个体中位数为{pdg["binary_disagreement_with_original"]["median"]*100:.1f}%，实际三分类输出{pdg["output_disagreement_with_original"]["median"]*100:.1f}%，任务集合{pdg["task_disagreement_with_original"]["median"]*100:.1f}%。',
        '仍未解决':'保持同一队列和固定类别总数，没有新增病例；成组抽样降低独立训练身份数，近邻中的重复权重也保留。抽样、重新分区与训练变化不能由本实验单独拆解。个体分数仍非已校准的临床风险。',
        '下一步':'当前优先级转向解释和独立检验完整程序，而不是继续堆叠内部重复来宣称证实。现有数据仍可支持预先定义的学习曲线或任务失败压力研究，但这些不是本轮结果。'},
    18:{'已写入':'结合BMJ临床预测评价系列，补充外部人群、可用预测变量、任务兼容性、参考标准及精度要求。未计算缺乏目标人群参数的外部样本量，也未用健康对照数据代替PD/DD外部验证。'},
    20:{'答':'累计包保留历史结果，并新增完整500次重拟合、逐次预测、阈值开发分数、身份台账和复现代码。主文实际引用48条不同文献记录；每条有可支持论点、所在节、出版信息及读取范围。',
        '已写入':'主文和补充材料重新编号并统一引用。7幅主图与18幅补图均有PNG/PDF/SVG；新增图全部从审计后数值表生成。NMSQuest DOI已核实，主动采集评价论文改用正式JMLR出版页。',
        '仍未解决':'未把摘要核查冒充全文审计；预印本、数据集、更正、报告指南和原始研究分别识别。全部10,360,000个bootstrap模型对象未序列化，保留代码/种子、预测、台账和外层系数哈希以复现。作者、利益冲突、治理与AI辅助披露仍须作者提供。'}
    }
    add_sources={1:[38,39,50],3:[27,30],4:[38,39,50],6:[28,26],7:[45],8:[36,37,49],11:[25,26],12:[25,30,33,34,50],13:[31,35],15:[42,45,47],16:[25,27,34],18:[30,31,32],19:[45,47],20:[25,28,50]}
    def source_links(ids):
        return '、'.join('['+str(i)+']('+('https://doi.org/'+reference[i]['doi'] if reference[i].get('doi') else reference[i]['inspected_url'])+')' for i in ids)
    for num in range(1,21):
        pattern=r'(### Q'+f'{num:02d}'+r'[^\n]*\n\n)(.*?)(?=\n### Q|\n## |\Z)'
        match=re.search(pattern,audit,re.S);assert match,num
        body=match.group(2)
        for key,text in updates.get(num,{}).items():
            pat=r'\*\*'+key+r'：\*\* [^\n]*(?:\n(?!\n)[^\n]*)?'
            body,n=re.subn(pat,'**'+key+'：** '+text,body,count=1)
            assert n==1,(num,key)
        if num in add_sources:body+='\n\n**本轮补充依据：** '+source_links(add_sources[num])+'。\n'
        audit=audit[:match.start()]+match.group(1)+body+audit[match.end():]
    # Source-informed dimensions retain the original structure, with current experiment status.
    audit=audit.replace('实现核查通过；非外部验证','500次重复身份成组核查通过；非外部验证')
    audit=audit.replace('完整流程已记录','500次完整流程与阈值重选已完成')
    audit=audit.replace('分母与区间已澄清','原条件区间保留；新增完整重拟合经验区间')
    audit=audit.replace('五个完整新分区已整合','五个分区及500次受试者重抽样已整合')
    audit=audit.replace('复现材料已补足；作者声明待确认','48条引用与25幅图已核查；作者声明待补')
    for num,ids in add_sources.items():
        pat=r'(\| '+f'{num:02d}'+r' [^|]*\| )([^|]*)(\| [^\n]*\|)'
        audit,n=re.subn(pat,lambda m:m.group(1)+m.group(2).strip()+'；'+source_links(ids)+' '+m.group(3),audit,count=1)
        assert n==1,num
    audit=audit.replace('FIGURE_MANIFEST.json给出20幅图的来源及完整图注。完整代码与已拟合模型保存在累计包中。',
      'FIGURE_MANIFEST.json连接新增5幅图的数值来源与保留的20幅既有图。完整图注在主文/补充材料中；新重拟合代码、预测、阈值与台账可追溯。')
    audit=audit.replace('https://www.movementdisorders.org/MDS/MDS-Rating-Scales/Non-Motor-Symptoms-Questionnaire.htm','https://doi.org/10.1002/mds.20844')
    audit=audit.replace('https://arxiv.org/abs/2312.01530','https://www.jmlr.org/papers/v26/23-1635.html')
    audit=re.sub(r'(\| DECIDE-AI[^\n]*\n)',r'\1| STARD-AI，Nature Medicine 2025 | 正式摘要、出版记录及2026署名更正全文；未获得完整清单正文 | 说明AI候选检测、评价数据、偏倚和适用性 [50](https://doi.org/10.1038/s41591-025-03953-8) | 不能据摘要核查宣称逐项通过正式清单 |\n',audit,count=1)
    current=f'''## 本轮新增的完整重拟合结果

| 比较（前者减后者） | BA差，百分点 | 名义用时差，秒 | BA更高且用时更短 |
| --- | --- | --- | --- |
'''
    names={'adaptive':'个体化','fixed_optimized':'优化固定','fixed_sequence_stop':'固定顺序停止','random_next_stop':'随机下一动作'}
    for j in result['joint_accuracy_time']:
        l,r=j['method'],j['comparator']
        current+=f'| {names[l]} − {names[r]} | {ci(d[l,r,"balanced_accuracy"],100,True)} | {ci(d[l,r,"nominal_seconds"],1,True)} | {j["joint_count"]}/500（{j["joint_fraction"]*100:.1f}%） |\n'
    current+='\n括号为500次完整重拟合分布的中央95%经验区间，保留原主要点估计及其条件置信区间。联合频率不是优效概率或P值。500次固定于拟合前，全部结果进入汇总；没有基于结果提前停止。阈值、近邻和固定动作选择均重新开发，采用同组受试者与同组外层划分进行配对比较。\n\n'
    audit=audit.replace('## 保留的第6轮参数敏感性结果',current+'## 保留的第6轮参数敏感性结果')
    audit=audit.replace('| 是否适用于每个亚组/疾病',f'| 完整抽样与阈值重选后是否稳定 | 500次全流程；BA差{ci(a,100,True)}点；个体预测变化 | 图7、表5、图S15–S18、表S5–S8 | 描述训练与阈值不确定性；没有新病例或外部确认 |\n| 是否适用于每个亚组/疾病')
    audit=re.sub(r'\[(\d+)\](?=\()',lambda m:'['+str(mapping[int(m.group(1))])+']',audit)
    intro=f'''本轮完成用户优先指定的“按受试者重抽样、每次完整重新拟合”：500/500次，四个核心采集策略配对评估，重新学习标准化、PCA、状态分类器、策略和输出阈值。原主要分析及其区间未改动。

本文件以20个文献支持的审稿问题组织修改，属于研究自查与证据追踪，不冒充真实期刊审稿意见或独立PROBAST双人评估。引言已前置主要研究问题，全文按临床问题、比较设计、结果和解释重写；写作流畅度不替代实验或准确的AI辅助披露。

'''
    first=audit.index('\n\n')
    # Replace the old round-6 opening paragraphs with the current scope.
    audit=audit[:first+2]+intro+audit[audit.index('## 先从已发表论文提炼评价依据'):]
    audit+='''\n## 引用文献逐条审核

本轮主文实际引用48条不同文献，按首次出现重新编号，并与补充材料和本审查表统一。引用覆盖临床诊断、数字测量、固定/主动采集、嵌套评估、完整重拟合、校准、拒判和临床评价。目录包含原始研究、方法学论文、综述、报告规范、作者更正、数据/软件归属及明确标注的预印本，不能统称为48项独立临床验证。

每条记录注明读取范围；摘要、书目、局部全文和完整官方文件彼此区分。没有获得必要内容的候选文献未用于新增实质论据。NMSQuest原始论文DOI已核实为10.1002/mds.20844；主动采集评估文献使用2025年JMLR正式记录。全文不借用异质人群研究的准确率来抬高本研究的证据级别。

| 主文编号 | 文献 | 核查范围 | 支持范围与限制 | 主文位置 |
| --- | --- | --- | --- | --- |
'''
    for r in refs:
        url='https://doi.org/'+r['doi'] if r.get('doi') else r['inspected_url']
        cols=[str(r['manuscript_number']),f'[{r["title"]}]({url}) {r["publication"]}',r['inspection_level'],r['support_boundary'],'; '.join(r['main_locations'])]
        if r.get('correction_url'):cols[1]+=' [更正记录]('+r['correction_url']+')'
        audit+='| '+' | '.join(s.replace('|','/').replace('\n',' ') for s in cols)+' |\n'
    audit+='''

## 本轮问答落实到文章的方式

“到底要回答什么？”——摘要Objective和引言前部给出RQ1，明确PD/DD人群、个体化选择、优化固定对照、共同采集上限和全受试者BA。RQ2–RQ4分别对应任务选择/停止、校准/拒判和完整重拟合，避免把事后探索写成原主要假设。

“把训练和阈值不确定性加进来了吗？”——500次均重新拟合四个核心采集策略，保存每次预测和阈值。校准、类别分配和置换任务池未被含糊地归入该结论。

“更宽的证据链是否支持更强的临床结论？”——方法、结果和讨论逐层说明何者已被检验、何者仍依赖假设。对未证实优势、等效、泛化与临床效用使用不同表述，避免由阴性比较跳到所有个体化方法无效。

“语言改动是否只换同义词？”——引言按临床问题和研究缺口组织；方法按数据、采集规则、模型、嵌套评价和不确定性组织；结果将主要比较与后续解释性实验衔接；讨论解释不同控制和不确定性来源。将软件运行细节移至补充方法，将待作者确认的事实独立保存，没有虚构资助、伦理豁免或利益冲突声明。
'''
    (OUT/'PADS_20_dimension_review_ZH.md').write_text(audit)
    assert len(re.findall(r'^### Q\d\d',audit,re.M))==20
    print('20 review questions updated; 48 citation records mapped to the manuscript.')

if __name__=='__main__':main()
