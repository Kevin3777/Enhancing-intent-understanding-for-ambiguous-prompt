# 复现范围

目标为method_reimplemented的可审计工程实现，逐项验收；不是对论文绝对数值的默认复现声明。

## 可实现范围

冻结SD兼容双分支生成、采样forward注意力缓存/注入、Replace/Refine/Reweight，带来源和形状trace；可离线测试的多轮SESD/向量RAG；保存行为logprob与policy版本的一步TD on-policy PPO；同版本优先经验回放训练及旧版本经验诊断；实际CLIP双奖励；会话历史、CLI和局部消融。

## 明确未复现范围

- 原论文Efficient U-Net预训练、4×4090三天12000 episodes全量PPO、原始LLM微调和未公开checkpoint。
- 真实条件互信息及其生成器梯度；Eq22/24未说明的可学习mapper训练；Eq29/30直接attention参数梯度。
- 1673 JSON及原始95/5 split在数据审计确认前不可用；30位评审者和用户满意度调查不可替代。
- 在同数据/模型/标度/评估协议未齐备前，0.92 CLIP、4.3轮、4.73/5及论文表格数值均为论文报告，不是本地结果。
- 生成图重采样不是任意真实图像反演；same seed保留布局程度须实测，不能保证结构完美保持。

## 变体隔离

paper_literal: Eq13减号、Eq14 prev+0.2new、Eq21字面目标图、Eq28原式并严格校验。遇到负概率/非单位行和即显式失败，不能把非法结果传进网络后称成功。

safe: Eq14仍为论文权重；cross-map来源使用PDF保结构意图的base->edited实现假设，token映射后归一化；Eq28采用logit bias + softmax；PPO离散控制与固定mapper均标ENGINEERING_CHOICE/RESEARCH_VARIANT。mi_ascent和new_priority各自需显式选择。

## 状态词

- method_reimplemented: 只对有执行证据的机制分项使用；不能覆盖未实现公式。
- partial_experiment_replication: 有真实模型/样本/配置的局部实验，但不同于原论文实验规模。
- metric_verified: 指标实现/原始结果文件可核对，不代表复得论文数值。
- BLOCKED/NOT_REPRODUCED: 权重、数据、设备、网络或数学定义不足；绝不输出虚构成功。

阶段证据与遗留问题见experiments.md及milestones.json。

## 用户补充原则后的核心范围说明

当前依据仅main.pdf；旧README对照内容已经退出活动审计，仅保留历史备份。核心闭环的真实执行不依赖README。
“核心工程闭环可运行”与“所有未公开数学训练细节被唯一复原”是不同主张：前者须实际验收，后者不能凭工程推断宣称完成。
多轮流程默认确定性mock解析器，用于无供应商密钥的机制验证；可接provider并作schema校验，但不能声称等价于论文ChatGPT-4微调权重。
人类文字反馈通过SESD进入状态、提示和编辑；实际评分独立记录。论文没有给出评分如何加入Eq14的唯一公式，所以不额外创造混合奖励目标。
