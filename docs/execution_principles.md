# 当前执行原则（用户补充要求优先）

生效日期：2026-10-27；本文件记录本次会话中用户明确的范围调整。

1. 唯一方法学依据是 `paper/main.pdf`（Neurocomputing 2025, VCA）；原始输入README只留档，不作为公式、算法、参数或实现的依据。后续不继续参考或对照它。
2. 交付重点是完整核心数据流：多轮SESD/检索消歧 -> 编辑动作 -> 三种Cross-Attention -> frozen diffusion -> Eq14双奖励/用户反馈 -> Eq18 TD/Eq19回放/Eq20 PPO -> 下一轮 -> 最终选择。模块必须实际连接。
3. PDF明确内容标 **paper-faithful**；无法唯一确定的实现细节标 **implementation-assumption**，附页码、公式和行为测试。必要可执行变体保留，但不包装为原论文。
4. 不增加新算法或新优化目标。Diffusers/PyTorch等只作基础设施；外部注意力代码只可用作底层注入工程参考。默认无熵奖励；只监测entropy。
5. 保留所有已通过验收的文件、权重和结果。修改本次自建代码前核对哈希并留备份；不重新初始化或下载已有权重。
6. 必须交付固定环境、推理/训练入口、核心单元/端到端测试、公式与算法对应、真实运行证据及未定细节。

## 与先前分类的映射

PAPER_EXACT表达 -> paper-faithful；PAPER_AMBIGUOUS保留的原文表达仍为paper-faithful转写，其工程落地单列implementation-assumption；ENGINEERING_CHOICE和必要RESEARCH_VARIANT -> implementation-assumption；NOT_REPRODUCED保留为明确未复现项。旧标签留作审计元数据，不能据此夸大完成范围。

## 当前联合训练边界

按PDF p7 Eq19，每个PPO epoch按 `q_i=exp(-lambda*age_i)*abs(TD_i)+eps` 归一化采样，loss乘 `(N*p_i)^(-beta)`。persistent pool保存完整轨迹和旧行为logprob；训练只允许同一行为策略版本的rollout，4个epochs内可重采样；旧版本数据只做TD/年龄诊断。这是对PDF未定义的off-policy问题的implementation-assumption，不宣称已复现任意旧样本的校正策略。

PDF p8的2.2/0.3保留数值；映射为value MSE/行为策略KL系数是implementation-assumption。Eq18一步TD为默认，无GAE、无额外熵目标。Eq13真实MI及Eq24未公开mapper梯度路径仍无法唯一确定；保留原式和边界，不伪造梯度。当前策略真正训练算子分布及注入/混合/步数控制参数；固定token对齐不冒充原mapper参数训练。
