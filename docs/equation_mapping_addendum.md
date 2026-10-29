# 公式与算法映射：深化实现增补

本文件更新 `paper_to_code_matrix.md` 中本轮之前的状态；旧映射与审计报告作为历史快照保留。唯一方法依据是 `paper/main.pdf`（PDF 物理页码，SHA-256 `295d63314886f7b5c661a16289564001ea4754cf315451563dc6efd12c9a935c`）。忠实转写与全部 33 条公式的逐项分析仍见 `equation_audit.md`。

| PDF位置 | 现在的实现和检验 | 性质、限制 |
|---|---|---|
| p4–6 Eq1–2、Algorithm2 | `sesd/dialogue.py::Dialogue.refine`、`sesd/llm.py::LocalLlamaCppLLM` 与 `LocalTransformersLLM`；`test_sesd.py` | 多轮历史与真实冻结 LLM 相连。模型、消息模板、历史截取、是否需要追问的守卫和字符串 null 规范化是 implementation-assumption；记录 LLM 原始输出、依据和守卫来源，不等同论文 GPT-4 微调。 |
| p5–6 Algorithm3 | `sesd/rag.py`、`knowledge_base.py`、`dialogue.py`；`test_sesd.py` | CLIP 向量检索与知识项送入真实 LLM，追踪证据；索引/编码器、词义表及安全守卫是 implementation-assumption，不能把检索命中当成语义正确率。 |
| p7 Eq14、18 | `scoring/clip_reward.py`、`rl/rollout.py`；`test_reward.py`、`test_ppo.py` | 双原始余弦权重与一步 TD 的可执行表达；终止 mask 与具体 CLIP checkpoint 是实现选择。 |
| p7 Eq19–20 | `rl/replay.py`、`rl/ppo.py`、`rl/diffusion_batch.py`；`test_replay.py`、`test_ppo.py` | 按 TD 与经验年龄计算采样概率、重要性权重以及裁剪 PPO 损失；验证行为 log-prob、动作和状态。仅同一行为版本进入更新，陈旧轨迹保留供诊断；版本隔离、样本年龄时钟和不完全重要性修正系数属于 implementation-assumption。 |
| p8 Eq21–23、26–27 | `editing/controllers.py`、`editing/token_alignment.py`、`rl/mapper_alpha.py`；`test_attention.py`、`test_mapper_alpha.py` | 实际 U-Net 的 Replace/Refine 注意力注入；由对齐先验参数化并采样可学习 mapper，beta 混合保留目标新增 token。Eq26 维度不一致，对齐方向、采样分布和归一化均是 implementation-assumption。 |
| p8 Eq24–25、29 | `rl/mapper_alpha.py`、`rl/ppo.py`；`test_mapper_alpha.py` | PPO score-function 利用生成图奖励更新 mapper 参数和 beta 分布，记录旧联合 log-prob、梯度及参数变化。它不声称 CLIP 对 PIL/冻结扩散路径端到端反传；该梯度估计和参数化为 implementation-assumption。 |
| p8 Eq28、30 | `editing/controllers.py`；`test_attention.py` | Reweight 提供原式与安全可执行变体；原式可破坏概率非负性或行和，安全 softmax 重权是 implementation-assumption；逐 feature 的原作者梯度路径未披露。 |
| p8 Eq31–33 | `rl/state_action.py`、`rl/policy.py`、`editing/controllers.py`；`test_policy.py` | 自注意力注入、cross 时间比例和整数步数为策略动作；层和时段映射以及对整数步数的优化未唯一规定。固定预算对照中步数刻意固定，不应声称本轮验证了步数学习。 |
| p5–7 Eq3–13 | `scoring/mi.py` 与 `equation_audit.md` | 互信息表达及其前提经审计，Eq13 的下降符号与文字最大化有冲突。CLIP 代理不是条件互信息估计；真实条件 MI 优化仍未复现。 |
| p6 Algorithm1、最终闭环 | `runtime/pipeline.py`、`runtime/session.py`、`rl/diffusion_batch.py`；`test_runtime.py`、`test_mapper_alpha.py` | 真正连接对话、检索、动作、冻结扩散、奖励、回放、策略更新、历史及最终选择；冻结预训练 SD1.5 与原论文 Efficient U-Net 检查点不同。 |

核心实验请看本轮验收报告和实验目录；功能通过、真实模型低预算验证和论文级效果是三个不同的结论。既有 `equation_audit.md`、`paper_to_code_matrix.md`、`verification_report.md` 保留为此前的历史版本，发生冲突时本增补说明最新实际状态，不改写原始证据。
