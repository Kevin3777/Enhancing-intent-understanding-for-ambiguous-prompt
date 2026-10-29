# PDF公式/算法到源代码的对应

唯一方法依据：paper/main.pdf；p为PDF物理/印刷页码。原式及完整审查见equation_audit.md，当前执行原则见execution_principles.md。

|公式/位置|含义与输入输出|源码（src/vca内）|测试|性质与边界|
|---|---|---|---|---|
|Eq.(1), p4|输入原始请求、历次输出/反馈与系统指令，输出下一轮改写。|`sesd/dialogue.py::Dialogue.refine`|`test_sesd.py::test_multiturn`|PAPER_AMBIGUOUS；M/p_refine/x 的完整提示模板、上下文裁剪和停止条件未给出|
|Eq.(2), p5|只显式携带最近输出和反馈的简化递推。|`sesd/dialogue.py::Dialogue.refine`|`test_sesd.py::test_multiturn`|PAPER_AMBIGUOUS；与Eq1的全历史输入不同；实现保留全历史，构造最近状态，标为工程选择，不能断言二者等价|
|Eq.(3), p5|连续随机变量互信息定义。|`scoring/mi.py (theory only)`|`test_mi.py::test_not_clip_mi`|PAPER_EXACT；需联合密度、可积性和绝对连续性|
|Eq.(4), p5|单位范数图像和文本嵌入的原始余弦。|`scoring/clip_reward.py::CLIPReward`|`test_reward.py::test_cosine`|PAPER_EXACT；须同维、范数非零；不是logit_scale放大分数，也不是0-100 CLIPScore|
|Eq.(5), p5|交叉协方差秩一假设。|`scoring/mi.py (theory only)`|`test_mi.py::test_gaussian_domain`|PAPER_AMBIGUOUS；一般X/Y维数可不同，需u_X u_Y^T；同一个u要求同维且同方向|
|Eq.(6), p6|配对CLIP余弦的样本均值。|`scoring/clip_reward.py::cosine`|`test_reward.py::test_cosine`|PAPER_AMBIGUOUS；未中心化的点积均值不是一般意义的交叉协方差特征值或相关系数；不能由此证明Eq5|
|Eq.(7), p6|联合非退化高斯的互信息。|`scoring/mi.py::gaussian_mi`|`test_mi.py::test_gaussian_domain`|PAPER_EXACT；协方差必须正定；归一化嵌入位于球面，一般不严格联合高斯|
|Eq.(8), p6|Schur补行列式。|`scoring/mi.py::gaussian_mi`|`test_mi.py::test_gaussian_domain`|PAPER_EXACT；要求Sigma_X可逆，联合协方差块尺寸相容|
|Eq.(9), p6|各向同性假设。|`scoring/mi.py (theory only)`|`test_mi.py::test_gaussian_domain`|PAPER_AMBIGUOUS；不是CLIP预训练保证；本地不将该假设写成数据事实|
|Eq.(10), p6|重复Eq5秩一近似。|`scoring/mi.py (theory only)`|`test_mi.py::test_gaussian_domain`|PAPER_AMBIGUOUS；u范数未指定，C与sigma/u尺度有关；维度和中心化问题同Eq5|
|Eq.(11), p6|在同维D的各向同性、秩一条件下简化行列式。|`scoring/mi.py::rank_one_proxy`|`test_mi.py::test_gaussian_domain`|PAPER_AMBIGUOUS；需C=//u//^4/(sigma_x^2 sigma_y^2)，且C*rho^2<1；原文仅说C>0，未交代尺度/正定域|
|Eq.(12), p7|条件假设下的高斯MI近似。|`scoring/mi.py::rank_one_proxy`|`test_mi.py::test_gaussian_domain`|PAPER_AMBIGUOUS；仅在C*rho^2<1有效；关于/rho/递增，关于负rho并非递增|
|Eq.(13), p7|字面为条件MI的梯度下降。|`scoring/mi.py::update`|`test_mi.py::test_direction`|PAPER_AMBIGUOUS；PDF p7 Eq13减号与最大化叙述冲突|
|Eq.(14), p7|同一生成图对历史/新提示的双重奖励。|`scoring/clip_reward.py::dual_reward`|`test_reward.py::test_weights`|PAPER_EXACT；历史权重1，新权重0.2；与强调新意图的直觉不同但不得静默交换|
|Eq.(15), p7|控制器参数梯度上升。|`rl/ppo.py::PPO.update`|`test_ppo.py::test_updates_actor_critic`|PAPER_AMBIGUOUS；未解释离散动作、PIL图像和采样步骤的可微路径；本实现使用PPO score-function更新策略，不能称本式逐像素直接梯度精确复现|
|Eq.(16), p7|图像迭代更新抽象算子。|`editing/pipeline_adapter.py::PairDiffusion.generate`|`scripts/test_attention_effect.py`|PAPER_AMBIGUOUS；Upd未定义，未给真实图像反演方法|
|Eq.(17), p7|最大化折扣累积奖励。|`rl/state_action.py; runtime/pipeline.py`|`test_ppo.py; test_runtime.py`|PAPER_AMBIGUOUS；此段说r_i=CLIP(I_i,P_i)，与Eq14双奖励不完全相同|
|Eq.(18), p7|一步TD误差。|`rl/rollout.py::td_targets`|`test_ppo.py::test_done_mask`|PAPER_EXACT；原式省略终止mask；实现r+gamma*(1-done)*Vnext-V，critic target detach；不默认GAE|
|Eq.(19), p7|按经验年龄衰减和绝对TD误差的优先采样。|`rl/replay.py::DiagnosticReplay; rl/ppo.py::PPO.update`|`test_replay.py`|PAPER_AMBIGUOUS；训练实际使用q_i=exp(-lambda*age_i)*abs(TD_i)+eps，p_i=q_i/sum(q)，w_i=(N*p_i)^(-beta)|
|Eq.(20), p7|clipped PPO surrogate；rho=pi_new/pi_old；A为一步TD。|`rl/ppo.py::PPO.update`|`test_ppo.py`|PAPER_EXACT；以old_logprob.detach、exp(new-old)、clipped surrogate、critic bootstrap、done mask、KL/entropy监测和optimizer.step实施|
|Eq.(21), p8|阈值内替换注意力。|`editing/controllers.py::AttentionController`|`test_attention.py; scripts/test_attention_effect.py`|PAPER_AMBIGUOUS；PDF p8称M*为更新/梯度细化图，同时要求保留整体结构；未唯一规定来源|
|Eq.(22), p8|更新映射矩阵后乘cross-attention。|`editing/token_alignment.py::align`|`test_alignment.py::test_mapping_shape`|PAPER_AMBIGUOUS；M/mapper二义，DeltaM来源、矩阵维度和归一化缺失|
|Eq.(23), p8|新插入token用目标图，匹配token用来源图。|`editing/controllers.py; editing/token_alignment.py`|`test_alignment.py; test_attention.py`|PAPER_AMBIGUOUS；A是离散token对应，如何梯度上升未定义|
|Eq.(24), p8|mapper参数奖励梯度上升。|`docs/reproduction_scope.md (not reproduced)`|`test_alignment.py`|PAPER_AMBIGUOUS；参数化、动作采样、梯度估计器、训练目标未给；离散token mapper固定对齐不等于本式训练|
|Eq.(25), p8|混合系数参数更新。|`rl/state_action.py; rl/policy.py`|`test_policy.py`|PAPER_AMBIGUOUS；未说明与Eq27 beta、Eq31 alpha的关系|
|Eq.(26), p8|原文以三指标乘法后permute四个维度。|`editing/token_alignment.py::TokenAlignment.project`|`test_alignment.py::test_einsum`|PAPER_AMBIGUOUS；三维/四维矛盾、k同时为空间或token不明、无显式求和|
|Eq.(27), p8|映射后来源/目标attention逐层逐步混合。|`editing/controllers.py`|`test_attention.py::test_identity`|PAPER_AMBIGUOUS；orig先对齐到目标token轴；beta约束[0,1]未给，工程加界；新token保留目标|
|Eq.(28), p8|指定token后softmax权重缩放。|`editing/controllers.py::literal_reweight; safe_reweight`|`test_attention.py::test_negative_literal`|PAPER_AMBIGUOUS；c<0破坏非负性，c!=1一般破坏行和|
|Eq.(29), p8|先混合，再用奖励梯度更新beta。|`editing/controllers.py; rl/policy.py`|`test_attention.py; test_policy.py`|PAPER_AMBIGUOUS；右侧对beta_t求导但左侧也beta_t，隐式/更新时序不清；gamma也用于discount|
|Eq.(30), p8|按feature权重组合attention。|`editing/controllers.py; docs/reproduction_scope.md`|`test_attention.py`|PAPER_AMBIGUOUS；i是token/head/feature未定义，约束、归一化及梯度路径缺失|
|Eq.(31), p8|自注意力注入比例更新。|`editing/controllers.py; rl/state_action.py`|`test_attention.py::test_self_injection`|PAPER_AMBIGUOUS；注入层、时段、比例边界未给|
|Eq.(32), p8|cross-attention替换比例更新。|`editing/controllers.py; rl/state_action.py`|`test_attention.py`|PAPER_AMBIGUOUS；以0..1进度门控和混合强度工程映射；策略分布间接优化，非直接对PIL反传；与Eq25/27同名不同义风险记入配置|
|Eq.(33), p8|采样步数更新。|`rl/state_action.py; rl/policy.py`|`test_policy.py`|PAPER_AMBIGUOUS；步数为整数，不能普通微分；delta又与TD误差同名|
|Fig2/3,p3-4; Algorithm1,p6|prompt/history -> retrieval/action/image/reward/next history|runtime/pipeline.py; runtime/session.py|test_runtime.py|ENGINEERING_CHOICE: append H、p_current更新、best-of、stop、KB来源追踪|
|Algorithm2,p6; Table1,p9|condition/ambiguity check，current_prompt+user_input -> new_prompt/type|sesd/dialogue.py; sesd/prompt_schema.py|test_sesd.py|ENGINEERING_CHOICE: bounded rounds/mock/provider接口；非ChatGPT-4原始权重|
|Algorithm3,p6|按知识分类检索/增补/调整/替换|sesd/rag.py; sesd/knowledge_base.py|test_sesd.py|implementation-assumption:实际运行CLIP文本向量余弦检索；离线TF-IDF后端，不声称与原编码器相同|
|Table A7-A11,p17-18|Style/Enhancers/Viewpoint/Atmosphere/Polysemy知识|data/knowledge.json|test_sesd.py|PAPER_EXACT子集+Layout工程项；Mouse来自Fig1/4，spring water来自Fig1，附录spring含coil/season，保留差异|
|4.1,p8|4×4090/3 days/12000 episodes/4 epochs/256/5e-5|configs/model.yaml; docs/experiments.md|scripts/train_ppo.py|NOT_REPRODUCED全量；预训练U-Net Adam1e-4/1e4 warmup/1000 noise steps与PPO区分|
|4.2,p8-9; Fig6/7|1673 JSON、约3000轮、95/5、30评审者|scripts/audit_dataset.py|test_dataset.py|NOT_REPRODUCED原始split/调查；禁止用图像数当对话数|
|p10-17,Table2-6|CLIP/LPIPS/SSIM/FID/rounds/satisfaction/ablations|scoring/metrics.py; scripts/eval_ablation.py|test_metrics.py|小样本仅local measurement；真实调查缺席=null，FID不足显式拒绝|


## 实际核心连接

runtime/pipeline.py::VCARuntime.step将Dialogue.refine、向量RAG、PairDiffusion.generate、AttentionController、CLIPReward.score、TD/Eq19 priority sampling、PPO.update、历史/经验池/知识记忆回写直接连接。finalize对同一最终提示词重评所有active history图像。

真实模型验收：scripts/test_attention_effect.py；真实数据训练：scripts/train_ppo.py --env diffusion --catalog data/catalog.jsonl；完整三轮：scripts/interactive_demo.py --fixture；保存结果与checkpoint验证：tests/integration/test_saved_evidence.py及scripts/verify_evidence.py。

固定token mapper不冒充Eq24直接梯度训练；CLIP proxy不冒充Eq13真实条件MI；原始Efficient U-Net/LLM微调权重未取得。必要实现假设均保留边界，不以完整工程闭环运行推定未知数学细节已精确复原。


## 实际核心连接

runtime/pipeline.py::VCARuntime.step将Dialogue.refine、向量RAG、PairDiffusion.generate、AttentionController、CLIPReward.score、TD/Eq19 priority sampling、PPO.update、历史/经验池/知识记忆回写直接连接。finalize对同一最终提示词重评所有active history图像。

真实模型验收：scripts/test_attention_effect.py；真实数据训练：scripts/train_ppo.py --env diffusion --catalog data/catalog.jsonl；完整三轮：scripts/interactive_demo.py --fixture；保存结果与checkpoint验证：tests/integration/test_saved_evidence.py及scripts/verify_evidence.py。

固定token mapper不冒充Eq24直接梯度训练；CLIP proxy不冒充Eq13真实条件MI；原始Efficient U-Net/LLM微调权重未取得。必要实现假设均保留边界，不以完整工程闭环运行推定未知数学细节已精确复原。
